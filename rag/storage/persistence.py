"""Persistencia en Parquet para chunks de texto y tablas."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import json
import shutil

from rag.models import TextChunk, FullTable, ProcessingMetadata, TableRow


class ParquetPersistence:
    """Gestor de persistencia en formato Parquet para chunks y tablas."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.text_chunks_path = self.storage_dir / "text_chunks.parquet"
        self.tables_dir = self.storage_dir / "tables"
        self.metadata_path = self.storage_dir / "metadata.json"
        
        # Crear directorios
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

    def save_text_chunks(self, chunks: List[TextChunk], mode: str = "append"):
        """
        Guarda chunks de texto en Parquet.
        
        Args:
            chunks: Lista de TextChunk validados
            mode: 'append' (añadir) o 'overwrite' (reemplazar)
        """
        if not chunks:
            print("Warning: No hay chunks para guardar")
            return
        
        # Convertir a DataFrame
        records = [c.dict() for c in chunks]
        df = pd.DataFrame(records)
        
        # Convertir datetime a string para compatibilidad
        df['created_at'] = df['created_at'].astype(str)
        
        if mode == "append" and self.text_chunks_path.exists():
            # Leer existente y concatenar
            existing_df = pd.read_parquet(self.text_chunks_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
            # Deduplicar por chunk_id
            df = df.drop_duplicates(subset=['chunk_id'], keep='last')
        
        # Guardar con compresión
        df.to_parquet(
            self.text_chunks_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # print(f" Guardados {len(df)} chunks en {self.text_chunks_path}")
    
    def load_text_chunks(
        self,
        doc_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None
    ) -> List[TextChunk]:
        """Carga chunks de texto desde Parquet, opcionalmente filtrados."""
        if not self.text_chunks_path.exists():
            return []
        
        df = pd.read_parquet(self.text_chunks_path)
        
        # Filtros
        if doc_ids:
            df = df[df['doc_id'].isin(doc_ids)]
        
        if chunk_ids:
            df = df[df['chunk_id'].isin(chunk_ids)]
        
        # Reconstruir objetos Pydantic
        chunks = []
        for _, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                if isinstance(row_dict.get('created_at'), str):
                    row_dict['created_at'] = datetime.fromisoformat(row_dict['created_at'])
                chunk = TextChunk(**row_dict)
                chunks.append(chunk)
            except Exception as e:
                print(f"Warning: Error reconstruyendo chunk {row.get('chunk_id')}: {e}")
                continue
        
        return chunks
    
    def get_text_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de chunks almacenados."""
        if not self.text_chunks_path.exists():
            return {"total_chunks": 0}
        
        df = pd.read_parquet(self.text_chunks_path)
        
        return {
            "total_chunks": len(df),
            "unique_docs": df['doc_id'].nunique(),
            "avg_char_count": df['char_count'].mean(),
            "avg_word_count": df['word_count'].mean(),
            "docs": df['doc_id'].unique().tolist()
        }

    def save_table(self, table: FullTable):
        """Guarda una tabla completa en su propio archivo Parquet."""
        # Convertir tabla a formato pandas-friendly
        records = table.to_dict_records()
        df = pd.DataFrame(records)
        
        # Añadir metadata como atributos
        metadata = {
            "table_id": table.table_id,
            "doc_id": table.doc_id,
            "page": str(table.page),
            "caption": table.caption or "",
            "source_file": table.source_file,
            "extraction_method": table.extraction_method,
            "created_at": table.created_at.isoformat(),
            "num_rows": str(table.num_rows()),
            "num_cols": str(table.num_cols())
        }
        
        # Path: tables/table_id.parquet
        table_path = self.tables_dir / f"{table.table_id}.parquet"
        
        # Guardar con metadata
        table_pyarrow = pa.Table.from_pandas(df)
        table_pyarrow = table_pyarrow.replace_schema_metadata(metadata)
        
        pq.write_table(
            table_pyarrow,
            table_path,
            compression='snappy'
        )
        
        # print(f" Tabla guardada: {table_path}")
    
    def save_tables(self, tables: List[FullTable]):
        """Guarda múltiples tablas."""
        for table in tables:
            self.save_table(table)
    
    def load_table(self, table_id: str) -> Optional[FullTable]:
        """
        Carga una tabla específica por ID.
        
        Args:
            table_id: ID de la tabla
        
        Returns:
            FullTable reconstruido o None si no existe
        """
        table_path = self.tables_dir / f"{table_id}.parquet"
        
        if not table_path.exists():
            print(f"Warning: Tabla no encontrada: {table_id}")
            return None
        
        # Leer con metadata
        table_pyarrow = pq.read_table(table_path)
        df = table_pyarrow.to_pandas()
        metadata = table_pyarrow.schema.metadata
        
        headers = df.columns.tolist()
        rows = [TableRow(values=row.to_dict()) for _, row in df.iterrows()]

        caption_raw = metadata[b'caption'].decode()
        caption = caption_raw if caption_raw else None

        table = FullTable(
            table_id=metadata[b'table_id'].decode(),
            doc_id=metadata[b'doc_id'].decode(),
            page=int(metadata[b'page'].decode()),
            headers=headers,
            rows=rows,
            caption=caption,
            source_file=metadata[b'source_file'].decode(),
            extraction_method=metadata[b'extraction_method'].decode(),
            created_at=datetime.fromisoformat(metadata[b'created_at'].decode())
        )
        
        return table
    
    def load_tables(
        self, 
        doc_id: Optional[str] = None
    ) -> List[FullTable]:
        """Carga todas las tablas, opcionalmente filtradas por doc_id."""
        tables = []
        
        for table_file in self.tables_dir.glob("*.parquet"):
            # Leer metadata sin cargar datos completos
            table_pyarrow = pq.read_table(table_file)
            metadata = table_pyarrow.schema.metadata
            
            # Filtrar por doc_id si se especifica
            if doc_id and metadata[b'doc_id'].decode() != doc_id:
                continue
            
            # Cargar tabla completa
            table_id = metadata[b'table_id'].decode()
            table = self.load_table(table_id)
            
            if table:
                tables.append(table)
        
        return tables
    
    def get_table_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de tablas almacenadas."""
        table_files = list(self.tables_dir.glob("*.parquet"))
        
        if not table_files:
            return {"total_tables": 0}
        
        stats = {
            "total_tables": len(table_files),
            "tables": []
        }
        
        for table_file in table_files:
            table_pyarrow = pq.read_table(table_file)
            metadata = table_pyarrow.schema.metadata
            
            stats["tables"].append({
                "table_id": metadata[b'table_id'].decode(),
                "doc_id": metadata[b'doc_id'].decode(),
                "page": int(metadata[b'page'].decode()),
                "rows": int(metadata[b'num_rows'].decode()),
                "cols": int(metadata[b'num_cols'].decode()),
                "source": metadata[b'source_file'].decode()
            })
        
        return stats

    def save_metadata(self, metadata: ProcessingMetadata):
        """Guarda metadata de procesamiento en JSON."""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata.to_json_dict(), f, indent=2, default=str)
        
        print(f" Metadata guardado: {self.metadata_path}")
    
    def load_metadata(self) -> Optional[ProcessingMetadata]:
        """Carga metadata de procesamiento."""
        if not self.metadata_path.exists():
            return None
        
        with open(self.metadata_path, 'r') as f:
            data = json.load(f)
        
        return ProcessingMetadata(**data)

    def get_storage_info(self) -> Dict[str, Any]:
        """Obtiene información completa del almacenamiento."""
        text_stats = self.get_text_stats()
        table_stats = self.get_table_stats()
        metadata = self.load_metadata()
        
        return {
            "storage_dir": str(self.storage_dir),
            "text_chunks": text_stats,
            "tables": table_stats,
            "metadata": metadata.dict() if metadata else None
        }
    
    def clear_all(self):
        """Elimina todos los datos persistidos."""
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
            self.storage_dir.mkdir(parents=True)
            self.tables_dir.mkdir(exist_ok=True)
        
        print(" Storage limpiado")
