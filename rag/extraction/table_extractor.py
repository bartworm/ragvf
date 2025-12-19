"""
NIVEL 3: Table Extractor - Extracción de tablas con Camelot
Hereda de LayoutExtractor y añade extracción de tablas.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import camelot
import sys
from rag.models import TextChunk, FullTable, TableRow
from rag.extraction.layout_extractor import LayoutExtractor




class TableExtractor(LayoutExtractor):
    """
    Extractor con soporte para tablas.
    Usa Camelot con estrategia híbrida (Lattice -> Stream).
    """
    
    def __init__(
        self,
        min_accuracy: float = 10.0,  # Baja exigencia para detectar más tablas
        min_rows: int = 2,
        flavor: str = "lattice",     # Preferencia inicial
        detect_sections: bool = False
    ):
        super().__init__(detect_sections=detect_sections)
        self.name = "TableExtractor"
        self.min_accuracy = min_accuracy
        self.min_rows = min_rows
        self.flavor = flavor
    
    def extract_tables_from_pdf(
            self,
            pdf_path: Path,
            doc_id: str,
            pages: str = "all"
        ) -> List[FullTable]:
            """Extrae tablas usando estrategia híbrida Lattice->Stream."""
            # print(f"  Extrayendo tablas (Lattice -> Stream)...")

            tables = []
            used_flavor = "lattice"

            # Intento con Lattice (tablas con bordes)
            try:
                lattice_tables = camelot.read_pdf(
                    str(pdf_path),
                    flavor="lattice",
                    pages=pages,
                    suppress_stdout=True
                )

                high_quality_tables = [t for t in lattice_tables if t.accuracy >= self.min_accuracy]

                if len(high_quality_tables) > 0:
                    # print(f"   Lattice detectó {len(lattice_tables)} tablas ({len(high_quality_tables)} con buena calidad).")
                    tables = lattice_tables
                else:
                    print(f"  Lattice detectó {len(lattice_tables)} candidatos pero eran ruido (accuracy < {self.min_accuracy}). Descartando.")
                    tables = []

            except Exception as e:
                print(f"  Warning: Error en Lattice: {e}")
                tables = []

            # Fallback a Stream si Lattice falló
            if len(tables) == 0:
                print("  Activando modo 'stream' (para tablas sin bordes)...")
                try:
                    tables = camelot.read_pdf(
                        str(pdf_path),
                        flavor="stream",
                        pages=pages,
                        suppress_stdout=True
                    )
                    used_flavor = "stream"
                except Exception as e:
                    print(f"  Warning: Error en Stream: {e}")
                    tables = []
            
            if len(tables) == 0:
                print("  No se detectaron tablas con ningún método.")
                return []

            # 3. PROCESAMIENTO Y VALIDACIÓN FINAL
            tables_extracted = []
            for i, table in enumerate(tables):
                # Filtrar por calidad
                if table.accuracy < self.min_accuracy:
                    print(f"  Warning: Tabla {i} descartada (accuracy: {table.accuracy:.1f}%)")
                    continue
                
                # Filtrar por tamaño (filas mínimas)
                if len(table.df) < self.min_rows:
                    print(f"  Warning: Tabla {i} descartada (solo {len(table.df)} filas)")
                    continue
                
                # Convertir a FullTable
                try:
                    full_table = self._camelot_to_full_table(
                        table, doc_id, pdf_path.name, i, used_flavor
                    )
                    if full_table is not None:
                        tables_extracted.append(full_table)
                        print(f"   Tabla {i} ({used_flavor}): {full_table.num_rows()}x{full_table.num_cols()} en pág {table.page} (acc: {table.accuracy:.1f}%)")
                except Exception as e:
                    print(f"  Warning: Error validando tabla {i}: {e}")
                    continue
            
            print(f"   Total extraídas: {len(tables_extracted)} tablas válidas")
            return tables_extracted
    def _camelot_to_full_table(
        self,
        camelot_table,
        doc_id: str,
        source_file: str,
        table_idx: int,
        flavor_used: str
    ) -> FullTable:
        """
        Convierte objeto de Camelot a FullTable validado.
        """
        df = camelot_table.df
        page = camelot_table.page
        
        # Headers: primera fila
        headers = df.iloc[0].tolist()
        headers = [str(h).strip() or f"col_{i}" for i, h in enumerate(headers)]
        
        # Desduplicar headers
        seen = {}
        unique_headers = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                unique_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique_headers.append(h)
        
        # Rows: resto del dataframe
        rows_data = []
        for idx in range(1, len(df)):
            row_raw_values = df.iloc[idx].tolist()
            
            # Crear diccionario {Header: Valor}
            row_dict = {}
            # Usamos zip para asegurar que no falle si longitudes difieren levemente
            for header, val in zip(unique_headers, row_raw_values):
                clean_val = str(val).strip().replace('\n', ' ')
                row_dict[header] = clean_val
            
            row = TableRow(
                row_id=f"{doc_id}_table_{table_idx}_row_{idx}",
                values=row_dict 
            )
            rows_data.append(row)
        
        # Caption
        caption = getattr(camelot_table, 'caption', None) or f"Table {table_idx + 1}"

        # Validar que la tabla no esté vacía
        if not rows_data:
            print(f"  Warning: Tabla {table_idx} descartada: 0 filas válidas")
            return None

        table_id = f"{doc_id}_table_{table_idx}"

        full_table = FullTable(
            table_id=table_id,
            doc_id=doc_id,
            page=page,
            headers=unique_headers,
            rows=rows_data,
            caption=caption,
            source_file=source_file,
            accuracy=camelot_table.accuracy,
            extraction_method=f"camelot_{flavor_used}" # Pasamos el método real usado
        )

        return full_table
    
    def process_pdf(
        self,
        pdf_path: Path,
        doc_id: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        use_layout_detection: bool = True,
        use_title_detection: bool = True,
        use_heuristics_fallback: bool = True,
        extract_tables: bool = True,
        table_pages: str = "all"
    ) -> Tuple[List[TextChunk], List[TextChunk], List[FullTable], dict]:
        """Orquestador principal del Nivel 3"""
        print(f"[NIVEL 3] Procesando {pdf_path.name} con extracción de tablas")
        
        # 1. Extraer texto y bibliografía (heredado de nivel 2)
        content_chunks, bib_chunks, text_metadata = super().process_pdf(
            pdf_path=pdf_path,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_layout_detection=use_layout_detection,
            use_title_detection=use_title_detection,
            use_heuristics_fallback=use_heuristics_fallback
        )
        
        # 2. Extraer tablas (Estrategia Híbrida)
        tables = []
        flavor_used = "none"
        if extract_tables:
            tables = self.extract_tables_from_pdf(
                pdf_path=pdf_path,
                doc_id=doc_id,
                pages=table_pages
            )
            if tables:
                # Tomamos el método de la primera tabla como referencia
                flavor_used = tables[0].extraction_method
        
        # 3. Metadata
        metadata = {
            **text_metadata,
            "level": 3,
            "extractor": self.name,
            "num_tables": len(tables),
            "table_flavor_used": flavor_used,
            "features": text_metadata.get("features", []) + ["table_extraction"]
        }
        
        print(f"✓ {len(content_chunks)} chunks de contenido")
        print(f"✓ {len(bib_chunks)} chunks de bibliografía")
        print(f"✓ {len(tables)} tablas extraídas")
        
        return content_chunks, bib_chunks, tables, metadata

# ========== TESTING ==========
if __name__ == "__main__":
    extractor = TableExtractor(
        min_accuracy=80.0,
        min_rows=2,
        flavor="lattice"
    )
    
    test_pdf = Path("/mnt/user-data/uploads/cf6896f938cab26391a2541c2ecada647fd2.pdf")
    
    if test_pdf.exists():
        content, bib, tables, metadata = extractor.process_pdf(
            pdf_path=test_pdf,
            doc_id="test_doc_001",
            use_layout_detection=True,
            use_title_detection=True,
            extract_tables=True
        )
        
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        if tables:
            print(f"\n📋 Primera tabla:")
            print(f"  - ID: {tables[0].table_id}")
            print(f"  - Página: {tables[0].page}")
            print(f"  - Dimensiones: {tables[0].num_rows()}x{tables[0].num_cols()}")
            print(f"  - Headers: {tables[0].headers[:5]}...")
            print(f"  - Caption: {tables[0].caption}")
            print(f"  - Accuracy: {tables[0].accuracy:.1f}%")
    else:
        print(f"⚠️ PDF no encontrado: {test_pdf}")
