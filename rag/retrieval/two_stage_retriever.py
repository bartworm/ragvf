"""
Retrieval en dos etapas para manejo eficiente de tablas.

Etapa 1: Recuperación ligera (descriptores y resúmenes)
Etapa 2: Carga diferida de tablas completas solo cuando sea necesario

Este enfoque evita sobrecargar el contexto con tablas completas
que pueden no ser relevantes.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.persistence import ParquetPersistence
from rag.models import FullTable, RAGResponse # Aseguramos compatibilidad

@dataclass
class RetrievalResult:
    """Resultado unificado de retrieval."""
    source_type: str  # 'text', 'table_descriptor', 'table_summary', 'full_table'
    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str
    page: int
    
    def __repr__(self):
        return f"<{self.source_type} | score={self.score:.3f} | {self.content[:60]}...>"


class TwoStageRetriever:
    """
    Retriever en dos etapas para optimizar retrieval de tablas.
    """
    
    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        parquet_storage: ParquetPersistence,
        embedding_function,  # Callable[[str], List[float]]
        full_table_threshold: float = 0.75,
        k_light: int = 20,
        k_full_tables: int = 3
    ):
        self.qdrant = qdrant_store
        self.storage = parquet_storage
        self.embed = embedding_function
        self.full_table_threshold = full_table_threshold
        self.k_light = k_light
        self.k_full_tables = k_full_tables
    
    def retrieve(
        self,
        query: str,
        include_full_tables: bool = True,
        doc_ids: Optional[List[str]] = None,
        k_light: Optional[int] = None,
        text_only: bool = False,  # Para baseline: solo chunks de texto
        **kwargs
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """Recuperación en dos etapas."""
        # Usar el k pasado por argumento, o el default de la clase
        limit_search = k_light if k_light is not None else self.k_light
        
        metadata = {
            "query": query,
            "k_light": limit_search,
            "full_table_threshold": self.full_table_threshold,
            "stages": {}
        }

        query_vector = self.embed(query)
        
        # Buscar en colecciones
        text_results = self.qdrant.search_text(
            query_vector,
            limit=limit_search,
            doc_ids=doc_ids
        )

        # Si text_only=True (baseline), no buscar en tablas
        if text_only:
            descriptor_results = []
            summary_results = []
        else:
            descriptor_results = self.qdrant.search_table_descriptors(
                query_vector,
                limit=limit_search,
                doc_ids=doc_ids
            )

            summary_results = self.qdrant.search_table_summaries(
                query_vector,
                limit=limit_search,
                doc_ids=doc_ids
            )
        
        metadata["stages"]["light_retrieval"] = {
            "text_results": len(text_results),
            "descriptor_results": len(descriptor_results),
            "summary_results": len(summary_results)
        }
        
        # Convertir a RetrievalResult
        results = []
        
        for r in text_results:
            results.append(RetrievalResult(
                source_type="text",
                content=r["payload"]["content"],
                metadata=r["payload"],
                score=r["score"],
                doc_id=r["payload"]["doc_id"],
                page=r["payload"]["page"]
            ))
        
        for r in descriptor_results:
            results.append(RetrievalResult(
                source_type="table_descriptor",
                content=r["payload"]["description"],
                metadata=r["payload"],
                score=r["score"],
                doc_id=r["payload"]["doc_id"],
                page=r["payload"]["page"]
            ))
        
        for r in summary_results:
            results.append(RetrievalResult(
                source_type="table_summary",
                content=r["payload"]["summary_text"],
                metadata=r["payload"],
                score=r["score"],
                doc_id=r["payload"]["doc_id"],
                page=r["payload"]["page"]
            ))

        full_tables_loaded = []
        
        if include_full_tables:
            # Identificar descriptores altamente relevantes
            high_score_descriptors = [
                r for r in descriptor_results 
                if r["score"] >= self.full_table_threshold
            ][:self.k_full_tables]
            
            metadata["stages"]["full_table_loading"] = {
                "candidates": len(high_score_descriptors),
                "threshold": self.full_table_threshold,
                "loaded": 0
            }
            
            # Cargar tablas completas desde Parquet
            for desc in high_score_descriptors:
                table_id = desc["payload"]["table_id"]
                
                try:
                    full_table = self.storage.load_table(table_id)
                    
                    if full_table:
                        # Convertir tabla a texto para contexto
                        table_text = self._table_to_text(full_table)
                        
                        results.append(RetrievalResult(
                            source_type="full_table",
                            content=table_text,
                            metadata={
                                "table_id": table_id,
                                "doc_id": full_table.doc_id,
                                "page": full_table.page,
                                "num_rows": full_table.num_rows(),
                                "num_cols": full_table.num_cols(),
                                "headers": full_table.headers
                            },
                            score=desc["score"],
                            doc_id=full_table.doc_id,
                            page=full_table.page
                        ))
                        
                        full_tables_loaded.append(table_id)
                        metadata["stages"]["full_table_loading"]["loaded"] += 1
                
                except Exception as e:
                    print(f"⚠️ Error cargando tabla {table_id}: {e}")
                    continue
            
            metadata["stages"]["full_table_loading"]["table_ids"] = full_tables_loaded

        results.sort(key=lambda r: r.score, reverse=True)
        
        metadata["total_results"] = len(results)
        
        return results, metadata
    
    def _table_to_text(self, table: FullTable) -> str:
        """
        Convierte tabla completa a representación textual.
        """
        lines = ["[TABLA COMPLETA]"]
        if table.caption:
            lines.append(f"Título: {table.caption}")
        lines.append(f"Fuente: {table.source_file}, página {table.page}")
        lines.append("")
        
        # Headers
        lines.append(" | ".join(table.headers))
        lines.append("-" * (len(table.headers) * 15))
        
        # Rows (limitar a 30 para no saturar contexto)
        max_rows = min(30, len(table.rows))
        for row in table.rows[:max_rows]:
            # Obtener valores en orden de headers, manejando Nones
            vals = [str(row.values.get(h, "")) for h in table.headers]
            lines.append(" | ".join(vals))
        
        if len(table.rows) > max_rows:
            lines.append(f"... ({len(table.rows) - max_rows} filas más)")
        
        return "\n".join(lines)