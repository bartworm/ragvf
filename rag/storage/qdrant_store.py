import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rag.models import TextChunk, TableDescriptor, TableSummary
from dotenv import load_dotenv

class QdrantVectorStore:
    def __init__(self, path: str, embedding_dim: int = 384):
        load_dotenv()
        self.client = QdrantClient(path=str(path))
        self.embedding_dim = embedding_dim
        
        # Nombres de colecciones
        self.COLLECTION_TEXT = os.getenv("COLLECTION_TEXT_CHUNKS", "text_chunks")
        self.COLLECTION_TABLE_DESCRIPTORS = os.getenv("COLLECTION_TABLE_DESCRIPTORS", "table_descriptors")
        self.COLLECTION_TABLE_SUMMARIES = os.getenv("COLLECTION_TABLE_SUMMARIES", "table_summaries")

    def setup_collections(self):
        """Crea las colecciones si no existen."""
        collections = [
            self.COLLECTION_TEXT,
            self.COLLECTION_TABLE_DESCRIPTORS,
            self.COLLECTION_TABLE_SUMMARIES
        ]
        
        existing = {c.name for c in self.client.get_collections().collections}
        
        for col in collections:
            if col not in existing:
                self.client.create_collection(
                    collection_name=col,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                print(f" Colección creada: {col}")

    def _build_doc_id_filter(self, doc_ids: Optional[List[str]]) -> Optional[models.Filter]:
        """Ayudante para construir el filtro de Qdrant."""
        if not doc_ids:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=doc_ids)
                )
            ]
        )

    def _to_uuid(self, id_str: str) -> str:
        """Convierte cualquier string a un UUID determinista válido."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))

    def upsert_text_chunks(self, chunks: List[TextChunk], embedding_function):
        if not chunks: return
        
        points = []
        for chunk in chunks:
            vector = embedding_function(chunk.content)
            
            # Convertimos el ID string a UUID válido
            point_uuid = self._to_uuid(chunk.chunk_id)
            
            payload = {
                "chunk_id": chunk.chunk_id, # Guardamos el ID original en payload
                "doc_id": chunk.doc_id,
                "page": chunk.page,
                "content": chunk.content,
                "type": "text"
            }
            points.append(models.PointStruct(
                id=point_uuid, # <--- AQUI USAMOS EL UUID
                vector=vector,
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.COLLECTION_TEXT,
            points=points
        )

    def search_text(self, query_vector: List[float], limit: int = 5, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        query_filter = self._build_doc_id_filter(doc_ids)
        results = self.client.search(
            collection_name=self.COLLECTION_TEXT,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        return [{"score": r.score, "payload": r.payload} for r in results]

    def upsert_table_descriptors(self, descriptors: List[TableDescriptor], embedding_function):
        if not descriptors: return

        points = []
        for desc in descriptors:
            # Construir texto para embedding con información estructural
            headers_str = ', '.join(desc.headers) if desc.headers else ''
            text_to_embed = f"{desc.description} Columnas: {headers_str}"
            vector = embedding_function(text_to_embed)

            point_uuid = self._to_uuid(desc.descriptor_id)

            payload = {
                "descriptor_id": desc.descriptor_id,
                "table_id": desc.table_id,
                "doc_id": desc.doc_id,
                "page": desc.page,
                "description": desc.description,
                "num_rows": desc.num_rows,
                "num_cols": desc.num_cols,
                "headers": desc.headers,
                "column_types": desc.column_types,
                "type": "table_descriptor"
            }
            points.append(models.PointStruct(
                id=point_uuid, # <--- UUID
                vector=vector,
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.COLLECTION_TABLE_DESCRIPTORS,
            points=points
        )

    def search_table_descriptors(self, query_vector: List[float], limit: int = 5, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        query_filter = self._build_doc_id_filter(doc_ids)
        results = self.client.search(
            collection_name=self.COLLECTION_TABLE_DESCRIPTORS,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        return [{"score": r.score, "payload": r.payload} for r in results]

    def upsert_table_summaries(self, summaries: List[TableSummary], embedding_function):
        if not summaries: return
        
        points = []
        for summ in summaries:
            vector = embedding_function(summ.summary_text)
            
            point_uuid = self._to_uuid(summ.summary_id)
            
            payload = {
                "summary_id": summ.summary_id,
                "table_id": summ.table_id,
                "doc_id": summ.doc_id,
                "page": summ.page,
                "summary_text": summ.summary_text,
                "type": "table_summary"
            }
            points.append(models.PointStruct(
                id=point_uuid, # <--- UUID
                vector=vector,
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.COLLECTION_TABLE_SUMMARIES,
            points=points
        )

    def search_table_summaries(self, query_vector: List[float], limit: int = 5, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        query_filter = self._build_doc_id_filter(doc_ids)
        results = self.client.search(
            collection_name=self.COLLECTION_TABLE_SUMMARIES,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        return [{"score": r.score, "payload": r.payload} for r in results]