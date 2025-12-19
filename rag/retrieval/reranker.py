"""
Reranker: Reordena resultados usando modelos cross-encoder.

Los cross-encoders son más precisos que bi-encoders (embeddings) porque:
- Procesan query + document juntos
- Capturan interacciones entre términos
- Mejor para ranking final

Trade-off: Más lentos (no se pueden pre-computar)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import CrossEncoder


@dataclass
class RerankedResult:
    """Resultado después de reranking."""
    original_result: any  # RetrievalResult original
    rerank_score: float
    original_rank: int
    new_rank: int


class Reranker:
    """
    Reordena resultados usando cross-encoder.
    
    Flujo:
    1. Retrieval inicial (bi-encoder) → top-20 rápido
    2. Reranking (cross-encoder) → top-10 preciso
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Modelo cross-encoder de HuggingFace
                Options:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (rápido, 80MB)
                - "cross-encoder/ms-marco-electra-base" (preciso, 400MB)
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print(f"✓ Reranker cargado: {model_name}")
    
    def rerank(
        self,
        query: str,
        results: List[any],  # List[RetrievalResult]
        top_k: int = 10,
        content_field: str = "content"
    ) -> List[any]:
        """
        Reordena resultados usando cross-encoder.
        
        Args:
            query: Query del usuario
            results: Lista de resultados del retrieval inicial
            top_k: Número de resultados a retornar después de reranking
            content_field: Campo del objeto result que contiene el texto
        
        Returns:
            Lista reordenada de top-k resultados
        
        Example:
            >>> retriever = TwoStageRetriever(...)
            >>> reranker = Reranker()
            >>> 
            >>> # Retrieval inicial (top-20)
            >>> initial_results, _ = retriever.retrieve(query, k_light=20)
            >>> 
            >>> # Reranking (top-20 → top-10)
            >>> final_results = reranker.rerank(query, initial_results, top_k=10)
        """
        if not results:
            return []
        
        if len(results) <= top_k:
            # Si hay menos resultados que top_k, retornar todos sin reranking
            return results
        
        # Preparar pares (query, document)
        pairs = []
        for result in results:
            # Extraer contenido del resultado
            if hasattr(result, content_field):
                content = getattr(result, content_field)
            elif isinstance(result, dict) and content_field in result:
                content = result[content_field]
            else:
                # Fallback: convertir a string
                content = str(result)
            
            pairs.append((query, content))
        
        # Calcular scores con cross-encoder
        scores = self.model.predict(pairs)
        
        # Combinar resultados con scores
        scored_results = list(zip(results, scores, range(len(results))))
        
        # Ordenar por score descendente
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top-k
        reranked = [result for result, score, orig_rank in scored_results[:top_k]]
        
        return reranked
    
    def rerank_with_metadata(
        self,
        query: str,
        results: List[any],
        top_k: int = 10,
        content_field: str = "content"
    ) -> Tuple[List[any], List[RerankedResult]]:
        """
        Reranking con metadata detallada.
        
        Returns:
            (reranked_results, metadata_list)
            
            metadata_list contiene información de ranking para cada resultado:
            - original_rank: Posición antes de reranking
            - new_rank: Posición después de reranking
            - rerank_score: Score del cross-encoder
            - score_delta: Cambio de score
        """
        if not results:
            return [], []
        
        # Preparar pares
        pairs = []
        for result in results:
            if hasattr(result, content_field):
                content = getattr(result, content_field)
            elif isinstance(result, dict):
                content = result.get(content_field, str(result))
            else:
                content = str(result)
            pairs.append((query, content))
        
        # Calcular scores
        scores = self.model.predict(pairs)
        
        # Crear lista con metadata
        scored_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            scored_results.append({
                "result": result,
                "rerank_score": float(score),
                "original_rank": i + 1,
                "original_score": getattr(result, 'score', 0.0) if hasattr(result, 'score') else 0.0
            })
        
        # Ordenar por rerank_score
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Asignar nuevos ranks
        for i, item in enumerate(scored_results[:top_k]):
            item["new_rank"] = i + 1
        
        # Separar resultados y metadata
        reranked_results = [item["result"] for item in scored_results[:top_k]]
        
        metadata_list = [
            RerankedResult(
                original_result=item["result"],
                rerank_score=item["rerank_score"],
                original_rank=item["original_rank"],
                new_rank=item["new_rank"]
            )
            for item in scored_results[:top_k]
        ]
        
        return reranked_results, metadata_list
    
    def explain_reranking(
        self,
        query: str,
        results: List[any],
        top_k: int = 10,
        content_field: str = "content"
    ) -> str:
        """
        Genera explicación del reranking para debugging.
        
        Returns:
            String con tabla comparativa antes/después
        """
        _, metadata = self.rerank_with_metadata(query, results, top_k, content_field)
        
        if not metadata:
            return "No hay resultados para reranking"
        
        explanation = [
            f"Reranking para query: '{query}'",
            f"Modelo: {self.model_name}",
            "",
            "Cambios en ranking:",
            "-" * 80,
            f"{'Rank Orig':<12} {'Rank Nuevo':<12} {'Score':<12} {'Movimiento':<12} {'Doc ID':<30}",
            "-" * 80
        ]
        
        for meta in metadata:
            movement = meta.new_rank - meta.original_rank
            movement_str = f"↑{abs(movement)}" if movement < 0 else f"↓{abs(movement)}" if movement > 0 else "="
            
            doc_id = getattr(meta.original_result, 'doc_id', 'N/A')[:28]
            
            explanation.append(
                f"{meta.original_rank:<12} {meta.new_rank:<12} "
                f"{meta.rerank_score:<12.4f} {movement_str:<12} {doc_id:<30}"
            )
        
        return "\n".join(explanation)


class HybridReranker:
    """
    Combina múltiples señales para reranking:
    - Cross-encoder score
    - Score original del retriever
    - Metadata (tipo de fuente, recency, etc.)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_weight: float = 0.7,
        original_score_weight: float = 0.2,
        metadata_weight: float = 0.1
    ):
        """
        Args:
            cross_encoder_weight: Peso del score del cross-encoder
            original_score_weight: Peso del score original (bi-encoder)
            metadata_weight: Peso de metadata (tipo de fuente, etc.)
        """
        self.cross_encoder = Reranker(model_name)
        self.cross_encoder_weight = cross_encoder_weight
        self.original_score_weight = original_score_weight
        self.metadata_weight = metadata_weight
    
    def rerank(
        self,
        query: str,
        results: List[any],
        top_k: int = 10,
        content_field: str = "content"
    ) -> List[any]:
        """
        Reranking híbrido combinando múltiples señales.
        """
        if not results:
            return []
        
        # Obtener scores del cross-encoder
        pairs = [(query, getattr(r, content_field, str(r))) for r in results]
        cross_scores = self.cross_encoder.model.predict(pairs)
        
        # Normalizar scores
        cross_scores_norm = self._normalize_scores(cross_scores)
        
        # Obtener scores originales
        original_scores = [getattr(r, 'score', 0.5) for r in results]
        original_scores_norm = self._normalize_scores(original_scores)
        
        # Calcular scores de metadata
        metadata_scores = [self._metadata_score(r) for r in results]
        metadata_scores_norm = self._normalize_scores(metadata_scores)
        
        # Combinar scores
        final_scores = []
        for i in range(len(results)):
            combined = (
                self.cross_encoder_weight * cross_scores_norm[i] +
                self.original_score_weight * original_scores_norm[i] +
                self.metadata_weight * metadata_scores_norm[i]
            )
            final_scores.append(combined)
        
        # Ordenar
        scored = list(zip(results, final_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [r for r, s in scored[:top_k]]
    
    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        """Normaliza scores a rango [0, 1]."""
        scores_array = np.array(scores)
        if scores_array.max() == scores_array.min():
            return np.ones_like(scores_array) * 0.5
        return (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
    
    def _metadata_score(self, result: any) -> float:
        """
        Calcula score basado en metadata.
        
        Heurísticas:
        - full_table: +0.2 (tablas completas son valiosas)
        - text: 0.0 (neutral)
        - bibliography: -0.5 (penalizar bibliografía)
        """
        if hasattr(result, 'source_type'):
            if result.source_type == "full_table":
                return 1.0
            elif result.source_type == "table_descriptor":
                return 0.8
            elif result.source_type == "bibliography":
                return 0.0
            else:
                return 0.5
        return 0.5


# Ejemplo de uso
if __name__ == "__main__":
    from two_stage_retriever import TwoStageRetriever, RetrievalResult
    
    # Simular resultados
    mock_results = [
        RetrievalResult(
            source_type="text",
            content="Benzoato de sodio es efectivo a pH bajo",
            metadata={},
            score=0.85,
            doc_id="doc1",
            page=1
        ),
        RetrievalResult(
            source_type="text",
            content="El extracto de clavo es una alternativa natural",
            metadata={},
            score=0.80,
            doc_id="doc2",
            page=5
        ),
        RetrievalResult(
            source_type="bibliography",
            content="Smith et al. (2020). Natural preservatives.",
            metadata={},
            score=0.75,
            doc_id="doc3",
            page=20
        ),
    ]
    
    # Reranking
    reranker = Reranker()
    query = "Alternativa natural a benzoato"
    
    reranked = reranker.rerank(query, mock_results, top_k=2)
    
    print("Resultados después de reranking:")
    for i, r in enumerate(reranked, 1):
        print(f"{i}. {r.doc_id}: {r.content[:50]}...")
    
    # Explicación detallada
    print("\n" + reranker.explain_reranking(query, mock_results, top_k=3))
