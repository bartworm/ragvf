"""
Wrapper para el retriever del Baseline (Hito 1).

Este módulo permite usar tu vectorstore de Chroma existente
en el benchmark de forma compatible con las otras configuraciones.

INSTRUCCIONES:
1. Ejecuta tu notebook actual (proyecto_1_hito_2.ipynb) hasta crear el vectorstore
2. Guarda el vectorstore de Chroma en una ubicación conocida
3. Este módulo lo cargará para el benchmark
"""

from pathlib import Path
from typing import List, Any, Dict
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.retrieval.two_stage_retriever import RetrievalResult


class BaselineRetriever:
    """
    Wrapper que adapta el retriever de Chroma (Hito 1) 
    a la interfaz esperada por el benchmark.
    """
    
    def __init__(self, vectorstore: Chroma, retriever_type: str = "similarity"):
        """
        Args:
            vectorstore: Instancia de Chroma vectorstore
            retriever_type: 'similarity', 'mmr', 'hybrid', etc.
        """
        self.vectorstore = vectorstore
        self.retriever_type = retriever_type
        
        # Configurar retriever según tipo
        if retriever_type == "mmr":
            self.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 20}
            )
        elif retriever_type == "similarity":
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
        else:
            # Por defecto similarity
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    def retrieve(
        self, 
        query: str,
        include_full_tables: bool = False,  # Ignorado en baseline
        include_bibliography: bool = False,  # Ignorado en baseline
        doc_ids: List[str] = None  # Ignorado en baseline
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Realiza retrieval usando el vectorstore de Chroma.
        
        Args:
            query: Pregunta del usuario
            include_full_tables: Ignorado (baseline no tiene esta funcionalidad)
            include_bibliography: Ignorado (baseline incluye todo)
            doc_ids: Ignorado (baseline no filtra por doc_id)
        
        Returns:
            (results, metadata) en formato compatible con benchmark
        """
        # Llamar al retriever de Chroma
        try:
            chroma_results = self.retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"⚠️ Error en retrieval baseline: {e}")
            chroma_results = []
        
        # Convertir a formato RetrievalResult
        results = []
        for i, doc in enumerate(chroma_results):
            # Extraer metadata
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # Detectar si es chunk de bibliografía (heurística)
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            is_bibliography = self._looks_like_bibliography(content)
            
            result = RetrievalResult(
                source_type="bibliography" if is_bibliography else "text",
                content=content,
                metadata=metadata,
                score=0.0,  # Chroma similarity_search no devuelve score por defecto
                doc_id=metadata.get("doc_id", metadata.get("source", "unknown")),
                page=metadata.get("page", metadata.get("page_number", 0))
            )
            results.append(result)
        
        # Metadata del retrieval
        metadata = {
            "query": query,
            "retriever_type": self.retriever_type,
            "total_results": len(results),
            "result_types": {
                "text": sum(1 for r in results if r.source_type == "text"),
                "bibliography": sum(1 for r in results if r.source_type == "bibliography")
            },
            "k_light": 10,
            "full_table_threshold": None,  # No aplica en baseline
            "stages": {
                "light_retrieval": {
                    "text_results": len(results)
                }
            }
        }
        
        return results, metadata
    
    def _looks_like_bibliography(self, text: str) -> bool:
        """
        Heurística para detectar si un chunk parece bibliografía.

        Esto permite medir contaminación por bibliografía incluso en baseline.
        """
        # Buscar patrones típicos de referencias
        year_pattern = r'\((\d{4})\)'
        year_matches = re.findall(year_pattern, text)
        
        # Si hay >5 años en el chunk, probablemente es bibliografía
        if len(year_matches) > 5:
            return True
        
        # Buscar patrones de citación
        citation_patterns = [
            r'[A-Z][a-z]+,\s+[A-Z]\.\s*\(\d{4}\)',  # Smith, J. (2020)
            r'[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)',  # Smith et al. (2020)
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, text):
                return True
        
        return False


def load_baseline_vectorstore(chroma_dir: Path = None) -> Chroma:
    """
    Carga el vectorstore de Chroma del baseline (Hito 1).
    
    Args:
        chroma_dir: Ruta al directorio de Chroma. Si es None, usa ruta por defecto.
    
    Returns:
        Instancia de Chroma vectorstore
    
    IMPORTANTE: Debes haber ejecutado tu notebook actual y guardado el vectorstore primero.
    """
    if chroma_dir is None:
        # Ruta por defecto (ajustar según tu configuración)
        chroma_dir = Path("/content/proyecto_aplicado_preservantes/chroma_preservantes")
    
    # Cargar embedding model (debe ser el mismo que usaste en Hito 1)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Cargar vectorstore
    try:
        vectorstore = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embedding_model
        )
        print(f"✓ Vectorstore baseline cargado desde: {chroma_dir}")
        print(f"  - Documentos en vectorstore: {vectorstore._collection.count()}")
        return vectorstore
    
    except Exception as e:
        raise RuntimeError(
            f"Error cargando vectorstore baseline: {e}\n"
            f"Asegúrate de haber ejecutado tu notebook actual primero y que "
            f"el vectorstore esté en: {chroma_dir}"
        )


def create_baseline_retriever(
    chroma_dir: Path = None,
    retriever_type: str = "similarity"
) -> BaselineRetriever:
    """
    Crea retriever baseline completo.
    
    Args:
        chroma_dir: Ruta al directorio de Chroma
        retriever_type: Tipo de retriever ('similarity', 'mmr', etc.)
    
    Returns:
        BaselineRetriever listo para usar en benchmark
    """
    vectorstore = load_baseline_vectorstore(chroma_dir)
    return BaselineRetriever(vectorstore, retriever_type)


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar baseline
    try:
        retriever = create_baseline_retriever(
            chroma_dir=Path("/content/proyecto_aplicado_preservantes/chroma_preservantes"),
            retriever_type="similarity"
        )
        
        # Probar retrieval
        query = "¿Cuáles son los límites de benzoato?"
        results, metadata = retriever.retrieve(query)
        
        print(f"\nQuery: {query}")
        print(f"Resultados: {len(results)}")
        print(f"Tipos: {metadata['result_types']}")
        
        for i, r in enumerate(results[:3], 1):
            print(f"\n{i}. [{r.source_type}] {r.content[:100]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPara usar este módulo:")
        print("1. Ejecuta tu notebook actual (proyecto_1_hito_2.ipynb)")
        print("2. Asegúrate de que el vectorstore se guarde en chroma_preservantes/")
        print("3. Vuelve a ejecutar este script")
