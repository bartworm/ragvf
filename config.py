"""
Configuración centralizada del RAG Pipeline.

Define todos los parámetros en un solo lugar.
Carga variables de entorno desde .env

Uso:
    from config import RAGConfig
    config = RAGConfig.from_env()
    
    # Acceder a parámetros
    print(config.pdf_dir)
    print(config.chunk_size)
"""

from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv


class RAGConfig:
    """Configuración centralizada del RAG."""
    
    # ==================== RUTAS ====================
    
    # Directorios de datos
    PDF_DIR: Path = Path("data/pdfs")
    QDRANT_DIR: Path = Path("data/qdrant")
    PARQUET_DIR: Path = Path("data/parquet")
    
    # Directorios de resultados
    RESULTS_DIR: Path = Path("results")
    BENCHMARK_DIR: Path = Path("results/benchmark_5configs")
    
    # ==================== EXTRACCIÓN ====================
    
    # Nivel de extracción (0-4)
    # 0: Texto plano
    # 1: + Bibliografía
    # 2: + Layout (columnas)
    # 3: + Tablas
    # 4: + Contexto tablas
    EXTRACTION_LEVEL: int = 3
    
    # Parámetros de chunking
    CHUNK_SIZE: int = 1000          # Caracteres por chunk
    CHUNK_OVERLAP: int = 150        # Solapamiento entre chunks
    
    # Detección de secciones
    DETECT_SECTIONS: bool = True
    
    # Extracción de tablas
    EXTRACT_TABLES: bool = True
    TABLE_PAGES: str = "all"        # "all" o lista de páginas [1, 2, 3]
    
    # ==================== PERSISTENCIA ====================
    
    # Parquet storage
    USE_PARQUET: bool = True
    PARQUET_COMPRESSION: str = "snappy"  # "snappy", "gzip", "brotli"
    
    # ==================== VECTOR STORE (QDRANT) ====================
    
    # Colecciones
    COLLECTION_TEXT_CHUNKS: str = "text_chunks"
    COLLECTION_TABLE_DESCRIPTORS: str = "table_descriptors"
    COLLECTION_TABLE_SUMMARIES: str = "table_summaries"
    
    # Vector size (depende del embedding model)
    VECTOR_SIZE: int = 384  # MiniLM-L6-v2 produce 384 dims
    
    # ==================== EMBEDDINGS ====================
    
    # Modelo de embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Batch size para embeddings
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ==================== RETRIEVAL ====================
    
    # Número de resultados iniciales
    K_RETRIEVAL: int = 20
    
    # Número de resultados finales (después de reranking)
    K_FINAL: int = 10
    
    # Umbral para cargar tablas completas
    FULL_TABLE_THRESHOLD: float = 0.75
    
    # Máximo de tablas completas a cargar
    MAX_FULL_TABLES: int = 3
    
    # ==================== QUERY PREPROCESSING ====================
    
    # Preprocesamiento de queries
    USE_QUERY_PREPROCESSING: bool = True
    
    # Corrección ortográfica
    USE_SPELLING_CORRECTION: bool = True
    
    # Reescritura de queries (requiere LLM)
    USE_QUERY_REWRITING: bool = True
    USE_LLM_FOR_REWRITING: bool = False  # Si False, usa heurísticas
    
    # Expansión de sinónimos
    USE_SYNONYM_EXPANSION: bool = True
    
    # ==================== RERANKING ====================
    
    # Habilitar reranking
    USE_RERANKING: bool = True
    
    # Modelo cross-encoder
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Batch size para reranking
    RERANKING_BATCH_SIZE: int = 32
    
    # ==================== LLM ====================

    # Proveedor (openai, anthropic, google, ollama, mock)
    LLM_PROVIDER: str = "ollama"  # openai, anthropic, google, ollama, mock

    # Modelo a usar
    LLM_MODEL: str = "llama3.2-vision"
    
    # Temperatura (0.0-1.0)
    LLM_TEMPERATURE: float = 0.3
    
    # Max tokens en respuesta
    LLM_MAX_TOKENS: int = 1024
    
    # Timeout (segundos)
    LLM_TIMEOUT: int = 30

    # Usar LLM para generación de descriptores de tablas (puede ser costoso en API calls)
    USE_LLM_FOR_TABLE_DESCRIPTORS: bool = False  # False = usa descriptores heurísticos
    
    # ==================== RAG PIPELINE ====================
    
    # Máximo de caracteres en contexto
    MAX_CONTEXT_LENGTH: int = 8000
    
    # Incluir fuentes en respuesta
    INCLUDE_SOURCES: bool = True
    
    # ==================== BENCHMARK ====================
    
    # Preguntas de benchmark
    BENCHMARK_QUESTIONS_FILE: Path = Path("data/questions.json")
    
    # Número de preguntas a evaluar
    BENCHMARK_NUM_QUESTIONS: int = 50
    
    # ==================== AMBIENTE ====================
    
    # Debug mode
    DEBUG: bool = False
    
    # Logging level
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # ==================== API KEYS (desde .env) ====================
    
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """
        Carga configuración desde variables de entorno.
        
        Lee .env si existe y carga credenciales.
        """
        load_dotenv()
        
        # Crear instancia con valores por defecto
        config = cls()
        
        # Sobreescribir con variables de entorno si existen
        config.PDF_DIR = Path(os.getenv("PDF_DIR", str(config.PDF_DIR)))
        config.QDRANT_DIR = Path(os.getenv("QDRANT_DIR", str(config.QDRANT_DIR)))
        config.PARQUET_DIR = Path(os.getenv("PARQUET_DIR", str(config.PARQUET_DIR)))
        
        config.EXTRACTION_LEVEL = int(os.getenv("EXTRACTION_LEVEL", config.EXTRACTION_LEVEL))
        config.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", config.CHUNK_SIZE))
        config.K_RETRIEVAL = int(os.getenv("K_RETRIEVAL", config.K_RETRIEVAL))
        config.K_FINAL = int(os.getenv("K_FINAL", config.K_FINAL))
        
        config.LLM_PROVIDER = os.getenv("LLM_PROVIDER", config.LLM_PROVIDER)
        config.LLM_MODEL = os.getenv("LLM_MODEL", config.LLM_MODEL)
        config.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", config.LLM_TEMPERATURE))
        
        # API Keys
        config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        config.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        config.GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
        
        config.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        
        return config
    
    @classmethod
    def validate_paths(cls, config: 'RAGConfig') -> bool:
        """
        Valida que existan los directorios necesarios.
        
        Returns:
            True si todo está bien, False si faltan directorios
        """
        required_dirs = [
            (config.PDF_DIR, "PDF_DIR"),
            (config.QDRANT_DIR, "QDRANT_DIR"),
            (config.PARQUET_DIR, "PARQUET_DIR"),
        ]
        
        missing = []
        for path, name in required_dirs:
            if not path.exists():
                missing.append(f"{name}: {path}")
        
        if missing:
            print("⚠️ Directorios faltantes:")
            for m in missing:
                print(f"  - {m}")
            print("\nCreando directorios...")
            for path, _ in required_dirs:
                path.mkdir(parents=True, exist_ok=True)
            print("✓ Directorios creados")
        
        return True
    
    def to_dict(self) -> dict:
        """Convierte configuración a diccionario."""
        return {
            k: v for k, v in self.__class__.__dict__.items()
            if not k.startswith('_') and k.isupper()
        }
    
    def __repr__(self) -> str:
        """Representación en string."""
        return f"RAGConfig(extraction_level={self.EXTRACTION_LEVEL}, k_retrieval={self.K_RETRIEVAL}, llm={self.LLM_MODEL})"


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    # Cargar configuración
    config = RAGConfig.from_env()
    
    print("=" * 80)
    print("CONFIGURACIÓN DEL RAG PIPELINE")
    print("=" * 80)
    
    # Mostrar parámetros principales
    print("\n📁 RUTAS:")
    print(f"  PDF_DIR: {config.PDF_DIR}")
    print(f"  QDRANT_DIR: {config.QDRANT_DIR}")
    print(f"  PARQUET_DIR: {config.PARQUET_DIR}")
    
    print("\n🔄 EXTRACCIÓN:")
    print(f"  Nivel: {config.EXTRACTION_LEVEL}")
    print(f"  Chunk size: {config.CHUNK_SIZE}")
    print(f"  Chunk overlap: {config.CHUNK_OVERLAP}")
    
    print("\n🔍 RETRIEVAL:")
    print(f"  K inicial: {config.K_RETRIEVAL}")
    print(f"  K final: {config.K_FINAL}")
    print(f"  Threshold para tablas: {config.FULL_TABLE_THRESHOLD}")
    
    print("\n🤖 LLM:")
    print(f"  Proveedor: {config.LLM_PROVIDER}")
    print(f"  Modelo: {config.LLM_MODEL}")
    print(f"  Temperatura: {config.LLM_TEMPERATURE}")
    
    print("\n✅ CARACTERÍSTICAS:")
    print(f"  Preprocesamiento queries: {config.USE_QUERY_PREPROCESSING}")
    print(f"  Reranking: {config.USE_RERANKING}")
    print(f"  Incluir fuentes: {config.INCLUDE_SOURCES}")
    
    # Validar directorios
    print("\n🔍 Validando directorios...")
    config.validate_paths(config)
    
    print("\n✓ Configuración cargada exitosamente")
    print(f"\n{config}")
