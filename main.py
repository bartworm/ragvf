"""
Punto de entrada principal para Ingesta de Documentos.
Uso: python main.py --index data/pdfs/
"""

import argparse
import sys
import torch
from pathlib import Path

# Imports del sistema
from config import RAGConfig
from rag.rag_pipeline import RAGPipeline, MockLLM
from rag.retrieval.two_stage_retriever import TwoStageRetriever
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.persistence import ParquetPersistence
from rag.preprocessing.query_preprocessor import QueryPreprocessor
from rag.retrieval.reranker import Reranker
from sentence_transformers import SentenceTransformer

# LangChain / LLMs
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

class RAGCLIApp:
    def __init__(self):
        self.config = RAGConfig.from_env()
        self.pipeline = None

    def initialize_pipeline(self) -> bool:
        """Inicializa todos los componentes del RAG (GPU enabled)."""
        try:
            print("\n" + "=" * 60)
            print(f"Inicializando RAG Pipeline ({self.config.LLM_PROVIDER.upper()})")
            print("=" * 60)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Hardware: {device.upper()}")
            if device == "cuda":
                print(f"  GPU: {torch.cuda.get_device_name(0)}")

            print("\nCargando almacenamiento...")
            storage = ParquetPersistence(storage_dir=self.config.PARQUET_DIR)
            qdrant = QdrantVectorStore(
                path=self.config.QDRANT_DIR,
                embedding_dim=self.config.VECTOR_SIZE
            )
            qdrant.setup_collections()

            print(f"Cargando embeddings ({self.config.EMBEDDING_MODEL})...", end="", flush=True)
            embed_model = SentenceTransformer(self.config.EMBEDDING_MODEL, device=device)
            def embedding_function(text):
                return embed_model.encode(text).tolist()
            print(" OK")

            print(f"Conectando LLM ({self.config.LLM_MODEL})...", end="", flush=True)
            if self.config.LLM_PROVIDER == "google":
                llm = ChatGoogleGenerativeAI(
                    model=self.config.LLM_MODEL,
                    google_api_key=self.config.GOOGLE_API_KEY,
                    temperature=self.config.LLM_TEMPERATURE,
                    timeout=60,
                    max_retries=3
                )
            elif self.config.LLM_PROVIDER == "openai":
                llm = ChatOpenAI(
                    api_key=self.config.OPENAI_API_KEY,
                    model=self.config.LLM_MODEL,
                    temperature=self.config.LLM_TEMPERATURE,
                    timeout=60,
                    max_retries=3
                )
            elif self.config.LLM_PROVIDER == "ollama":
                llm = ChatOllama(
                    model=self.config.LLM_MODEL,
                    temperature=self.config.LLM_TEMPERATURE
                )
            else:
                llm = MockLLM()
            print(" OK")

            print("Configurando retriever...", end="", flush=True)
            retriever = TwoStageRetriever(
                qdrant_store=qdrant,
                parquet_storage=storage,
                embedding_function=embedding_function,
                full_table_threshold=self.config.FULL_TABLE_THRESHOLD,
                k_light=self.config.K_RETRIEVAL,
                k_full_tables=self.config.MAX_FULL_TABLES
            )
            print(" OK")

            preprocessor = QueryPreprocessor(use_llm=False)

            print("Configurando reranker (GPU)...", end="", flush=True)
            reranker = Reranker(self.config.RERANKER_MODEL)
            print(" OK")
            self.pipeline = RAGPipeline(
                config=self.config,
                retriever=retriever,
                llm=llm,
                preprocessor=preprocessor,
                reranker=reranker
            )

            print("\nSistema listo")
            return True

        except Exception as e:
            print(f"\nError al inicializar: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("--index", type=str, help="Ruta a un PDF o carpeta de PDFs para indexar")
    args = parser.parse_args()

    app = RAGCLIApp()
    
    # Si no inicializa bien, salir
    if not app.initialize_pipeline():
        sys.exit(1)

    # Lógica de Ingesta (Arreglada para carpetas)
    if args.index:
        input_path = Path(args.index)
        
        if not input_path.exists():
            print(f"❌ La ruta no existe: {input_path}")
            return

        # A) Es una carpeta
        if input_path.is_dir():
            print(f"\n📂 Carpeta detectada: {input_path}")
            pdf_files = list(input_path.glob("*.pdf"))
            
            if not pdf_files:
                print("⚠️ No hay PDFs en la carpeta.")
                return

            print(f"📄 Encontrados {len(pdf_files)} archivos. Iniciando...")
            
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"\n--- [{i}/{len(pdf_files)}] Procesando: {pdf_file.name} ---")
                try:
                    app.pipeline.run_ingestion(pdf_file)
                except Exception as e:
                    print(f"❌ Error en {pdf_file.name}: {e}")

        # B) Es un archivo
        elif input_path.suffix.lower() == ".pdf":
            print(f"\n📄 Archivo único detectado: {input_path.name}")
            app.pipeline.run_ingestion(input_path)
        
        else:
            print("❌ El archivo indicado no es un PDF.")

if __name__ == "__main__":
    main()