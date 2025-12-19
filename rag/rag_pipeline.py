"""
Pipeline principal RAG.
Orquesta la extracción, ingesta, recuperación y generación.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Imports internos
from config import RAGConfig
from rag.models import RAGResponse, TextChunk, FullTable
# Importar desde pipeline específico para evitar ciclos
from rag.extraction.pipeline import UnifiedPDFExtractor, ExtractionLevel
from rag.extraction.table_representations import TableRepresentationGenerator
from rag.preprocessing.query_preprocessor import QueryPreprocessor
from rag.retrieval.reranker import Reranker
from rag.retrieval.two_stage_retriever import TwoStageRetriever

# LLM Interfaces
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

class MockLLM(BaseChatModel):
    """LLM Simulado para pruebas sin API Key."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatResult, ChatGeneration
        return ChatResult(generations=[ChatGeneration(message=SystemMessage(content="Respuesta simulada."))])
    
    @property
    def _llm_type(self): return "mock"

class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        retriever: TwoStageRetriever,
        llm: BaseChatModel,
        preprocessor: Optional[QueryPreprocessor] = None,
        reranker: Optional[Reranker] = None,
        use_preprocessing: bool = True,
        use_reranking: bool = True,
        k_retrieval: int = 20,
        k_final: int = 5,
        max_context_length: int = 8000,
        text_only: bool = False
    ):
        self.config = config
        self.retriever = retriever
        self.llm = llm
        self.preprocessor = preprocessor
        self.reranker = reranker
        self.use_preprocessing = use_preprocessing
        self.use_reranking = use_reranking
        self.k_retrieval = k_retrieval
        self.k_final = k_final
        self.max_context_length = max_context_length
        self.text_only = text_only
        
        # Componentes de extracción e ingesta
        self.extractor = UnifiedPDFExtractor(level=ExtractionLevel(config.EXTRACTION_LEVEL))
        self.table_gen = TableRepresentationGenerator(
            llm=llm,
            use_llm=config.USE_LLM_FOR_TABLE_DESCRIPTORS
        )

    def run_ingestion(self, pdf_path: Path):
        """
        Procesa un PDF: Extrae -> Genera Embeddings -> Guarda en Qdrant/Parquet
        """
        print(f"\n📄 Procesando: {pdf_path.name}")
        start_time = time.time()
        
        # 1. Extracción
        doc_id = pdf_path.name
        # Llamamos al extractor
        result = self.extractor.process_pdf(pdf_path, doc_id=doc_id)
        
        # --- AQUÍ ESTABA EL ERROR ---
        # El extractor devuelve: 'content_chunks', 'bibliography_chunks', 'tables'
        # Usamos .get() para ser ultra seguros y evitar KeyErrors
        
        content_chunks = result.get("content_chunks", [])
        # Si por alguna razón antigua viene como 'content', lo intentamos recuperar:
        if not content_chunks:
            content_chunks = result.get("content", [])
            
        biblio_chunks = result.get("bibliography_chunks", [])
        # Fallback para biblio
        if not biblio_chunks:
            biblio_chunks = result.get("bibliography", [])
            
        tables: List[FullTable] = result.get("tables", [])
        # ----------------------------
        
        print(f"   ✓ Extracción: {len(content_chunks)} textos | {len(tables)} tablas | {len(biblio_chunks)} biblio")

        # 2. Ingesta de TEXTO
        if content_chunks:
            print("   🧠 Indexando texto...", end="", flush=True)
            self.retriever.storage.save_text_chunks(content_chunks)
            self.retriever.qdrant.upsert_text_chunks(content_chunks, self.retriever.embed)
            print(" ✓")
            
        # 3. Ingesta de BIBLIOGRAFÍA
        if biblio_chunks:
            print("   📚 Indexando bibliografía...", end="", flush=True)
            self.retriever.storage.save_text_chunks(biblio_chunks)
            self.retriever.qdrant.upsert_text_chunks(biblio_chunks, self.retriever.embed)
            print(" ✓")

        # 4. Ingesta de TABLAS
        if tables:
            print("   📊 Procesando tablas...", end="", flush=True)
            for table in tables:
                # A. Guardar Tabla Completa
                self.retriever.storage.save_table(table)
                
                # B. Descriptor (LLM)
                try:
                    desc = self.table_gen.generate_descriptor(table)
                    self.retriever.qdrant.upsert_table_descriptors([desc], self.retriever.embed)
                except Exception as e:
                    print(f"\n   ⚠️ Error descr. tabla {table.table_id}: {e}")
                
                # C. Resumen (LLM)
                try:
                    summary = self.table_gen.generate_summary(table)
                    self.retriever.qdrant.upsert_table_summaries([summary], self.retriever.embed)
                except Exception as e:
                    print(f"\n   ⚠️ Error resumen tabla {table.table_id}: {e}")
                    
            print(" ✓")

        elapsed = time.time() - start_time
        print(f"   ✨ Archivo completado en {elapsed:.2f}s")

    def query(self, user_query: str) -> RAGResponse:
        """Flujo principal de consulta (RAG)."""
        start_time = time.time()
        metadata = {"steps": {}}
        
        # 1. Preprocesamiento
        processed_query = user_query
        if self.use_preprocessing and self.preprocessor:
            processed_query = self.preprocessor.preprocess(user_query)
            metadata["steps"]["preprocessing"] = {"original": user_query, "processed": processed_query}
            metadata["preprocessing"] = {"enabled": True}
        
        # 2. Retrieval
        retrieval_results, retrieval_meta = self.retriever.retrieve(
            processed_query,
            k_light=self.k_retrieval,
            text_only=self.text_only
        )
        metadata["steps"]["retrieval"] = retrieval_meta
        
        # 3. Reranking
        final_results = retrieval_results
        if self.use_reranking and self.reranker:
            final_results = self.reranker.rerank(processed_query, retrieval_results, top_k=self.k_final)
            metadata["reranking"] = {"enabled": True}
        else:
             final_results = retrieval_results[:self.k_final]
        
        # 4. Contexto
        context_str, context_meta = self._build_context(final_results)
        metadata["context"] = context_meta
        
        # 5. Generación
        answer = self._generate_answer(user_query, context_str)
        
        latency = (time.time() - start_time) * 1000
        
        return RAGResponse(
            query=user_query,
            answer=answer,
            sources=final_results,
            latency_ms=latency,
            metadata=metadata,
            query_processed=processed_query
        )

    def _build_context(self, results) -> tuple[str, dict]:
        context_parts = []
        current_len = 0
        for res in results:
            part = f"[{res.source_type.upper()}] (Fuente: {res.doc_id}, pág {res.page})\n{res.content}\n"
            if current_len + len(part) > self.max_context_length: break
            context_parts.append(part)
            current_len += len(part)
        context_str = "\n---\n".join(context_parts)
        return context_str, {"length_chars": len(context_str), "truncated": current_len > self.max_context_length}

    def _generate_answer(self, query: str, context: str) -> str:
        system_prompt = (
            "Eres un experto en ciencia de alimentos. Responde usando el contexto provisto. "
            "Si no sabes, dilo. Cita las fuentes.\n\nCONTEXTO:\n" + context
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        try:
            return self.llm.invoke(messages).content
        except Exception as e:
            return f"Error generando respuesta: {e}"