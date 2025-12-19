"""
Benchmark Incremental - COMPARATIVA INTERNA
Compara qué tanto aporta cada módulo nuevo de tu sistema.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

from config import RAGConfig
from rag.rag_pipeline import RAGPipeline
from rag.models import RAGResponse
from rag.preprocessing.query_preprocessor import QueryPreprocessor
from rag.retrieval.reranker import Reranker
from rag.retrieval.two_stage_retriever import TwoStageRetriever
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.persistence import ParquetPersistence
from sentence_transformers import SentenceTransformer

# Configurar LLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.rag_pipeline import MockLLM

@dataclass
class BenchmarkResult:
    config_name: str
    question_id: int
    question: str
    answer: str
    latency_ms: float
    num_sources: int
    source_types: Dict[str, int]
    avg_score: float
    timestamp: str

@dataclass
class ConfigMetrics:
    config_name: str
    avg_latency: float
    avg_score: float
    avg_sources: float
    table_usage: float
    total_errors: int
    def to_dict(self): return asdict(self)

class BenchmarkRunner:
    def __init__(self, questions_file: Path, output_dir: Path, num_questions: Optional[int] = None):
        self.config = RAGConfig.from_env()
        self.questions_file = questions_file
        self.output_dir = output_dir
        self.num_questions = num_questions
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.questions = self._load_questions()
        self._init_shared_components()

    def _load_questions(self):
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            qs = json.load(f)
        return qs[:self.num_questions] if self.num_questions else qs

    def _init_shared_components(self):
        print("🔧 Cargando componentes compartidos...")
        # Embeddings
        self.embed_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.embed_func = lambda text: self.embed_model.encode(text).tolist()
        # Storage
        self.storage = ParquetPersistence(self.config.PARQUET_DIR)
        self.qdrant = QdrantVectorStore(path=self.config.QDRANT_DIR, embedding_dim=self.config.VECTOR_SIZE)
        # LLM
        if self.config.LLM_PROVIDER == "google":
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.LLM_MODEL, 
                google_api_key=self.config.GOOGLE_API_KEY, 
                temperature=0.0
            )
        elif self.config.LLM_PROVIDER == "openai":
            self.llm = ChatOpenAI(
                api_key=self.config.OPENAI_API_KEY, 
                model=self.config.LLM_MODEL, 
                temperature=0.0
            )
        else:
            self.llm = MockLLM()
        print("✅ Componentes listos.")

    def create_pipeline(self, config_name: str) -> RAGPipeline:
        """Configura el pipeline activando/desactivando módulos."""
        print(f"\n⚙️  Configurando: {config_name}")
        
        # Instanciar Retriever
        retriever = TwoStageRetriever(
            qdrant_store=self.qdrant,
            parquet_storage=self.storage,
            embedding_function=self.embed_func,
            full_table_threshold=self.config.FULL_TABLE_THRESHOLD,
            k_light=self.config.K_RETRIEVAL,
            k_full_tables=self.config.MAX_FULL_TABLES
        )

        pipeline_args = {
            "config": self.config,
            "retriever": retriever,
            "llm": self.llm,
            "preprocessor": None,
            "reranker": None,
            "use_preprocessing": False,
            "use_reranking": False,
            "k_retrieval": self.config.K_RETRIEVAL,
            "k_final": self.config.K_FINAL,
            "text_only": False
        }

        if config_name == "1_baseline":
            # NIVEL 1: TEXTO PLANO - Solo chunks de texto, sin tablas
            retriever.k_full_tables = 0
            pipeline_args["k_retrieval"] = 10
            pipeline_args["text_only"] = True  # Forzar solo texto

        elif config_name == "2_tablas":
            # NIVEL 2: ACTIVAMOS TABLAS
            # Dejamos que el retriever use su configuración normal (trae tablas)
            pipeline_args["k_retrieval"] = 15

        elif config_name == "3_two_stage":
            # NIVEL 3: MAYOR RECUPERACIÓN (Para dar espacio al filtrado)
            pipeline_args["k_retrieval"] = 30
            
        elif config_name == "4_reranking":
            # NIVEL 4: ACTIVAMOS RERANKER
            pipeline_args["k_retrieval"] = 30
            pipeline_args["reranker"] = Reranker(self.config.RERANKER_MODEL)
            pipeline_args["use_reranking"] = True
            
        elif config_name == "5_full":
            # NIVEL 5: FULL (Preprocessing + Reranker)
            pipeline_args["k_retrieval"] = 30
            pipeline_args["reranker"] = Reranker(self.config.RERANKER_MODEL)
            pipeline_args["use_reranking"] = True
            pipeline_args["preprocessor"] = QueryPreprocessor(use_llm=False)
            pipeline_args["use_preprocessing"] = True

        return RAGPipeline(**pipeline_args)

    def run(self, specific_config=None):
        configs = ["1_baseline", "2_tablas", "3_two_stage", "4_reranking", "5_full"]
        if specific_config: configs = [specific_config]
        
        all_metrics = {}

        for conf in configs:
            pipeline = self.create_pipeline(conf)
            results = []
            print(f"▶️  Procesando {len(self.questions)} preguntas...")
            
            for i, q_data in enumerate(self.questions):
                q_text = q_data.get("question", "")
                print(f"\r   {i+1}/{len(self.questions)}", end="", flush=True)
                
                try:
                    start = time.time()
                    resp = pipeline.query(q_text)
                    latency = (time.time() - start) * 1000
                    
                    src_types = defaultdict(int)
                    scores = []
                    for s in resp.sources:
                        stype = getattr(s, "source_type", "text")
                        src_types[stype] += 1
                        scores.append(getattr(s, "score", 0.0))
                    
                    results.append(BenchmarkResult(
                        config_name=conf,
                        question_id=q_data.get("id", i),
                        question=q_text,
                        answer=resp.answer,
                        latency_ms=latency,
                        num_sources=len(resp.sources),
                        source_types=dict(src_types),
                        avg_score=sum(scores)/len(scores) if scores else 0,
                        timestamp=datetime.now().isoformat()
                    ))
                except Exception as e:
                    print(f"\n   ❌ Error: {e}")

            self._save_results(conf, results)
            metrics = self._calc_metrics(conf, results)
            all_metrics[conf] = metrics.to_dict()
            print(f"\n   ✅ Latencia media: {metrics.avg_latency:.0f}ms | Tablas: {metrics.table_usage*100:.0f}%")

        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

    def _calc_metrics(self, conf, results) -> ConfigMetrics:
        if not results: return ConfigMetrics(conf, 0, 0, 0, 0, len(self.questions))
        lats = [r.latency_ms for r in results]
        scores = [r.avg_score for r in results]
        srcs = [r.num_sources for r in results]
        # Detectar uso de tablas
        table_usage = sum(1 for r in results if any(t in r.source_types for t in ["full_table", "table_descriptor"]))
        
        return ConfigMetrics(
            config_name=conf,
            avg_latency=sum(lats)/len(lats),
            avg_score=sum(scores)/len(scores),
            avg_sources=sum(srcs)/len(srcs),
            table_usage=table_usage/len(results),
            total_errors=len(self.questions) - len(results)
        )

    def _save_results(self, conf, results):
        fpath = self.output_dir / f"{conf}_results.json"
        data = [asdict(r) for r in results]
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        questions_file=Path("data/questions.json"),
        output_dir=Path("results/benchmark_real"),
        num_questions=args.questions
    )
    runner.run(args.config)