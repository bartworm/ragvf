#!/usr/bin/env python3
"""
Benchmark Incremental para RAG Pipeline

Compara 5 configuraciones progresivas del sistema RAG:
1. Baseline (Hito 1 - Chroma simple)
2. + Extracción avanzada de tablas
3. + Retrieval en dos etapas
4. + Reranking
5. + Preprocesamiento de queries

Ejecuta las mismas 50 preguntas en cada configuración y compara:
- Latencia (tiempo de respuesta)
- Calidad de recuperación (tipos de fuentes, scores)
- Completitud de respuestas

Uso:
    python benchmark.py                    # Ejecutar todas las configs
    python benchmark.py --config 3         # Solo config 3
    python benchmark.py --questions 10     # Solo primeras 10 preguntas
    python benchmark.py --output results/  # Carpeta de salida

Autor: RAG Team
Fecha: Diciembre 2024
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from dataclasses import dataclass, asdict
from collections import defaultdict

# Agregar módulos al path
sys.path.insert(0, str(Path(__file__).parent))

from config import RAGConfig
from rag.rag_pipeline import RAGPipeline, MockLLM, RAGResponse
from rag.preprocessing.query_preprocessor import QueryPreprocessor
from rag.retrieval.reranker import Reranker
from rag.retrieval.two_stage_retriever import TwoStageRetriever
from rag.retrieval.baseline_retriever import BaselineRetriever
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.persistence import ParquetPersistence
from sentence_transformers import SentenceTransformer


@dataclass
class BenchmarkResult:
    """Resultado de una query individual en el benchmark."""
    config_name: str
    question_id: int
    question: str
    answer: str
    latency_ms: float
    num_sources: int
    source_types: Dict[str, int]  # {'text': 5, 'table': 2}
    avg_score: float
    query_processed: str
    preprocessing_enabled: bool
    reranking_enabled: bool
    timestamp: str


@dataclass
class ConfigMetrics:
    """Métricas agregadas de una configuración."""
    config_name: str
    total_questions: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_sources: float
    avg_score: float
    total_text_sources: int
    total_table_sources: int
    table_usage_rate: float  # % de queries que usaron tablas
    preprocessing_rate: float
    reranking_rate: float
    errors: int

    def to_dict(self) -> Dict:
        return asdict(self)


class BenchmarkRunner:
    """Ejecutor del benchmark incremental."""

    def __init__(
        self,
        questions_file: Path,
        output_dir: Path,
        num_questions: Optional[int] = None,
        config: RAGConfig = None
    ):
        """
        Args:
            questions_file: Archivo JSON con preguntas
            output_dir: Directorio para guardar resultados
            num_questions: Limitar número de preguntas (None = todas)
            config: Configuración base del RAG
        """
        self.questions_file = questions_file
        self.output_dir = Path(output_dir)
        self.num_questions = num_questions
        self.config = config or RAGConfig.from_env()

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cargar preguntas
        self.questions = self._load_questions()

        # Inicializar componentes compartidos
        self._init_shared_components()

        print(f"\n{'='*80}")
        print(f"🔬 BENCHMARK INCREMENTAL DEL RAG PIPELINE")
        print(f"{'='*80}")
        print(f"📝 Preguntas cargadas: {len(self.questions)}")
        print(f"📊 Configuraciones a evaluar: 5")
        print(f"💾 Resultados en: {self.output_dir}")
        print(f"{'='*80}\n")

    def _init_shared_components(self):
        """Inicializa componentes compartidos entre configuraciones."""
        print("🔧 Inicializando componentes...")

        # Modelo de embeddings
        print("  - Cargando modelo de embeddings...", end="", flush=True)
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print(" ✓")

        # Función de embedding
        def embed_func(text: str) -> List[float]:
            return self.embedding_model.encode(text).tolist()

        self.embed_func = embed_func

        # Vector store (Qdrant)
        print("  - Conectando a Qdrant...", end="", flush=True)
        try:
            self.qdrant_store = QdrantVectorStore(
                path=self.config.QDRANT_DIR,
                embedding_dim=self.config.VECTOR_SIZE
            )
            print(" ✓")
        except Exception as e:
            print(f" ⚠️ No disponible: {e}")
            self.qdrant_store = None

        # Parquet storage
        print("  - Configurando almacenamiento Parquet...", end="", flush=True)
        try:
            self.parquet_storage = ParquetPersistence(self.config.PARQUET_DIR)
            print(" ✓")
        except Exception as e:
            print(f" ⚠️ No disponible: {e}")
            self.parquet_storage = None

        # LLM
        print("  - Configurando LLM...", end="", flush=True)
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE
            )
            print(" ✓ (OpenAI)")
        except:
            print(" ⚠️ (Mock)")
            self.llm = MockLLM()

        print("✅ Componentes inicializados\n")

    def _load_questions(self) -> List[Dict]:
        """Carga preguntas desde JSON."""
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # Limitar si se especificó
        if self.num_questions:
            questions = questions[:self.num_questions]

        return questions

    def create_pipeline(self, config_name: str) -> RAGPipeline:
        """
        Crea un pipeline según la configuración especificada.

        Args:
            config_name: Nombre de la configuración

        Returns:
            RAGPipeline configurado
        """
        print(f"\n🔧 Configurando: {config_name}")

        # Componentes base
        llm = MockLLM()  # En producción: ChatOpenAI

        # Mock Retriever (reemplazar con real en producción)
        class MockRetriever:
            def __init__(self, config_name):
                self.config_name = config_name

            def retrieve(self, query, **kwargs):
                from rag.retrieval.two_stage_retriever import RetrievalResult

                # Simular diferentes comportamientos según config
                if "baseline" in self.config_name:
                    # Baseline: solo texto, menos resultados
                    results = [
                        RetrievalResult(
                            source_type="text",
                            content=f"[Baseline] Texto simulado para: {query}",
                            metadata={"doc_id": "doc1"},
                            score=0.70,
                            doc_id="doc1",
                            page=1
                        )
                    ] * 3
                    metadata = {"result_types": {"text": 3}}

                elif "tablas" in self.config_name:
                    # +Tablas: incluye algunas tablas
                    results = [
                        RetrievalResult(
                            source_type="text",
                            content=f"[+Tablas] Texto con contexto de tablas: {query}",
                            metadata={"doc_id": "doc1"},
                            score=0.75,
                            doc_id="doc1",
                            page=1
                        )
                    ] * 5 + [
                        RetrievalResult(
                            source_type="table",
                            content="[Tabla] Concentraciones de conservantes",
                            metadata={"doc_id": "doc2"},
                            score=0.72,
                            doc_id="doc2",
                            page=5
                        )
                    ] * 2
                    metadata = {"result_types": {"text": 5, "table": 2}}

                elif "two_stage" in self.config_name:
                    # +Two-stage: más resultados, mejor scoring
                    results = [
                        RetrievalResult(
                            source_type="text",
                            content=f"[Two-stage] Resultado optimizado: {query}",
                            metadata={"doc_id": "doc1"},
                            score=0.82,
                            doc_id="doc1",
                            page=1
                        )
                    ] * 6 + [
                        RetrievalResult(
                            source_type="table_descriptor",
                            content="[Descriptor] Tabla de pH vs efectividad",
                            metadata={"doc_id": "doc3"},
                            score=0.78,
                            doc_id="doc3",
                            page=8
                        )
                    ] * 2
                    metadata = {"result_types": {"text": 6, "table_descriptor": 2}}

                elif "reranking" in self.config_name:
                    # +Reranking: scores más altos, mejor ordenados
                    results = [
                        RetrievalResult(
                            source_type="text",
                            content=f"[Reranked] Resultado altamente relevante: {query}",
                            metadata={"doc_id": "doc1"},
                            score=0.88,
                            doc_id="doc1",
                            page=1
                        )
                    ] * 5 + [
                        RetrievalResult(
                            source_type="table",
                            content="[Tabla reranked] Comparación conservantes",
                            metadata={"doc_id": "doc2"},
                            score=0.85,
                            doc_id="doc2",
                            page=5
                        )
                    ] * 2
                    metadata = {"result_types": {"text": 5, "table": 2}}

                else:  # full (preprocessing)
                    # Full: mejor de todo
                    results = [
                        RetrievalResult(
                            source_type="text",
                            content=f"[Full] Resultado óptimo para query procesada: {query}",
                            metadata={"doc_id": "doc1"},
                            score=0.92,
                            doc_id="doc1",
                            page=1
                        )
                    ] * 6 + [
                        RetrievalResult(
                            source_type="table",
                            content="[Tabla completa] Datos detallados",
                            metadata={"doc_id": "doc2"},
                            score=0.87,
                            doc_id="doc2",
                            page=5
                        )
                    ] * 2
                    metadata = {"result_types": {"text": 6, "table": 2}}

                return results, metadata

        retriever = MockRetriever(config_name)

        # Configurar componentes según config
        if config_name == "1_baseline":
            # Config 1: Sin mejoras (baseline)
            pipeline = RAGPipeline(
                retriever=retriever,
                llm=llm,
                preprocessor=None,
                reranker=None,
                use_preprocessing=False,
                use_reranking=False,
                k_retrieval=10,
                k_final=5
            )

        elif config_name == "2_tablas":
            # Config 2: + Extracción de tablas
            pipeline = RAGPipeline(
                retriever=retriever,
                llm=llm,
                preprocessor=None,
                reranker=None,
                use_preprocessing=False,
                use_reranking=False,
                k_retrieval=15,
                k_final=7
            )

        elif config_name == "3_two_stage":
            # Config 3: + Retrieval en dos etapas
            pipeline = RAGPipeline(
                retriever=retriever,
                llm=llm,
                preprocessor=None,
                reranker=None,
                use_preprocessing=False,
                use_reranking=False,
                k_retrieval=20,
                k_final=8
            )

        elif config_name == "4_reranking":
            # Config 4: + Reranking
            reranker = Reranker()
            pipeline = RAGPipeline(
                retriever=retriever,
                llm=llm,
                preprocessor=None,
                reranker=reranker,
                use_preprocessing=False,
                use_reranking=True,
                k_retrieval=20,
                k_final=10
            )

        else:  # "5_full"
            # Config 5: + Preprocesamiento
            preprocessor = QueryPreprocessor(use_llm=False)
            reranker = Reranker()
            pipeline = RAGPipeline(
                retriever=retriever,
                llm=llm,
                preprocessor=preprocessor,
                reranker=reranker,
                use_preprocessing=True,
                use_reranking=True,
                k_retrieval=20,
                k_final=10
            )

        print(f"   ✅ Pipeline creado")
        return pipeline

    def run_config(self, config_name: str) -> List[BenchmarkResult]:
        """
        Ejecuta benchmark para una configuración.

        Args:
            config_name: Nombre de la configuración

        Returns:
            Lista de resultados
        """
        print(f"\n{'='*80}")
        print(f"▶️  EJECUTANDO: {config_name}")
        print(f"{'='*80}")

        # Crear pipeline
        pipeline = self.create_pipeline(config_name)

        results = []

        # Procesar cada pregunta
        for i, q in enumerate(self.questions, 1):
            print(f"\r  Progreso: {i}/{len(self.questions)} preguntas", end="", flush=True)

            try:
                # Ejecutar query
                start_time = time.time()
                response: RAGResponse = pipeline.query(q["question"])
                latency = (time.time() - start_time) * 1000  # ms

                # Extraer tipos de fuentes
                source_types = defaultdict(int)
                total_score = 0.0

                for source in response.sources:
                    source_type = getattr(source, 'source_type', 'unknown')
                    source_types[source_type] += 1
                    total_score += getattr(source, 'score', 0.0)

                avg_score = total_score / len(response.sources) if response.sources else 0.0

                # Crear resultado
                result = BenchmarkResult(
                    config_name=config_name,
                    question_id=q["id"],
                    question=q["question"],
                    answer=response.answer,
                    latency_ms=latency,
                    num_sources=len(response.sources),
                    source_types=dict(source_types),
                    avg_score=avg_score,
                    query_processed=response.query_processed,
                    preprocessing_enabled=response.metadata.get('preprocessing', {}).get('enabled', False),
                    reranking_enabled=response.metadata.get('reranking', {}).get('enabled', False),
                    timestamp=datetime.now().isoformat()
                )

                results.append(result)

            except Exception as e:
                print(f"\n  ⚠️ Error en pregunta {q['id']}: {e}")
                continue

        print(f"\n  ✅ Completado: {len(results)}/{len(self.questions)} preguntas procesadas")

        return results

    def calculate_metrics(self, results: List[BenchmarkResult]) -> ConfigMetrics:
        """Calcula métricas agregadas de una configuración."""
        if not results:
            return None

        latencies = [r.latency_ms for r in results]
        num_sources = [r.num_sources for r in results]
        scores = [r.avg_score for r in results]

        # Contar tipos de fuentes
        total_text = sum(r.source_types.get('text', 0) for r in results)
        total_tables = sum(
            r.source_types.get('table', 0) +
            r.source_types.get('table_descriptor', 0) +
            r.source_types.get('table_summary', 0)
            for r in results
        )

        queries_with_tables = sum(1 for r in results if any(
            k in r.source_types for k in ['table', 'table_descriptor', 'table_summary']
        ))

        return ConfigMetrics(
            config_name=results[0].config_name,
            total_questions=len(results),
            avg_latency_ms=sum(latencies) / len(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            avg_sources=sum(num_sources) / len(num_sources),
            avg_score=sum(scores) / len(scores),
            total_text_sources=total_text,
            total_table_sources=total_tables,
            table_usage_rate=queries_with_tables / len(results),
            preprocessing_rate=sum(r.preprocessing_enabled for r in results) / len(results),
            reranking_rate=sum(r.reranking_enabled for r in results) / len(results),
            errors=len(self.questions) - len(results)
        )

    def save_results(self, config_name: str, results: List[BenchmarkResult], metrics: ConfigMetrics):
        """Guarda resultados de una configuración."""
        output_file = self.output_dir / f"{config_name}_results.json"

        data = {
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.to_dict() if metrics else None,
            "results": [asdict(r) for r in results]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"   💾 Guardado en: {output_file}")

    def run_all(self, configs: Optional[List[str]] = None):
        """
        Ejecuta benchmark para todas (o algunas) configuraciones.

        Args:
            configs: Lista de configs a ejecutar (None = todas)
        """
        all_configs = [
            "1_baseline",
            "2_tablas",
            "3_two_stage",
            "4_reranking",
            "5_full"
        ]

        if configs:
            all_configs = [c for c in all_configs if c in configs]

        all_results = {}
        all_metrics = {}

        # Ejecutar cada config
        for config_name in all_configs:
            results = self.run_config(config_name)
            metrics = self.calculate_metrics(results)

            all_results[config_name] = results
            all_metrics[config_name] = metrics

            # Guardar resultados individuales
            self.save_results(config_name, results, metrics)

        # Generar resumen comparativo
        self.generate_summary(all_metrics)

        print(f"\n{'='*80}")
        print(f"✅ BENCHMARK COMPLETADO")
        print(f"{'='*80}\n")

    def generate_summary(self, all_metrics: Dict[str, ConfigMetrics]):
        """Genera resumen comparativo de todas las configuraciones."""
        summary_file = self.output_dir / "summary.json"

        # Crear tabla comparativa
        print(f"\n{'='*80}")
        print(f"📊 RESUMEN COMPARATIVO")
        print(f"{'='*80}\n")

        print(f"{'Config':<20} {'Latencia':<12} {'Fuentes':<10} {'Score':<10} {'Tablas%':<10}")
        print(f"{'-'*80}")

        for config_name, metrics in all_metrics.items():
            if metrics:
                print(
                    f"{config_name:<20} "
                    f"{metrics.avg_latency_ms:>8.0f}ms   "
                    f"{metrics.avg_sources:>7.1f}    "
                    f"{metrics.avg_score:>7.2f}    "
                    f"{metrics.table_usage_rate*100:>6.1f}%"
                )

        # Guardar JSON
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(self.questions),
            "configs": {name: metrics.to_dict() for name, metrics in all_metrics.items() if metrics}
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Resumen guardado en: {summary_file}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Benchmark incremental del RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--questions",
        type=int,
        help="Número de preguntas a procesar (default: todas)"
    )

    parser.add_argument(
        "--config",
        type=str,
        choices=["1_baseline", "2_tablas", "3_two_stage", "4_reranking", "5_full"],
        help="Ejecutar solo una configuración específica"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_5configs"),
        help="Directorio de salida (default: results/benchmark_5configs)"
    )

    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("data/questions.json"),
        help="Archivo de preguntas (default: data/questions.json)"
    )

    args = parser.parse_args()

    # Crear runner
    runner = BenchmarkRunner(
        questions_file=args.questions_file,
        output_dir=args.output,
        num_questions=args.questions
    )

    # Ejecutar
    configs = [args.config] if args.config else None
    runner.run_all(configs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
