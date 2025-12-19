# 🔧 Guía: Integración de Datos Reales en el Benchmark

## ⚠️ Estado Actual

El benchmark actual (`benchmark.py`) usa **MockRetriever** que genera datos simulados. Para usar **datos reales de tus PDFs**, necesitas completar estos pasos:

---

## 📋 Paso 1: Indexar tus PDFs

Antes de ejecutar el benchmark con datos reales, debes indexar tus PDFs en Qdrant y Parquet.

### Crear script de indexación (`indexar_pdfs.py`):

```python
#!/usr/bin/env python3
"""
Script para indexar PDFs en Qdrant y Parquet.

Uso:
    python indexar_pdfs.py data/pdfs/
"""

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import RAGConfig
from rag.extraction.extractors import PDFContentExtractor
from rag.extraction.table_representations import TableRepresentationGenerator
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.persistence import ParquetPersistence

def main():
    # Cargar config
    config = RAGConfig.from_env()

    # Inicializar componentes
    print("🔧 Inicializando componentes...")

    # 1. Extractor de PDFs
    extractor = PDFContentExtractor()
    table_gen = TableRepresentationGenerator()

    # 2. Modelo de embeddings
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed_func(text):
        return embedding_model.encode(text).tolist()

    # 3. Qdrant
    qdrant = QdrantVectorStore(
        path=config.QDRANT_DIR,
        embedding_dim=config.VECTOR_SIZE
    )
    qdrant.setup_collections()

    # 4. Parquet
    parquet = ParquetPersistence(config.PARQUET_DIR)

    # Procesar cada PDF
    pdf_dir = Path("data/pdfs")
    pdfs = list(pdf_dir.glob("*.pdf"))

    print(f"\n📄 Procesando {len(pdfs)} PDFs...")

    for i, pdf_file in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_file.name}")

        try:
            # Extraer contenido
            text_chunks, tables = extractor.extract_from_pdf(pdf_file)

            # Generar representaciones de tablas
            descriptors = []
            summaries = []

            for table in tables:
                desc = table_gen.create_descriptor(table)
                summ = table_gen.create_summary(table)
                descriptors.append(desc)
                summaries.append(summ)

            # Indexar en Qdrant
            for chunk in text_chunks:
                embedding = embed_func(chunk.content)
                qdrant.index_text_chunk(chunk, embedding)

            for desc in descriptors:
                embedding = embed_func(desc.description)
                qdrant.index_table_descriptor(desc, embedding)

            for summ in summaries:
                embedding = embed_func(summ.summary_text)
                qdrant.index_table_summary(summ, embedding)

            # Guardar tablas completas en Parquet
            for table in tables:
                parquet.save_table(table)

            print(f"  ✅ {len(text_chunks)} chunks, {len(tables)} tablas")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print("\n✅ Indexación completada!")
    print(f"   📊 Qdrant: {config.QDRANT_DIR}")
    print(f"   💾 Parquet: {config.PARQUET_DIR}")


if __name__ == "__main__":
    main()
```

### Ejecutar indexación:

```bash
python indexar_pdfs.py
```

---

## 📋 Paso 2: Modificar benchmark.py

### 2.1 Ya están los imports correctos (líneas 40-48):

```python
from rag.retrieval.two_stage_retriever import TwoStageRetriever
from rag.retrieval.baseline_retriever import BaselineRetriever
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.persistence import ParquetPersistence
from sentence_transformers import SentenceTransformer
```

### 2.2 Ya está `_init_shared_components()` (líneas 123-173)

Esto carga:
- ✅ Modelo de embeddings
- ✅ Qdrant store
- ✅ Parquet storage
- ✅ LLM

### 2.3 Añadir método `_create_retriever()`:

**Insertar después de la línea 173** (después de `_init_shared_components`):

```python
    def _create_retriever(self, config_name: str):
        """
        Crea retriever según configuración.

        Args:
            config_name: Nombre de la configuración

        Returns:
            Retriever configurado
        """
        if "baseline" in config_name:
            # Config 1: Baseline con Chroma (si tienes un vectorstore de Chroma)
            # Nota: Necesitas cargar tu vectorstore de Chroma del Hito 1
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
                vectorstore = Chroma(
                    persist_directory=str(self.config.CHROMA_DIR),  # Añadir a config.py
                    embedding_function=embeddings
                )
                return BaselineRetriever(vectorstore)
            except:
                # Si no tienes Chroma, usa TwoStageRetriever básico
                return TwoStageRetriever(
                    qdrant_store=self.qdrant_store,
                    parquet_storage=self.parquet_storage,
                    embedding_function=self.embed_func,
                    k_light=10,
                    k_full_tables=0  # Sin tablas para baseline
                )

        else:
            # Configs 2-5: TwoStageRetriever con diferentes parámetros
            if "tablas" in config_name:
                k_light = 15
                k_full = 2
                threshold = 0.70
            elif "two_stage" in config_name:
                k_light = 20
                k_full = 3
                threshold = 0.75
            elif "reranking" in config_name or "full" in config_name:
                k_light = 20
                k_full = 3
                threshold = 0.75
            else:
                k_light = 20
                k_full = 3
                threshold = 0.75

            return TwoStageRetriever(
                qdrant_store=self.qdrant_store,
                parquet_storage=self.parquet_storage,
                embedding_function=self.embed_func,
                full_table_threshold=threshold,
                k_light=k_light,
                k_full_tables=k_full
            )
```

### 2.4 Reemplazar `create_pipeline()` (líneas 191-395):

**Eliminar TODO el código desde línea 191 hasta línea 395 y reemplazar con:**

```python
    def create_pipeline(self, config_name: str) -> RAGPipeline:
        """
        Crea un pipeline según la configuración especificada usando DATOS REALES.

        Args:
            config_name: Nombre de la configuración

        Returns:
            RAGPipeline configurado

        Raises:
            RuntimeError: Si no hay datos indexados
        """
        print(f"\n🔧 Configurando: {config_name}")

        # Validar que tenemos datos reales
        if self.qdrant_store is None or self.parquet_storage is None:
            raise RuntimeError(
                "\n❌ No hay datos indexados. Antes de ejecutar el benchmark debes:\n"
                "   1. Indexar PDFs: python indexar_pdfs.py\n"
                "   2. Verificar que existen:\n"
                f"      - {self.config.QDRANT_DIR}/\n"
                f"      - {self.config.PARQUET_DIR}/\n"
            )

        # Crear retriever REAL
        retriever = self._create_retriever(config_name)

        # Configurar componentes según config
        if config_name == "1_baseline":
            # Config 1: Sin mejoras (baseline)
            pipeline = RAGPipeline(
                retriever=retriever,
                llm=self.llm,
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
                llm=self.llm,
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
                llm=self.llm,
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
                llm=self.llm,
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
                llm=self.llm,
                preprocessor=preprocessor,
                reranker=reranker,
                use_preprocessing=True,
                use_reranking=True,
                k_retrieval=20,
                k_final=10
            )

        print(f"   ✅ Pipeline creado (datos REALES)")
        return pipeline
```

---

## 📋 Paso 3: Verificar

### 3.1 Verificar que hay datos indexados:

```bash
ls -la data/qdrant/
ls -la data/parquet/
```

Deberías ver archivos de base de datos de Qdrant y archivos .parquet.

### 3.2 Ejecutar benchmark:

```bash
python benchmark.py --questions 5
```

Si no hay datos, verás:
```
❌ No hay datos indexados. Antes de ejecutar el benchmark debes:
   1. Indexar PDFs: python indexar_pdfs.py
   ...
```

Si hay datos, verás:
```
🔬 BENCHMARK INCREMENTAL DEL RAG PIPELINE
================================================================================
📝 Preguntas cargadas: 5
...
```

---

## 🎯 Resumen

### Antes de poder ejecutar el benchmark con datos reales necesitas:

1. ✅ **Tener PDFs** en `data/pdfs/`
2. ✅ **Ejecutar indexación**: `python indexar_pdfs.py`
3. ✅ **Modificar benchmark.py** según esta guía
4. ✅ **Ejecutar benchmark**: `python benchmark.py`

### El benchmark ahora:
- ❌ **NO acepta** datos simulados/mock
- ✅ **REQUIERE** datos reales indexados
- ✅ **Verifica** que existan Qdrant y Parquet
- ✅ **Falla rápido** con mensaje claro si faltan datos

---

## 💡 Alternativa más rápida

Si quieres probar el benchmark SIN modificar código, puedes:

1. Usar el `benchmark.py` actual (con Mocks) para **testing de la estructura**
2. Cuando tengas PDFs indexados, hacer los cambios de esta guía
3. Ejecutar el benchmark final con datos reales

---

¿Necesitas ayuda con algún paso específico?
