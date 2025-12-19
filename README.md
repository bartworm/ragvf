# Sistema RAG con Tratamiento Avanzado de Tablas

Sistema de Recuperación Aumentada de Generación (RAG) especializado en procesamiento de papers científicos con tablas, implementando un pipeline de múltiples etapas que emula la lectura humana de documentos.

## 📋 Descripción

Este proyecto implementa un sistema RAG avanzado que:
- Extrae y procesa tablas de PDFs científicos manteniendo su estructura semántica
- Utiliza two-stage retrieval (descriptores livianos → tablas completas)
- Captura contexto narrativo alrededor de las tablas
- Soporta múltiples proveedores de LLM (OpenAI, Google Gemini, Groq, Ollama)
- Incluye benchmark comparativo de 5 configuraciones

## 🏗️ Arquitectura

```
PDF → Extracción → Chunking → Embeddings → Qdrant → Two-Stage → Reranking → LLM
      (Tablas)     (Contexto)  (Vectores)   (Index)   Retrieval   (Cross-Enc)
```

### Componentes Principales

- **Extracción Multi-Modal**: Camelot + PDFPlumber + PyMuPDF
- **Vector Store**: Qdrant (3 colecciones: chunks, descriptors, summaries)
- **Storage**: Parquet (compresión snappy)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: Flexible (OpenAI, Gemini, Groq, Ollama local)

## 🚀 Instalación

### Requisitos Previos

- Python 3.10+
- GPU con CUDA (opcional, pero recomendado para embeddings y reranking)
- 4GB+ RAM
- Ghostscript (para Camelot)

### Paso 1: Clonar el Repositorio

```bash
git clone <repo-url>
cd hito2
```

### Paso 2: Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Configurar Variables de Entorno

Copia el archivo de ejemplo y configura tus API keys:

```bash
cp .env.example .env
nano .env  # o usa tu editor favorito
```

**Variables importantes:**

```bash
# LLM Provider (elige uno)
LLM_PROVIDER=ollama  # opciones: openai, google, groq, ollama, mock
LLM_MODEL=llama3.2-vision

# API Keys (si usas servicios cloud)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# Directorios
PDF_DIR=data/pdfs
QDRANT_DIR=data/qdrant
PARQUET_DIR=data/parquet
```

### Paso 5 (Opcional): Instalar Ollama para LLM Local

Si quieres usar un LLM local sin API keys:

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelo
ollama pull llama3.2-vision  # 7.9GB (recomendado para GPU 12GB)
# o
ollama pull llama3.1:8b      # 4.7GB (más liviano)
```

## 📦 Estructura del Proyecto

```
hito2/
├── data/
│   ├── pdfs/          # PDFs a indexar (no incluidos en Git)
│   ├── qdrant/        # Base de datos vectorial (no en Git)
│   ├── parquet/       # Storage de tablas (no en Git)
│   └── questions.json # Preguntas para benchmark
├── rag/
│   ├── extraction/    # Extractores de PDF y tablas
│   ├── preprocessing/ # Query preprocessing
│   ├── retrieval/     # Two-stage retriever + reranker
│   ├── storage/       # Qdrant + Parquet
│   ├── models.py      # Modelos Pydantic
│   └── rag_pipeline.py
├── results/           # Resultados de benchmarks (no en Git)
├── main.py           # Script principal de indexación
├── benchmark.py      # Script de evaluación
├── config.py         # Configuración centralizada
├── INFORME.md        # Informe técnico completo
└── README.md         # Este archivo
```


## 📊 Resultados del Benchmark

Ejemplo con 5 preguntas sobre papers de ML:

| Configuración | Score | Latencia | Tablas |
|--------------|-------|----------|--------|
| 1. Baseline | 0.525 | 54 ms | 0% |
| 2. Con tablas | 0.533 | 10 ms | 80% |
| 3. Two-stage | 0.533 | 13 ms | 80% |
| 4. Reranking | 0.511 | 242 ms | 100% |
| **5. Full** | **0.555** | 195 ms | 100% |

**Conclusión**: El tratamiento de tablas mejora +5.7% el score, con latencia competitiva.

Ver [INFORME.md](INFORME.md) para análisis detallado.



