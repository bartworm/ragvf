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

## 🎯 Uso

### 1. Indexar PDFs

Coloca tus PDFs en `data/pdfs/` y ejecuta:

```bash
python main.py --index data/pdfs/
```

Esto generará:
- Embeddings en Qdrant (chunks, descriptors, summaries)
- Tablas completas en Parquet
- Metadata de procesamiento

**Salida esperada:**
```
✓ 20 PDFs procesados
✓ 2150 chunks de texto indexados
✓ 22 tablas extraídas
✓ Tiempo total: 45s
```

### 2. Ejecutar Benchmark

Evalúa 5 configuraciones del sistema:

```bash
# Benchmark completo (50 preguntas)
python benchmark.py

# Benchmark rápido (5 preguntas)
python benchmark.py --questions 5

# Solo una configuración
python benchmark.py --config 5_full --questions 10
```

**Configuraciones evaluadas:**
1. `1_baseline`: Solo texto (sin tablas)
2. `2_tablas`: + Recuperación de tablas
3. `3_two_stage`: + Two-stage retrieval
4. `4_reranking`: + Cross-encoder reranking
5. `5_full`: Sistema completo (preprocesamiento + reranking)

### 3. Consultar el Sistema (Interactivo)

```python
from config import RAGConfig
from rag.rag_pipeline import RAGPipeline

# Cargar configuración
config = RAGConfig.from_env()

# Inicializar pipeline
pipeline = RAGPipeline.from_config(config)

# Hacer consulta
response = pipeline.query("¿Cuál es la precisión del algoritmo Random Forest?")

print(response.answer)
print(f"Fuentes: {len(response.sources)}")
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

## 🔧 Configuración Avanzada

### Modificar Parámetros de Retrieval

En `.env`:

```bash
K_RETRIEVAL=20        # Resultados iniciales
K_FINAL=10           # Después de reranking
CHUNK_SIZE=1000      # Tamaño de chunks
CHUNK_OVERLAP=150    # Solapamiento
```

### Cambiar Modelo de Embeddings

En `config.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
# Alternativas:
# "all-mpnet-base-v2"  # 768 dims, mejor calidad
# "paraphrase-multilingual-MiniLM-L12-v2"  # Multilenguaje
```

### Usar Descriptores LLM en Tablas

Por defecto se usan descriptores heurísticos (sin costo API). Para usar LLM:

```bash
# En .env
USE_LLM_FOR_TABLE_DESCRIPTORS=true
```

**⚠️ Advertencia**: Esto genera un API call por cada tabla durante indexación.

## 🐛 Troubleshooting

### Error: "No module named 'camelot'"

```bash
pip install camelot-py[cv]
sudo apt-get install ghostscript  # Linux
```

### Error: Qdrant "Collection already exists"

```bash
rm -rf data/qdrant/*  # Borrar índice existente
python main.py --index data/pdfs/  # Re-indexar
```

### Ollama no responde

```bash
# Verificar que está corriendo
ollama serve

# En otra terminal
ollama list  # Ver modelos disponibles
```

### GPU no detectada

```bash
import torch
print(torch.cuda.is_available())  # Debe ser True

# Si es False, reinstalar:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Documentación Adicional

- [INFORME.md](INFORME.md): Análisis técnico completo del proyecto
- [data/questions.json](data/questions.json): Preguntas de benchmark
- Código comentado en `/rag/` con docstrings

## 🤝 Contribuciones

Este es un proyecto académico (Hito 2), pero sugerencias son bienvenidas:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agrega X funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Pull Request

## 📄 Licencia

Proyecto académico - uso educativo.

## ✉️ Contacto

**Autor:** Bernardo Pinochet
**Institución:** Universidad de Chile
**Fecha:** Diciembre 2025

## 🙏 Agradecimientos

- LangChain por el framework RAG
- Qdrant por el vector store
- Sentence Transformers por los embeddings
- Camelot por la extracción de tablas
- Ollama por LLMs locales

---

**¿Necesitas ayuda?** Revisa [INFORME.md](INFORME.md) o abre un issue.
