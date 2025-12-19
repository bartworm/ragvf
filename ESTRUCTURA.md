# Estructura del Proyecto - Hito 2

## 📁 Organización de Archivos

```
hito2/
├── main.py                     # 🚀 CLI principal para ejecutar el RAG
├── config.py                   # ⚙️ Configuración centralizada
├── requirements.txt            # 📦 Dependencias del proyecto
├── .gitignore                  # 🚫 Archivos ignorados por Git
│
├── data/                       # 📊 Datos (crear si no existe)
│   ├── pdfs/                   # PDFs a procesar
│   ├── qdrant/                 # Base de datos vectorial
│   └── parquet/                # Almacenamiento en Parquet
│
├── rag/                        # 🧠 Módulo principal del RAG
│   ├── models.py               # 📋 Modelos Pydantic (TextChunk, FullTable, etc.)
│   ├── rag_pipeline.py         # 🔄 Orquestador principal del flujo RAG
│   │
│   ├── extraction/             # 📄 Extracción de contenido de PDFs
│   │   ├── __init__.py
│   │   ├── base_extractor.py
│   │   ├── extractors.py
│   │   ├── table_extractor.py
│   │   ├── bibliography_extractor.py
│   │   ├── layout_extractor.py
│   │   ├── layout_aware_extractor.py
│   │   ├── main_extractor.py
│   │   ├── section_detector.py
│   │   ├── table_representations.py
│   │   └── improved_bibliography_detector.py
│   │
│   ├── preprocessing/          # 🔍 Preprocesamiento de queries
│   │   ├── __init__.py
│   │   └── query_preprocessor.py
│   │
│   ├── retrieval/              # 🔎 Sistemas de búsqueda y recuperación
│   │   ├── __init__.py
│   │   ├── baseline_retriever.py
│   │   ├── two_stage_retriever.py
│   │   └── reranker.py
│   │
│   └── storage/                # 💾 Persistencia y almacenamiento
│       ├── __init__.py
│       ├── qdrant_store.py     # Vector store (Qdrant)
│       └── persistence.py      # Almacenamiento Parquet
│
└── venv/                       # 🐍 Entorno virtual de Python
```

## 🔧 Componentes Principales

### 1️⃣ **Extracción** (`rag/extraction/`)
- Procesa PDFs científicos
- Extrae texto, tablas y bibliografía
- Detecta layout y secciones

### 2️⃣ **Almacenamiento** (`rag/storage/`)
- **Qdrant**: Base de datos vectorial para embeddings
- **Parquet**: Almacenamiento de tablas completas

### 3️⃣ **Retrieval** (`rag/retrieval/`)
- **Two-stage retriever**: Búsqueda en dos etapas
- **Reranker**: Reordenamiento de resultados
- **Baseline**: Compatibilidad con Hito 1

### 4️⃣ **Preprocessing** (`rag/preprocessing/`)
- Mejora de queries del usuario
- Corrección ortográfica
- Expansión de sinónimos

### 5️⃣ **Pipeline** (`rag/rag_pipeline.py`)
- Orquesta todo el flujo
- Genera respuestas con LLM

## 🚀 Uso

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar en modo interactivo
```bash
python main.py
```

### Ejecutar con una query
```bash
python main.py --query "¿Qué es benzoato de sodio?"
```

### Ver ayuda
```bash
python main.py --help
```

## 📝 Archivos Importantes

- **`models.py`**: Define estructuras de datos (TextChunk, FullTable, TableDescriptor)
- **`config.py`**: Configuración centralizada (paths, parámetros, API keys)
- **`main.py`**: Interfaz CLI

## 🔄 Cambios Recientes

✅ Archivos `_CORREGIDO` renombrados a nombres normales
✅ Estructura reorganizada en subcarpetas funcionales
✅ Imports actualizados a rutas absolutas (`rag.module.file`)
✅ Error de Pydantic corregido (`@root_validator(skip_on_failure=True)`)
✅ `requirements.txt` y `.gitignore` creados

## ⚠️ Nota sobre Streamlit

El archivo original `rag/streamlit_app.py` era un duplicado de `rag_pipeline.py` y fue eliminado.
Si necesitas una interfaz web Streamlit, deberás crearla desde cero.
