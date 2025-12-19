# 🔬 RAG Pipeline - Conservantes Alimentarios

Sistema de **Retrieval-Augmented Generation (RAG)** para consultar documentos PDF científicos sobre conservantes alimentarios y antimicrobianos.

## 📋 Descripción

Este proyecto implementa un pipeline completo de RAG que:
- ✅ Extrae contenido estructurado de PDFs (texto, tablas, bibliografía)
- ✅ Almacena información en base de datos vectorial (Qdrant)
- ✅ Procesa queries de usuarios con preprocesamiento inteligente
- ✅ Recupera información relevante usando búsqueda semántica
- ✅ Genera respuestas usando LLMs (OpenAI GPT)
- ✅ Ofrece dos interfaces: CLI y Web (Streamlit)

## 🚀 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd /home/bartworm/Desktop/Programacion/hito2/hito2
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno (opcional)

Crea un archivo `.env` en la raíz del proyecto:

```bash
# API Keys (opcional - funciona en modo Mock sin estas)
OPENAI_API_KEY=tu_api_key_aqui
ANTHROPIC_API_KEY=tu_api_key_aqui

# Configuración (opcional - usa defaults si no se especifica)
EXTRACTION_LEVEL=3
K_RETRIEVAL=20
K_FINAL=10
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.3
DEBUG=false
```

## 🎯 Uso

### Opción 1: Interfaz Web (Streamlit) - RECOMENDADO

```bash
streamlit run streamlit_app.py
```

Esto abrirá tu navegador en `http://localhost:8501` con una interfaz completa que incluye:
- 💬 Input de preguntas con ejemplos
- 📖 Visualización de respuestas formateadas
- 📚 Exploración de fuentes consultadas
- ⚙️ Configuración interactiva en sidebar
- 📊 Métricas y estadísticas en tiempo real
- 📜 Historial de consultas

### Opción 2: CLI (Línea de comandos)

#### Modo interactivo:
```bash
python main.py
```

#### Query única:
```bash
python main.py --query "¿Qué es benzoato de sodio?"
```

#### Con configuración personalizada:
```bash
python main.py --k-retrieval 30 --k-final 10 --extraction-level 3
```

#### Ver ayuda:
```bash
python main.py --help
```

## 📂 Estructura del Proyecto

```
hito2/
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias
├── .gitignore                  # Archivos ignorados por Git
├── main.py                     # CLI principal
├── streamlit_app.py            # Interfaz web Streamlit
├── config.py                   # Configuración centralizada
│
├── data/                       # Datos (crear si no existe)
│   ├── pdfs/                   # PDFs a procesar
│   ├── qdrant/                 # Base de datos vectorial
│   └── parquet/                # Almacenamiento Parquet
│
└── rag/                        # Módulo principal
    ├── models.py               # Modelos de datos
    ├── rag_pipeline.py         # Orquestador principal
    │
    ├── extraction/             # Extracción de PDFs
    │   ├── extractors.py
    │   ├── table_extractor.py
    │   ├── bibliography_extractor.py
    │   └── ...
    │
    ├── preprocessing/          # Preprocesamiento
    │   └── query_preprocessor.py
    │
    ├── retrieval/              # Búsqueda y recuperación
    │   ├── two_stage_retriever.py
    │   ├── baseline_retriever.py
    │   └── reranker.py
    │
    └── storage/                # Persistencia
        ├── qdrant_store.py
        └── persistence.py
```

## 💡 Ejemplos de Uso

### Ejemplos de preguntas (Streamlit o CLI):

```
¿Qué es el benzoato de sodio y cómo funciona?
¿Cuáles son las alternativas naturales a los conservantes químicos?
¿A qué pH es efectiva la nisina?
¿Qué concentración de sorbato se recomienda para bebidas?
¿Qué microorganismos son resistentes al benzoato?
Comparación entre nisina y natamicina
```

### Ejemplo de uso programático:

```python
from config import RAGConfig
from rag.rag_pipeline import RAGPipeline
from rag.preprocessing.query_preprocessor import QueryPreprocessor
from rag.retrieval.reranker import Reranker

# Cargar configuración
config = RAGConfig.from_env()

# Crear componentes
preprocessor = QueryPreprocessor(use_llm=False)
reranker = Reranker()

# Inicializar pipeline
pipeline = RAGPipeline(
    retriever=your_retriever,
    llm=your_llm,
    preprocessor=preprocessor,
    reranker=reranker,
    k_retrieval=20,
    k_final=10
)

# Ejecutar query
response = pipeline.query("¿Qué es benzoato de sodio?")
print(response.answer)
print(f"Fuentes: {len(response.sources)}")
```

## ⚙️ Configuración

### Parámetros principales (en `config.py`):

#### Extracción:
- `EXTRACTION_LEVEL` (0-4): Nivel de detalle en extracción
  - 0: Texto plano
  - 1: + Bibliografía
  - 2: + Layout (columnas)
  - 3: + Tablas
  - 4: + Contexto de tablas

- `CHUNK_SIZE`: Tamaño de chunks de texto (default: 1000)
- `CHUNK_OVERLAP`: Solapamiento entre chunks (default: 150)

#### Retrieval:
- `K_RETRIEVAL`: Resultados iniciales (default: 20)
- `K_FINAL`: Resultados después de reranking (default: 10)
- `FULL_TABLE_THRESHOLD`: Umbral para cargar tablas completas (default: 0.75)

#### LLM:
- `LLM_MODEL`: Modelo a usar (default: "gpt-3.5-turbo")
- `LLM_TEMPERATURE`: Creatividad del modelo 0-1 (default: 0.3)
- `MAX_CONTEXT_LENGTH`: Máximo de contexto (default: 8000)

#### Pipeline:
- `USE_QUERY_PREPROCESSING`: Preprocesar queries (default: True)
- `USE_RERANKING`: Reordenar resultados (default: True)
- `INCLUDE_SOURCES`: Incluir fuentes en respuesta (default: True)

## 🔧 Desarrollo

### Ejecutar en modo debug:

```bash
# CLI
python main.py --debug

# Streamlit (activar en sidebar)
streamlit run streamlit_app.py
```

### Modificar componentes:

El sistema es modular. Puedes reemplazar componentes:

```python
# Usar tu propio retriever
from rag.retrieval.two_stage_retriever import TwoStageRetriever
retriever = TwoStageRetriever(vector_store, persistence)

# Usar tu propio LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key="...", model="gpt-4")

# Crear pipeline con tus componentes
pipeline = RAGPipeline(retriever=retriever, llm=llm, ...)
```

## 📊 Componentes del Sistema

### 1. Extracción (`rag/extraction/`)
- Procesa PDFs científicos
- Separa texto narrativo de tablas
- Detecta bibliografía y secciones
- Maneja layouts multi-columna

### 2. Almacenamiento (`rag/storage/`)
- **Qdrant**: Vector store para embeddings
- **Parquet**: Fuente de verdad para tablas completas

### 3. Retrieval (`rag/retrieval/`)
- **Two-stage**: Búsqueda en dos etapas (light + full)
- **Reranker**: Cross-encoder para reordenar resultados
- **Baseline**: Compatible con Hito 1 (Chroma)

### 4. Preprocessing (`rag/preprocessing/`)
- Corrección ortográfica
- Expansión de sinónimos
- Reescritura de queries

### 5. Pipeline (`rag/rag_pipeline.py`)
- Orquesta todo el flujo end-to-end
- Gestiona contexto y límites
- Genera respuestas con LLM

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'rag'"
```bash
# Asegúrate de estar en la raíz del proyecto
cd /home/bartworm/Desktop/Programacion/hito2/hito2
python main.py
```

### Error: "API key not found"
```bash
# El sistema funciona sin API keys en modo Mock
# Para usar OpenAI, crea .env con tu API key
echo "OPENAI_API_KEY=tu_key" > .env
```

### Streamlit no abre el navegador
```bash
# Abre manualmente en:
http://localhost:8501
```

### Error de Pydantic
```bash
# Ya está corregido en models.py
# Si persiste, verifica la versión:
pip install pydantic==2.10.4
```

## 📚 Dependencias Principales

- `streamlit` 1.41.1 - Interfaz web
- `langchain` 0.3.13 - Framework LLM
- `qdrant-client` 1.12.1 - Vector store
- `sentence-transformers` 3.3.1 - Embeddings
- `pydantic` 2.10.4 - Validación de datos
- `pandas` 2.2.3 - Manejo de datos
- `camelot-py` 0.11.0 - Extracción de tablas

Ver lista completa en [`requirements.txt`](requirements.txt)

## 🎓 Autores

- **RAG Team**
- Proyecto: Hito 2 - Diciembre 2024
- Universidad: [Tu universidad]

## 📄 Licencia

[Especificar licencia si aplica]

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Haz un fork
2. Crea una rama con tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Para preguntas o problemas:
- Abre un issue en el repositorio
- Contacta al equipo de desarrollo

---

**🚀 ¡Listo para usar!** Ejecuta `streamlit run streamlit_app.py` para comenzar.
