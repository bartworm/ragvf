"""
RAG Pipeline - Interfaz Web con Streamlit (Hito 2)

Incluye:
1. Chat interactivo con documentos (RAG Real).
2. Dashboard de métricas del Benchmark (Baseline vs Mejoras).

Uso:
    streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import json
import time
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Optional

# Agregar módulos al path
sys.path.insert(0, str(Path(__file__).parent))

from config import RAGConfig
from rag.rag_pipeline import RAGPipeline, MockLLM, RAGResponse
from rag.preprocessing.query_preprocessor import QueryPreprocessor
from rag.retrieval.reranker import Reranker
from rag.retrieval.two_stage_retriever import TwoStageRetriever
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.persistence import ParquetPersistence
from sentence_transformers import SentenceTransformer

# ==================== CONFIGURACIÓN DE PÁGINA ====================

st.set_page_config(
    page_title="RAG Pipeline - Conservantes",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ESTILOS CSS ====================

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #eee; margin-bottom: 1rem; }
    .source-box { background-color: #e8f4f8; padding: 1rem; border-left: 4px solid #1f77b4; margin: 0.5rem 0; border-radius: 0.3rem; font-size: 0.9rem; }
    .answer-box { background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #1f77b4; color: #1f77b4; font-weight: bold;}
</style>
""", unsafe_allow_html=True)


# ==================== FUNCIONES DE BENCHMARK (NUEVO) ====================

def cargar_resultados_benchmark():
    """Lee el resumen del benchmark desde el archivo JSON."""
    summary_path = Path("results/benchmark_real/summary.json")
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def render_benchmark_dashboard():
    """Renderiza la pestaña de métricas y gráficos."""
    st.markdown("### 📊 Comparativa de Rendimiento (Baseline vs Hito 2)")
    
    data = cargar_resultados_benchmark()
    
    if not data:
        st.warning("⚠️ No se encontraron resultados del benchmark.")
        st.info("💡 Ejecuta primero: `python benchmark.py` en tu terminal para generar las métricas.")
        return

    # 1. Preparar datos
    configs_order = ["1_baseline", "2_tablas", "3_two_stage", "4_reranking", "5_full"]
    metrics_list = []
    
    for conf in configs_order:
        if conf in data:
            m = data[conf]
            metrics_list.append({
                "Configuración": conf,
                "Latencia (ms)": m["avg_latency"],
                "Score Calidad": m["avg_score"],
                "Uso Tablas (%)": m["table_usage"] * 100,
                "Fuentes": m["avg_sources"]
            })
    
    if not metrics_list:
        st.error("El archivo JSON está vacío o tiene formato incorrecto.")
        return

    df = pd.DataFrame(metrics_list)

    # 2. KPIs de Mejora (Full vs Baseline)
    if "1_baseline" in data and "5_full" in data:
        base = data["1_baseline"]
        full = data["5_full"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Latencia (esperamos que suba, es normal)
        delta_lat = full["avg_latency"] - base["avg_latency"]
        col1.metric("Latencia", f"{full['avg_latency']:.0f} ms", f"{delta_lat:+.0f} ms", delta_color="off")
        
        # Score (esperamos que suba, es bueno)
        delta_score = full["avg_score"] - base["avg_score"]
        col2.metric("Score Calidad", f"{full['avg_score']:.2f}", f"{delta_score:+.2f}", delta_color="normal")
        
        # Uso de tablas
        delta_tablas = (full["table_usage"] - base["table_usage"]) * 100
        col3.metric("Uso de Tablas", f"{full['table_usage']*100:.0f}%", f"{delta_tablas:+.0f}%")
        
        col4.metric("Fuentes Recuperadas", f"{full['avg_sources']:.1f}")

    st.divider()

    # 3. Gráficos
    tab_graf1, tab_graf2 = st.tabs(["📈 Gráficos", "📅 Tabla de Datos"])
    
    with tab_graf1:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Calidad de Respuesta (Score)")
            fig_score = px.bar(
                df, x="Configuración", y="Score Calidad", 
                color="Configuración", text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_score.update_layout(showlegend=False)
            st.plotly_chart(fig_score, use_container_width=True)
            
        with col_g2:
            st.subheader("Tiempo de Respuesta (Latencia)")
            fig_lat = px.bar(
                df, x="Configuración", y="Latencia (ms)", 
                color="Configuración", text_auto='.0f',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_lat.update_layout(showlegend=False)
            st.plotly_chart(fig_lat, use_container_width=True)
            
        # Gráfico de Tablas
        st.subheader("Uso de Información Estructurada (Tablas)")
        fig_tables = px.line(
            df, x="Configuración", y="Uso Tablas (%)", 
            markers=True, text="Uso Tablas (%)"
        )
        fig_tables.update_traces(textposition="bottom right")
        st.plotly_chart(fig_tables, use_container_width=True)

    with tab_graf2:
        st.dataframe(df, use_container_width=True)


# ==================== FUNCIONES DE CHAT (RAG) ====================

@st.cache_resource
def initialize_pipeline(_config: RAGConfig):
    """Inicializa el pipeline RAG REAL."""
    try:
        with st.spinner("🚀 Cargando componentes reales (Qdrant + LLM)..."):
            
            # 1. Componentes de Almacenamiento Real
            storage = ParquetPersistence(storage_dir=_config.PARQUET_DIR)
            qdrant = QdrantVectorStore(path=_config.QDRANT_DIR, embedding_dim=_config.VECTOR_SIZE)
            
            # 2. Embeddings
            embed_model = SentenceTransformer(_config.EMBEDDING_MODEL)
            embedding_func = lambda text: embed_model.encode(text).tolist()

            # 3. LLM
            if _config.LLM_PROVIDER == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=_config.LLM_MODEL,
                    google_api_key=_config.GOOGLE_API_KEY,
                    temperature=_config.LLM_TEMPERATURE
                )
                st.sidebar.success("✅ Gemini Flash conectado")
            elif _config.LLM_PROVIDER == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    api_key=_config.OPENAI_API_KEY,
                    model=_config.LLM_MODEL,
                    temperature=_config.LLM_TEMPERATURE
                )
                st.sidebar.success("✅ OpenAI conectado")
            else:
                llm = MockLLM()
                st.sidebar.warning("⚠️ Usando Mock LLM")

            # 4. Retriever Real (Two-Stage)
            retriever = TwoStageRetriever(
                qdrant_store=qdrant,
                parquet_storage=storage,
                embedding_function=embedding_func,
                full_table_threshold=_config.FULL_TABLE_THRESHOLD,
                k_light=_config.K_RETRIEVAL,
                k_full_tables=_config.MAX_FULL_TABLES
            )

            # 5. Componentes extra
            preprocessor = QueryPreprocessor(use_llm=False)
            reranker = Reranker(_config.RERANKER_MODEL)

            # 6. Pipeline
            pipeline = RAGPipeline(
                config=_config,
                retriever=retriever,
                llm=llm,
                preprocessor=preprocessor,
                reranker=reranker,
                use_preprocessing=_config.USE_QUERY_PREPROCESSING,
                use_reranking=_config.USE_RERANKING,
                k_retrieval=_config.K_RETRIEVAL,
                k_final=_config.K_FINAL
            )

            st.sidebar.success("✅ Pipeline cargado (Datos Reales)")
            return pipeline

    except Exception as e:
        st.error(f"❌ Error crítico inicializando: {e}")
        return None

def load_config() -> RAGConfig:
    """Carga configuración."""
    config = RAGConfig.from_env()
    
    with st.sidebar:
        st.header("⚙️ Ajustes")
        config.K_RETRIEVAL = st.slider("Docs recuperados", 5, 50, config.K_RETRIEVAL)
        config.USE_RERANKING = st.checkbox("Activar Reranker", config.USE_RERANKING)
        config.LLM_TEMPERATURE = st.slider("Creatividad", 0.0, 1.0, config.LLM_TEMPERATURE)
    
    return config

# ==================== UI HELPERS ====================

def render_header():
    st.markdown('<div class="main-header">🔬 RAG Pipeline - Hito 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Conservantes Alimentarios: Texto + Tablas</div>', unsafe_allow_html=True)

def render_chat_interface(pipeline):
    """Interfaz del chat."""
    
    # Historial en Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Haz una pregunta sobre conservantes..."):
        # Guardar y mostrar pregunta
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Analizando documentos y tablas..."):
                try:
                    start = time.time()
                    response: RAGResponse = pipeline.query(prompt)
                    latency = (time.time() - start) * 1000
                    
                    st.markdown(response.answer)
                    
                    # Mostrar fuentes expandibles
                    with st.expander(f"📚 Ver {len(response.sources)} fuentes consultadas ({latency:.0f}ms)"):
                        for i, source in enumerate(response.sources, 1):
                            st.markdown(f"**{i}. {getattr(source, 'source_type', 'text').upper()}** - Score: {getattr(source, 'score', 0):.2f}")
                            st.caption(f"{source.content[:200]}...")
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response.answer})
                
                except Exception as e:
                    st.error(f"Error: {e}")

# ==================== MAIN APP ====================

def main():
    render_header()
    config = load_config()
    pipeline = initialize_pipeline(config)
    
    if not pipeline:
        return

    # PESTAÑAS PRINCIPALES (Lo que pide el Hito 2)
    tab_chat, tab_metrics = st.tabs(["💬 Chat con Documentos", "📊 Benchmark y Métricas"])
    
    with tab_chat:
        render_chat_interface(pipeline)
        
    with tab_metrics:
        render_benchmark_dashboard()

if __name__ == "__main__":
    main()