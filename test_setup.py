#!/usr/bin/env python3
"""
Script de verificación de instalación

Verifica que todos los componentes del proyecto estén correctamente instalados.

Uso:
    python test_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Verifica que todos los módulos se importen correctamente."""
    print("\n" + "="*60)
    print("🧪 VERIFICACIÓN DE INSTALACIÓN")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Configuración
    print("\n1️⃣ Verificando configuración...")
    try:
        from config import RAGConfig
        config = RAGConfig.from_env()
        print(f"   ✅ Config cargado: {config}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error en config: {e}")
        tests_failed += 1

    # Test 2: Modelos
    print("\n2️⃣ Verificando modelos...")
    try:
        from rag.models import TextChunk, FullTable, TableDescriptor
        print("   ✅ Modelos importados correctamente")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error en models: {e}")
        tests_failed += 1

    # Test 3: Pipeline
    print("\n3️⃣ Verificando RAG pipeline...")
    try:
        from rag.rag_pipeline import RAGPipeline, MockLLM
        print("   ✅ RAG Pipeline disponible")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error en pipeline: {e}")
        tests_failed += 1

    # Test 4: Preprocessing
    print("\n4️⃣ Verificando preprocessing...")
    try:
        from rag.preprocessing.query_preprocessor import QueryPreprocessor
        preprocessor = QueryPreprocessor(use_llm=False)
        print("   ✅ Query Preprocessor disponible")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error en preprocessing: {e}")
        tests_failed += 1

    # Test 5: Retrieval
    print("\n5️⃣ Verificando retrieval...")
    try:
        from rag.retrieval.reranker import Reranker
        from rag.retrieval.two_stage_retriever import TwoStageRetriever, RetrievalResult
        print("   ✅ Retrieval components disponibles")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error en retrieval: {e}")
        tests_failed += 1

    # Test 6: Storage
    print("\n6️⃣ Verificando storage...")
    try:
        from rag.storage.qdrant_store import QdrantVectorStore
        from rag.storage.persistence import ParquetPersistence
        print("   ✅ Storage components disponibles")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error en storage: {e}")
        tests_failed += 1

    # Test 7: Extraction
    print("\n7️⃣ Verificando extraction...")
    try:
        from rag.extraction.extractors import PDFContentExtractor
        print("   ✅ Extraction components disponibles")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error en extraction: {e}")
        tests_failed += 1

    # Test 8: Dependencias externas críticas
    print("\n8️⃣ Verificando dependencias externas...")
    critical_deps = {
        'streamlit': 'Streamlit',
        'langchain': 'LangChain',
        'qdrant_client': 'Qdrant',
        'pandas': 'Pandas',
        'pydantic': 'Pydantic',
    }

    missing_deps = []
    for module, name in critical_deps.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
            tests_passed += 1
        except ImportError:
            print(f"   ❌ {name} no instalado")
            missing_deps.append(name)
            tests_failed += 1

    # Test 9: Directorios
    print("\n9️⃣ Verificando estructura de directorios...")
    required_dirs = [
        Path("rag/extraction"),
        Path("rag/preprocessing"),
        Path("rag/retrieval"),
        Path("rag/storage"),
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"   ✅ {dir_path}/")
            tests_passed += 1
        else:
            print(f"   ❌ {dir_path}/ no existe")
            tests_failed += 1

    # Test 10: Archivos importantes
    print("\n🔟 Verificando archivos importantes...")
    important_files = [
        "main.py",
        "streamlit_app.py",
        "config.py",
        "requirements.txt",
        ".gitignore",
        "README.md"
    ]

    for file_name in important_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"   ✅ {file_name}")
            tests_passed += 1
        else:
            print(f"   ❌ {file_name} no existe")
            tests_failed += 1

    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN")
    print("="*60)
    print(f"Tests pasados: {tests_passed}")
    print(f"Tests fallidos: {tests_failed}")

    if tests_failed == 0:
        print("\n✅ ¡TODO CORRECTO! El proyecto está listo para usar.")
        print("\n🚀 Ejecuta uno de estos comandos para empezar:")
        print("   • streamlit run streamlit_app.py  (Interfaz web)")
        print("   • python main.py                   (CLI)")
        return True
    else:
        print("\n⚠️ Hay algunos problemas que resolver.")
        if missing_deps:
            print(f"\n📦 Dependencias faltantes: {', '.join(missing_deps)}")
            print("   Instala con: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    try:
        success = test_imports()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
