# 🔬 Benchmark Incremental - Guía Completa

## 📋 ¿Qué es?

El benchmark incremental compara **5 configuraciones progresivas** del sistema RAG, evaluando el impacto de cada mejora:

```
1. Baseline (Hito 1)        → Sistema básico con Chroma
   ↓ +Mejora
2. + Tablas                 → Extracción avanzada de tablas
   ↓ +Mejora
3. + Two-Stage Retrieval    → Recuperación en dos etapas
   ↓ +Mejora
4. + Reranking              → Reordenamiento con cross-encoder
   ↓ +Mejora
5. + Preprocessing          → Preprocesamiento de queries
```

## 🎯 ¿Qué mide?

Para las **mismas 50 preguntas**, compara:

### Métricas de rendimiento:
- ⏱️ **Latencia** (tiempo de respuesta en ms)
- 📊 **Número de fuentes** recuperadas
- 🎯 **Score promedio** de relevancia

### Métricas de calidad:
- 📄 **Uso de tablas** (% de queries que usan tablas)
- 🔍 **Tipos de fuentes** (texto vs tablas vs descriptores)
- ✅ **Activación de features** (preprocessing, reranking)

## 🚀 Uso Rápido

### 1️⃣ Ejecutar benchmark completo (50 preguntas, 5 configs):
```bash
python benchmark.py
```

### 2️⃣ Ver resultados en consola:
```bash
python visualize_benchmark.py
```

### 3️⃣ Generar reporte Markdown:
```bash
python visualize_benchmark.py --format md
```

### 4️⃣ Generar reporte HTML:
```bash
python visualize_benchmark.py --format html
```

---

## 📖 Uso Avanzado

### Ejecutar solo algunas configuraciones:

```bash
# Solo baseline
python benchmark.py --config 1_baseline

# Solo config final
python benchmark.py --config 5_full
```

### Probar con menos preguntas:

```bash
# Solo primeras 10 preguntas (testing rápido)
python benchmark.py --questions 10

# Solo primeras 25 preguntas
python benchmark.py --questions 25
```

### Cambiar directorio de salida:

```bash
python benchmark.py --output results/mi_benchmark
```

---

## 📁 Estructura de Resultados

Después de ejecutar el benchmark, se genera:

```
results/benchmark_5configs/
├── 1_baseline_results.json        # Resultados config 1
├── 2_tablas_results.json          # Resultados config 2
├── 3_two_stage_results.json       # Resultados config 3
├── 4_reranking_results.json       # Resultados config 4
├── 5_full_results.json            # Resultados config 5
├── summary.json                   # Resumen comparativo
├── REPORT.md                      # Reporte Markdown (si se generó)
└── REPORT.html                    # Reporte HTML (si se generó)
```

### Formato de `*_results.json`:

```json
{
  "config_name": "1_baseline",
  "timestamp": "2024-12-18T23:30:00",
  "metrics": {
    "config_name": "1_baseline",
    "total_questions": 50,
    "avg_latency_ms": 450.5,
    "avg_sources": 3.2,
    "avg_score": 0.70,
    "total_text_sources": 150,
    "total_table_sources": 10,
    "table_usage_rate": 0.20,
    "preprocessing_rate": 0.0,
    "reranking_rate": 0.0
  },
  "results": [
    {
      "config_name": "1_baseline",
      "question_id": 1,
      "question": "¿Qué es el benzoato de sodio?",
      "answer": "[Respuesta generada]",
      "latency_ms": 445.2,
      "num_sources": 3,
      "source_types": {"text": 3},
      "avg_score": 0.70,
      "preprocessing_enabled": false,
      "reranking_enabled": false
    },
    ...
  ]
}
```

---

## 📊 Ejemplo de Salida

### Consola:

```
================================================================================
📊 RESULTADOS DEL BENCHMARK INCREMENTAL
================================================================================
Fecha: 2024-12-18T23:30:00
Preguntas evaluadas: 50
================================================================================

Configuración         Latencia     Fuentes    Score      Tablas%
--------------------------------------------------------------------------------
1_baseline              450ms       3.2        0.70        20%
2_tablas                520ms       7.0        0.73        40%
3_two_stage             480ms       8.0        0.80        50%
4_reranking             650ms       7.0        0.85        45%
5_full                  680ms       8.0        0.90        50%

================================================================================
📈 ANÁLISIS
================================================================================
⚡ Mejor latencia: 1_baseline (450ms)
🎯 Mejor score:    5_full (0.90)
📊 Más tablas:     3_two_stage (50%)
================================================================================
```

---

## 🔧 Personalización

### Modificar las configuraciones:

Edita `benchmark.py`, método `create_pipeline()`:

```python
def create_pipeline(self, config_name: str) -> RAGPipeline:
    if config_name == "1_baseline":
        # Personaliza aquí
        pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            k_retrieval=10,  # Cambia parámetros
            k_final=5
        )
```

### Añadir nuevas preguntas:

Edita `data/questions.json`:

```json
{
  "id": 51,
  "question": "Tu nueva pregunta aquí",
  "category": "tecnica",
  "expected_topics": ["tema1", "tema2"],
  "difficulty": "media"
}
```

---

## 🎓 Interpretación de Resultados

### Latencia:
- **Más baja** = Más rápido (mejor para producción)
- **Trade-off**: Configuraciones avanzadas son más lentas pero mejores

### Score:
- **Más alto** = Documentos más relevantes
- **Esperable**: Configs avanzadas (con reranking) tienen mejor score

### Uso de Tablas:
- **Más alto** = Mejor aprovechamiento de información estructurada
- **Importante**: Tablas contienen datos críticos (concentraciones, pH, etc.)

### Análisis típico:

```
Config 1 (Baseline):     Rápido pero score bajo
Config 2 (+Tablas):      Más lento, mejor info
Config 3 (+Two-stage):   Balanceado, buen score
Config 4 (+Reranking):   Mejor score, más lento
Config 5 (+Full):        Máxima calidad, máxima latencia
```

**Recomendación**: Config 4 o 5 para producción si la latencia es aceptable.

---

## 🐛 Troubleshooting

### Error: "No se encontró summary.json"
```bash
# Solución: Ejecuta primero el benchmark
python benchmark.py
```

### Benchmark muy lento:
```bash
# Solución: Usa menos preguntas
python benchmark.py --questions 10
```

### Mock retriever devuelve datos simulados:
```
# Esto es esperado si no has indexado PDFs reales
# Para usar datos reales:
1. Indexa PDFs en Qdrant
2. Reemplaza MockRetriever con TwoStageRetriever real en benchmark.py
```

---

## 📈 Próximos Pasos

1. **Ejecutar con datos reales**:
   - Indexar PDFs en Qdrant
   - Reemplazar MockRetriever

2. **Añadir métricas adicionales**:
   - BLEU score (comparación con respuestas gold)
   - ROUGE score
   - Precisión@K

3. **Automatizar**:
   - Ejecutar benchmark automáticamente en CI/CD
   - Comparar con benchmarks anteriores

4. **Visualizaciones**:
   - Gráficos de latencia vs score
   - Distribución de tipos de fuentes
   - Análisis por categoría de pregunta

---

## 💡 Tips

- 🚀 **Testing rápido**: `--questions 5` para iteraciones rápidas
- 📊 **Comparar incrementos**: Ejecuta configs individualmente con `--config`
- 📁 **Múltiples runs**: Usa `--output` para guardar en carpetas diferentes
- 🎯 **Enfoque**: Filtra preguntas por categoría para análisis específico

---

**¿Preguntas?** Revisa el código en `benchmark.py` o `visualize_benchmark.py`
