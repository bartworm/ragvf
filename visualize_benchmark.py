#!/usr/bin/env python3
"""
Visualizador de resultados del Benchmark

Lee los resultados del benchmark y genera visualizaciones y tablas comparativas.

Uso:
    python visualize_benchmark.py
    python visualize_benchmark.py --results results/benchmark_5configs
    python visualize_benchmark.py --format html  # Genera HTML
    python visualize_benchmark.py --format md    # Genera Markdown

Autor: RAG Team
Fecha: Diciembre 2024
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class BenchmarkVisualizer:
    """Visualiza resultados del benchmark."""

    def __init__(self, results_dir: Path):
        """
        Args:
            results_dir: Directorio con resultados JSON
        """
        self.results_dir = Path(results_dir)
        self.summary = self._load_summary()
        self.configs = list(self.summary['configs'].keys())

    def _load_summary(self) -> Dict:
        """Carga el summary.json."""
        summary_file = self.results_dir / "summary.json"

        if not summary_file.exists():
            raise FileNotFoundError(
                f"No se encontró summary.json en {self.results_dir}. "
                "Ejecuta primero: python benchmark.py"
            )

        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_markdown_table(self) -> str:
        """Genera tabla comparativa en formato Markdown."""
        md = []

        md.append("# 📊 Resultados del Benchmark Incremental\n")
        md.append(f"**Fecha**: {self.summary['timestamp']}\n")
        md.append(f"**Preguntas evaluadas**: {self.summary['num_questions']}\n")
        md.append("\n## Comparación de Configuraciones\n")

        # Tabla principal
        md.append("| Configuración | Latencia (ms) | Fuentes | Score | Uso Tablas | Preproc | Rerank |")
        md.append("|--------------|---------------|---------|-------|------------|---------|--------|")

        for config_name in self.configs:
            metrics = self.summary['configs'][config_name]

            md.append(
                f"| {config_name} "
                f"| {metrics['avg_latency_ms']:.0f} "
                f"| {metrics['avg_sources']:.1f} "
                f"| {metrics['avg_score']:.2f} "
                f"| {metrics['table_usage_rate']*100:.0f}% "
                f"| {metrics['preprocessing_rate']*100:.0f}% "
                f"| {metrics['reranking_rate']*100:.0f}% |"
            )

        # Detalles por configuración
        md.append("\n## Detalles por Configuración\n")

        for config_name in self.configs:
            metrics = self.summary['configs'][config_name]

            md.append(f"\n### {config_name}\n")
            md.append(f"- **Latencia promedio**: {metrics['avg_latency_ms']:.0f}ms")
            md.append(f"- **Latencia mínima**: {metrics['min_latency_ms']:.0f}ms")
            md.append(f"- **Latencia máxima**: {metrics['max_latency_ms']:.0f}ms")
            md.append(f"- **Fuentes promedio**: {metrics['avg_sources']:.1f}")
            md.append(f"- **Score promedio**: {metrics['avg_score']:.2f}")
            md.append(f"- **Fuentes de texto**: {metrics['total_text_sources']}")
            md.append(f"- **Fuentes de tablas**: {metrics['total_table_sources']}")
            md.append(f"- **Uso de tablas**: {metrics['table_usage_rate']*100:.0f}%")
            md.append(f"- **Errores**: {metrics['errors']}\n")

        # Análisis comparativo
        md.append("\n## 📈 Análisis Comparativo\n")

        # Mejor latencia
        best_latency = min(self.configs, key=lambda c: self.summary['configs'][c]['avg_latency_ms'])
        md.append(f"**Mejor latencia**: {best_latency} ({self.summary['configs'][best_latency]['avg_latency_ms']:.0f}ms)")

        # Mejor score
        best_score = max(self.configs, key=lambda c: self.summary['configs'][c]['avg_score'])
        md.append(f"\n**Mejor score**: {best_score} ({self.summary['configs'][best_score]['avg_score']:.2f})")

        # Más fuentes
        most_sources = max(self.configs, key=lambda c: self.summary['configs'][c]['avg_sources'])
        md.append(f"\n**Más fuentes**: {most_sources} ({self.summary['configs'][most_sources]['avg_sources']:.1f})")

        # Mayor uso de tablas
        most_tables = max(self.configs, key=lambda c: self.summary['configs'][c]['table_usage_rate'])
        md.append(f"\n**Mayor uso de tablas**: {most_tables} ({self.summary['configs'][most_tables]['table_usage_rate']*100:.0f}%)")

        return "\n".join(md)

    def generate_html(self) -> str:
        """Genera reporte en formato HTML."""
        html = []

        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<meta charset='UTF-8'>")
        html.append("<title>Benchmark Results</title>")
        html.append("<style>")
        html.append("""
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #ddd; }
            tr:hover { background: #f9f9f9; }
            .metric { background: white; padding: 20px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .best { background: #e8f5e9; font-weight: bold; }
            .header { background: #333; color: white; padding: 20px; margin: -40px -40px 40px -40px; }
        </style>")
        html.append("</head>")
        html.append("<body>")

        html.append("<div class='header'>")
        html.append("<h1>📊 Benchmark Incremental del RAG Pipeline</h1>")
        html.append(f"<p>Fecha: {self.summary['timestamp']} | Preguntas: {self.summary['num_questions']}</p>")
        html.append("</div>")

        # Tabla comparativa
        html.append("<h2>Comparación de Configuraciones</h2>")
        html.append("<table>")
        html.append("<tr>")
        html.append("<th>Configuración</th>")
        html.append("<th>Latencia (ms)</th>")
        html.append("<th>Fuentes</th>")
        html.append("<th>Score</th>")
        html.append("<th>Uso Tablas</th>")
        html.append("<th>Preprocessing</th>")
        html.append("<th>Reranking</th>")
        html.append("</tr>")

        for config_name in self.configs:
            metrics = self.summary['configs'][config_name]

            html.append("<tr>")
            html.append(f"<td><strong>{config_name}</strong></td>")
            html.append(f"<td>{metrics['avg_latency_ms']:.0f}</td>")
            html.append(f"<td>{metrics['avg_sources']:.1f}</td>")
            html.append(f"<td>{metrics['avg_score']:.2f}</td>")
            html.append(f"<td>{metrics['table_usage_rate']*100:.0f}%</td>")
            html.append(f"<td>{'✓' if metrics['preprocessing_rate'] > 0 else '✗'}</td>")
            html.append(f"<td>{'✓' if metrics['reranking_rate'] > 0 else '✗'}</td>")
            html.append("</tr>")

        html.append("</table>")

        # Métricas destacadas
        html.append("<h2>Métricas Destacadas</h2>")

        best_latency = min(self.configs, key=lambda c: self.summary['configs'][c]['avg_latency_ms'])
        html.append(f"<div class='metric best'>⚡ <strong>Mejor latencia:</strong> {best_latency} ({self.summary['configs'][best_latency]['avg_latency_ms']:.0f}ms)</div>")

        best_score = max(self.configs, key=lambda c: self.summary['configs'][c]['avg_score'])
        html.append(f"<div class='metric best'>🎯 <strong>Mejor score:</strong> {best_score} ({self.summary['configs'][best_score]['avg_score']:.2f})</div>")

        most_tables = max(self.configs, key=lambda c: self.summary['configs'][c]['table_usage_rate'])
        html.append(f"<div class='metric best'>📊 <strong>Mayor uso de tablas:</strong> {most_tables} ({self.summary['configs'][most_tables]['table_usage_rate']*100:.0f}%)</div>")

        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)

    def generate_console_output(self):
        """Genera salida formateada para consola."""
        print("\n" + "="*80)
        print("📊 RESULTADOS DEL BENCHMARK INCREMENTAL")
        print("="*80)
        print(f"Fecha: {self.summary['timestamp']}")
        print(f"Preguntas evaluadas: {self.summary['num_questions']}")
        print("="*80 + "\n")

        # Tabla comparativa
        print(f"{'Configuración':<20} {'Latencia':<12} {'Fuentes':<10} {'Score':<10} {'Tablas%':<10}")
        print("-"*80)

        for config_name in self.configs:
            metrics = self.summary['configs'][config_name]
            print(
                f"{config_name:<20} "
                f"{metrics['avg_latency_ms']:>8.0f}ms   "
                f"{metrics['avg_sources']:>7.1f}    "
                f"{metrics['avg_score']:>7.2f}    "
                f"{metrics['table_usage_rate']*100:>6.1f}%"
            )

        # Análisis
        print("\n" + "="*80)
        print("📈 ANÁLISIS")
        print("="*80)

        best_latency = min(self.configs, key=lambda c: self.summary['configs'][c]['avg_latency_ms'])
        print(f"⚡ Mejor latencia: {best_latency} ({self.summary['configs'][best_latency]['avg_latency_ms']:.0f}ms)")

        best_score = max(self.configs, key=lambda c: self.summary['configs'][c]['avg_score'])
        print(f"🎯 Mejor score:    {best_score} ({self.summary['configs'][best_score]['avg_score']:.2f})")

        most_tables = max(self.configs, key=lambda c: self.summary['configs'][c]['table_usage_rate'])
        print(f"📊 Más tablas:     {most_tables} ({self.summary['configs'][most_tables]['table_usage_rate']*100:.0f}%)")

        print("="*80 + "\n")

    def save_report(self, format: str = "md"):
        """Guarda reporte en archivo."""
        if format == "md":
            content = self.generate_markdown_table()
            output_file = self.results_dir / "REPORT.md"
        elif format == "html":
            content = self.generate_html()
            output_file = self.results_dir / "REPORT.html"
        else:
            raise ValueError(f"Formato no soportado: {format}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"💾 Reporte guardado en: {output_file}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Visualiza resultados del benchmark"
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/benchmark_5configs"),
        help="Directorio con resultados (default: results/benchmark_5configs)"
    )

    parser.add_argument(
        "--format",
        choices=["console", "md", "html"],
        default="console",
        help="Formato de salida (default: console)"
    )

    args = parser.parse_args()

    # Crear visualizador
    try:
        visualizer = BenchmarkVisualizer(args.results)

        if args.format == "console":
            visualizer.generate_console_output()
        else:
            visualizer.save_report(args.format)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
