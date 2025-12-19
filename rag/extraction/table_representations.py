import uuid
import random
from typing import List, Any, Optional
from datetime import datetime
from rag.models import FullTable, TableDescriptor, TableSummary, TableRow

from langchain_core.messages import SystemMessage, HumanMessage

class TableRepresentationGenerator:
    """Genera representaciones semánticas de tablas usando LLM o heurísticas."""

    def __init__(self, llm, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm
        self.topk = 5  # Filas para el resumen

    def generate_descriptor(self, table: FullTable) -> TableDescriptor:
        """Genera un descriptor semántico de la tabla usando LLM o heurísticas."""
        if self.use_llm:
            prompt = self._create_descriptor_prompt(table)
            try:
                description = self._call_llm(prompt)
            except Exception as e:
                error_str = str(e)
                if 'RESOURCE_EXHAUSTED' in error_str or '429' in error_str:
                    print(f"⚠️ Cuota API excedida, usando descriptor heurístico para {table.table_id}")
                else:
                    print(f"⚠️ Fallo LLM en descriptor tabla {table.table_id}: {type(e).__name__}")
                description = self._build_fallback_descriptor(table)
        else:
            description = self._build_fallback_descriptor(table)

        if len(description) < 50:
             description += f" [Contexto adicional: Tabla técnica sobre conservantes y alimentos extraída del documento {table.source_file}.]"

        column_types = self._infer_column_types(table)
        statistics = self._calculate_statistics(table, column_types)

        return TableDescriptor(
            descriptor_id=f"{table.table_id}_desc",
            table_id=table.table_id,
            doc_id=table.doc_id,
            page=table.page,
            description=description,
            num_rows=len(table.rows),
            num_cols=len(table.headers),
            headers=table.headers,
            column_types=column_types,
            statistics=statistics,
            type="table_descriptor",
            created_at=datetime.now()
        )

    def generate_summary(self, table: FullTable, method: str = "topk") -> TableSummary:
        """Genera resumen de tabla."""
        real_total_rows = len(table.rows)

        if real_total_rows == 0:
            return TableSummary(
                summary_id=f"{table.table_id}_summary_{method}",
                table_id=table.table_id,
                doc_id=table.doc_id,
                page=table.page,
                summary_text="Tabla vacía sin datos extraíbles.",
                sample_rows=[],
                sampling_method=method,
                total_rows=0,
                type="table_summary"
            )

        sample_size = min(self.topk, real_total_rows)
        sample_rows = table.rows[:sample_size]

        if self.use_llm:
            prompt = self._create_summary_prompt(table, sample_rows)
            try:
                summary_text = self._call_llm(prompt)
            except Exception as e:
                if 'RESOURCE_EXHAUSTED' not in str(e) and '429' not in str(e):
                    print(f"⚠️ Fallo LLM en resumen tabla {table.table_id}: {type(e).__name__}")
                summary_text = self._build_fallback_summary(table, sample_rows)
        else:
            summary_text = self._build_fallback_summary(table, sample_rows)

        summary_id = f"{table.table_id}_summary_{method}"
        
        return TableSummary(
            summary_id=summary_id,
            table_id=table.table_id,
            doc_id=table.doc_id,
            page=table.page,
            summary_text=summary_text,
            sample_rows=sample_rows,
            sampling_method=method,
            total_rows=real_total_rows,
            type="table_summary"
        )

    def _call_llm(self, prompt: str) -> str:
        """Helper para llamar al LLM."""
        messages = [
            SystemMessage(content="Eres un analista de datos experto. Describe brevemente el contenido de las tablas proporcionadas."),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    def _create_descriptor_prompt(self, table: FullTable) -> str:
        """Crea el prompt para describir la tabla globalmente."""
        headers_str = ", ".join(table.headers)
        preview_rows = table.rows[:3]
        preview_str = "\n".join([str(r.values) for r in preview_rows])
        
        return (
            f"Analiza la siguiente tabla del documento '{table.source_file}'.\n"
            f"Título/Caption: {table.caption}\n"
            f"Columnas: {headers_str}\n"
            f"Primeras filas de ejemplo:\n{preview_str}\n\n"
            "Tarea: Escribe un párrafo descriptivo (aprox 30-50 palabras) que explique QUÉ información contiene esta tabla "
            "para que un motor de búsqueda pueda encontrarla. Menciona las variables principales."
        )

    def _create_summary_prompt(self, table: FullTable, sample_rows: List[TableRow]) -> str:
        """Crea el prompt para resumir los datos."""
        rows_str = "\n".join([str(r.values) for r in sample_rows])
        return (
            f"Resume los siguientes datos de una tabla sobre '{table.caption}':\n"
            f"{rows_str}\n\n"
            "Tarea: Escribe un resumen narrativo de estos datos. ¿Qué tendencias o valores destacan?"
        )

    def _build_fallback_summary(self, table: FullTable, rows: List[TableRow]) -> str:
        """Resumen básico si falla el LLM."""
        headers = ", ".join(table.headers)
        return f"Tabla con {len(table.rows)} filas. Columnas: {headers}. Datos de muestra: {str(rows[0].values) if rows else 'N/A'}"

    def _build_fallback_descriptor(self, table: FullTable) -> str:
        """Genera una descripción heurística de la tabla sin usar LLM."""
        num_rows = len(table.rows)
        num_cols = len(table.headers)
        caption = table.caption or "sin título"

        parts = [
            f"Tabla extraída de {table.source_file} (página {table.page})",
            f"con {num_rows} filas y {num_cols} columnas."
        ]

        if table.caption and table.caption != "sin título":
            parts.append(f"Título: '{caption}'.")

        parts.append(f"Columnas: {', '.join(table.headers)}.")

        if table.rows:
            first_row = table.rows[0].values
            sample_str = "; ".join([f"{k}={v}" for k, v in list(first_row.items())[:3]])
            parts.append(f"Ejemplo de datos: {sample_str}.")

        return " ".join(parts)

    def _infer_column_types(self, table: FullTable) -> dict:
        """Infiere el tipo de dato de cada columna."""
        column_types = {}

        for header in table.headers:
            values = [row.values.get(header) for row in table.rows if row.values.get(header) is not None]

            if not values:
                column_types[header] = "unknown"
                continue

            sample_value = values[0]

            if isinstance(sample_value, (int, float)):
                column_types[header] = "numeric"
            elif isinstance(sample_value, bool):
                column_types[header] = "boolean"
            elif isinstance(sample_value, str):
                try:
                    _ = [float(v) for v in values if v]
                    column_types[header] = "numeric"
                except (ValueError, TypeError):
                    column_types[header] = "text"
            else:
                column_types[header] = "mixed"

        return column_types

    def _calculate_statistics(self, table: FullTable, column_types: dict) -> dict:
        """Calcula estadísticas básicas para columnas numéricas."""
        statistics = {}

        for header, col_type in column_types.items():
            if col_type != "numeric":
                continue

            try:
                values = []
                for row in table.rows:
                    val = row.values.get(header)
                    if val is not None:
                        try:
                            values.append(float(val))
                        except (ValueError, TypeError):
                            pass

                if values:
                    statistics[header] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "count": len(values)
                    }
            except Exception:
                pass

        return statistics if statistics else None