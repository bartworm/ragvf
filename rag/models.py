"""
Modelos Pydantic para validación estructural del pipeline RAG.

Este módulo define los esquemas de datos para:
- Chunks de texto validados
- Tablas completas (fuente de verdad)
- Descriptores de tablas (para retrieval)
- Resúmenes de tablas (Top-K sampling)
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import hashlib

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RAGResponse:
    """Objeto estandarizado de respuesta del RAG."""
    query: str
    answer: str
    sources: List[Any]  # Lista de documentos/tablas recuperados
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_processed: Optional[str] = None

class TextChunk(BaseModel):
    """
    Esquema para chunks de texto narrativo.
    
    Validaciones:
    - content debe tener entre 50-5000 caracteres
    - doc_id no puede ser 'unknown' o vacío
    - chunk_id debe ser único y seguir patrón
    """
    chunk_id: str = Field(..., description="ID único del chunk: {doc_id}_{page}_{idx}")
    doc_id: str = Field(..., min_length=1, description="ID del documento fuente")
    content: str = Field(..., min_length=50, max_length=5000, description="Contenido textual")
    page: int = Field(..., ge=1, description="Número de página")
    char_count: int = Field(..., ge=50)
    word_count: int = Field(..., ge=10)
    type: str = "text"
    source_file: str = Field(..., description="Nombre del archivo PDF original")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('content')
    def validate_content(cls, v):
        """Valida que el contenido no esté vacío después de strip."""
        clean = v.strip()
        if not clean:
            raise ValueError("Content cannot be empty or whitespace-only")
        if len(clean) < 50:
            raise ValueError(f"Content too short: {len(clean)} chars (min: 50)")
        return clean
    
    @validator('doc_id')
    def validate_doc_id(cls, v):
        """Valida que doc_id no sea placeholder."""
        if v.lower() in ['unknown', 'unk', 'none', '']:
            raise ValueError("doc_id cannot be placeholder value")
        return v
    
    @validator('chunk_id')
    def validate_chunk_id_format(cls, v):
        """Valida formato de chunk_id: debe contener doc_id_page_idx."""
        if '_' not in v or len(v.split('_')) < 3:
            raise ValueError(f"chunk_id must follow pattern 'docid_page_idx': {v}")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_counts(cls, values):
        """Valida consistencia entre content, char_count y word_count."""
        content = values.get('content', '')
        char_count = values.get('char_count', 0)
        word_count = values.get('word_count', 0)

        actual_chars = len(content)
        actual_words = len(content.split())

        if abs(actual_chars - char_count) > 10:  # tolerancia de 10 chars
            raise ValueError(f"char_count mismatch: expected ~{actual_chars}, got {char_count}")

        if abs(actual_words - word_count) > 5:  # tolerancia de 5 palabras
            raise ValueError(f"word_count mismatch: expected ~{actual_words}, got {word_count}")

        return values
    
    def generate_hash(self) -> str:
        """Genera hash único basado en contenido para deduplicación."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return f"{self.doc_id}_{content_hash[:16]}"


class TableRow(BaseModel):
    """
    Esquema para una fila de tabla.
    
    values: Diccionario {columna: valor}
    """
    values: Dict[str, Any] = Field(..., description="Valores de la fila indexados por columna")
    
    @validator('values')
    def validate_non_empty(cls, v):
        if not v:
            raise ValueError("TableRow cannot be empty")
        return v


class FullTable(BaseModel):
    """
    Esquema para tabla completa (fuente de verdad).
    
    Esta es la representación canónica que se persiste en Parquet.
    NO se embeddea directamente, solo sus representaciones derivadas.
    """
    table_id: str = Field(..., description="ID único de la tabla")
    doc_id: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    headers: List[str] = Field(..., min_items=1, description="Nombres de columnas")
    rows: List[TableRow] = Field(..., min_items=1, description="Filas de la tabla")
    caption: Optional[str] = Field(None, description="Título o caption de la tabla")
    source_file: str = Field(..., description="Archivo PDF original")
    extraction_method: str = Field(..., description="Método usado: camelot, tabula, etc.")
    bbox: Optional[Dict[str, float]] = Field(None, description="Bounding box de la tabla")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('headers')
    def validate_unique_headers(cls, v):
        """Valida que los headers sean únicos."""
        if len(v) != len(set(v)):
            duplicates = [h for h in v if v.count(h) > 1]
            raise ValueError(f"Duplicate headers found: {duplicates}")
        return v
    
    @validator('rows')
    def validate_rows_match_headers(cls, v, values):
        """Valida que todas las filas tengan las mismas columnas que headers."""
        headers = values.get('headers', [])
        expected_cols = set(headers)
        
        for i, row in enumerate(v):
            row_cols = set(row.values.keys())
            if row_cols != expected_cols:
                missing = expected_cols - row_cols
                extra = row_cols - expected_cols
                raise ValueError(
                    f"Row {i} schema mismatch. Missing: {missing}, Extra: {extra}"
                )
        return v
    
    def num_rows(self) -> int:
        return len(self.rows)
    
    def num_cols(self) -> int:
        return len(self.headers)
    
    def get_column_values(self, col_name: str) -> List[Any]:
        """Extrae valores de una columna específica."""
        if col_name not in self.headers:
            raise ValueError(f"Column '{col_name}' not found in headers")
        return [row.values[col_name] for row in self.rows]
    
    def to_dict_records(self) -> List[Dict[str, Any]]:
        """Convierte tabla a formato lista de diccionarios (para pandas)."""
        return [row.values for row in self.rows]


class TableDescriptor(BaseModel):
    """
    Descriptor semántico de tabla para indexación.
    
    Este es el texto que SE EMBEDDEA y se indexa en Qdrant.
    Permite retrieval eficiente sin cargar tabla completa.
    """
    descriptor_id: str = Field(..., description="ID del descriptor")
    table_id: str = Field(..., description="ID de la tabla fuente")
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    description: str = Field(..., min_length=10, max_length=2000, 
                            description="Descripción semántica generada")
    num_rows: int = Field(..., ge=1)
    num_cols: int = Field(..., ge=1)
    headers: List[str] = Field(...)
    column_types: Dict[str, str] = Field(..., description="Tipos inferidos por columna")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Stats para cols numéricas")
    type: Literal["table_descriptor"] = "table_descriptor"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('description')
    def validate_description_content(cls, v):
        """Valida que la descripción contenga información útil."""
        v = v.strip()
        if not v:
            raise ValueError("Description cannot be empty")

        # Debe mencionar estructura de tabla (en español o inglés)
        table_keywords = ['columna', 'fila', 'column', 'row', 'tabla', 'table', 'dato', 'data', 'contiene', 'contains']
        if not any(keyword in v.lower() for keyword in table_keywords):
            raise ValueError("Description must mention table structure")

        return v


class TableSummary(BaseModel):
    """
    Resumen de tabla con Top-K filas representativas.
    
    Esta representación SE EMBEDDEA y permite dar contexto
    sin cargar toda la tabla.
    """
    summary_id: str = Field(...)
    table_id: str = Field(...)
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    summary_text: str = Field(..., min_length=50, max_length=3000,
                             description="Texto del resumen con filas ejemplo")
    sample_rows: List[TableRow] = Field(..., min_items=0, max_items=10,
                                       description="Top-K filas representativas")
    sampling_method: str = Field(..., description="Método: topk, stratified, random")
    total_rows: int = Field(..., ge=0)
    type: Literal["table_summary"] = "table_summary"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('sample_rows')
    def validate_sample_size(cls, v, values):
        """Valida que sample no exceda total_rows."""
        total = values.get('total_rows', None)
        # Si total_rows aún no está disponible, skip validación
        if total is None:
            return v
        # Permitir tablas vacías (total=0, sample=[])
        if total == 0 and len(v) == 0:
            return v
        # Validar que sample no exceda total
        if total > 0 and len(v) > total:
            raise ValueError(f"Sample size {len(v)} exceeds total rows {total}")
        return v


class ProcessingMetadata(BaseModel):
    """
    Metadata de procesamiento para auditoría y reproducibilidad.
    """
    pipeline_version: str = Field(...)
    embedding_model: str = Field(...)
    chunk_size: int = Field(...)
    chunk_overlap: int = Field(...)
    table_extraction_method: str = Field(...)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    total_text_chunks: int = Field(...)
    total_tables: int = Field(...)
    total_descriptors: int = Field(...)
    total_summaries: int = Field(...)
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Serializa a dict para guardar en JSON."""
        return self.dict()


# Ejemplo de uso y validación
if __name__ == "__main__":
    # Ejemplo 1: TextChunk válido
    chunk = TextChunk(
        chunk_id="paper_2023_1_0",
        doc_id="paper_2023",
        content="Los preservantes son sustancias que se añaden a los alimentos para prevenir su deterioro. " * 5,
        page=1,
        char_count=400,
        word_count=60,
        source_file="preservantes_overview.pdf"
    )
    print(f"✓ TextChunk válido: {chunk.chunk_id}")
    
    # Ejemplo 2: FullTable válida
    table = FullTable(
        table_id="table_limits_eu_2023",
        doc_id="regulation_eu_2023",
        page=15,
        headers=["Preservante", "Límite_ppm", "pH_optimo"],
        rows=[
            TableRow(values={"Preservante": "Benzoato", "Límite_ppm": 1000, "pH_optimo": 3.5}),
            TableRow(values={"Preservante": "Sorbato", "Límite_ppm": 1000, "pH_optimo": 5.0}),
        ],
        caption="Límites legales de preservantes en UE",
        source_file="regulation_eu_2023.pdf",
        extraction_method="camelot"
    )
    print(f"✓ FullTable válida: {table.table_id}, {table.num_rows()} filas")
    
    # Ejemplo 3: Validación fallida (contenido muy corto)
    try:
        bad_chunk = TextChunk(
            chunk_id="bad_1_0",
            doc_id="test",
            content="Muy corto",  # < 50 chars
            page=1,
            char_count=10,
            word_count=2,
            source_file="test.pdf"
        )
    except ValueError as e:
        print(f"✓ Validación correcta: {e}")
