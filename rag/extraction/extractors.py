"""
Extractor de contenido heterogéneo de PDFs - VERSIÓN CORREGIDA.

Este módulo separa texto narrativo de tablas estructuradas,
evitando el problema de chunking indiscriminado.

CORRECCIONES:
- Añadido import re
- Añadidos métodos: is_bibliography_section, classify_chunks_by_type, process_pdf_with_bibliography_handling
"""

import re  # ← AÑADIDO
from pathlib import Path
from typing import List, Tuple, Optional, Any
import hashlib
from datetime import datetime

# Extracción de tablas
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("⚠️ camelot-py no disponible. Instalar con: pip install camelot-py[cv]")

# Extracción de texto
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modelos de validación
from rag.models import TextChunk, FullTable, TableRow


class PDFContentExtractor:
    """
    Extractor que separa texto de tablas en PDFs académicos.
    
    Flujo:
    1. Detecta tablas con camelot
    2. Extrae texto plano con pypdf
    3. Identifica qué regiones de texto corresponden a tablas (overlap)
    4. Retorna texto limpio + tablas estructuradas por separado
    """
    
    def __init__(
        self,
        table_extraction_method: str = "lattice",
        min_table_rows: int = 2,
        min_table_accuracy: float = 60.0
    ):
        """
        Args:
            table_extraction_method: 'lattice' (tablas con bordes) o 'stream' (sin bordes)
            min_table_rows: Mínimo de filas para considerar algo como tabla
            min_table_accuracy: Threshold de accuracy de camelot (0-100)
        """
        if not CAMELOT_AVAILABLE:
            raise RuntimeError("camelot-py required. Install: pip install camelot-py[cv]")
        
        self.table_method = table_extraction_method
        self.min_rows = min_table_rows
        self.min_accuracy = min_table_accuracy
    
    def extract_tables(self, pdf_path: Path, pages: str = "all") -> List[FullTable]:
        """
        Extrae tablas de un PDF.
        
        Args:
            pdf_path: Ruta al archivo PDF
            pages: Páginas a procesar ('all', '1,2,3', '1-5')
        
        Returns:
            Lista de FullTable validados
        """
        doc_id = self._generate_doc_id(pdf_path)
        tables_extracted = []
        
        try:
            # Camelot extraction
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor=self.table_method,
                strip_text='\n'
            )
            
            for i, table in enumerate(tables):
                # Filtrar por calidad
                if table.accuracy < self.min_accuracy:
                    print(f"⚠️ Tabla {i} en {pdf_path.name} descartada (accuracy: {table.accuracy:.1f}%)")
                    continue
                
                # Filtrar por tamaño
                if len(table.df) < self.min_rows:
                    continue
                
                # Convertir a FullTable
                try:
                    full_table = self._camelot_to_full_table(
                        table, doc_id, pdf_path.name, i
                    )
                    if full_table is not None:
                        tables_extracted.append(full_table)
                except Exception as e:
                    print(f"⚠️ Error validando tabla {i}: {e}")
                    continue
            
            print(f"✓ Extraídas {len(tables_extracted)} tablas de {pdf_path.name}")
            return tables_extracted
        
        except Exception as e:
            print(f"❌ Error extrayendo tablas de {pdf_path}: {e}")
            return []
    
    def _camelot_to_full_table(
        self, 
        camelot_table, 
        doc_id: str, 
        source_file: str, 
        table_idx: int
    ) -> FullTable:
        """Convierte objeto de camelot a FullTable validado."""
        df = camelot_table.df
        page = camelot_table.page
        
        # Headers: primera fila como headers
        headers = df.iloc[0].tolist()
        headers = [str(h).strip() or f"col_{i}" for i, h in enumerate(headers)]
        
        # Desduplicar headers
        seen = {}
        unique_headers = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                unique_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique_headers.append(h)
        
        # Rows: resto del dataframe
        rows = []
        for _, row in df.iloc[1:].iterrows():
            values = {h: self._clean_cell_value(v) for h, v in zip(unique_headers, row)}
            # Filtrar filas completamente vacías
            if any(v for v in values.values()):
                rows.append(TableRow(values=values))
        
        if not rows:
            print(f"  ⚠️ Tabla {table_idx} descartada: 0 filas válidas")
            return None
        
        # Bounding box
        bbox = {
            "x1": float(camelot_table._bbox[0]),
            "y1": float(camelot_table._bbox[1]),
            "x2": float(camelot_table._bbox[2]),
            "y2": float(camelot_table._bbox[3])
        }
        
        table_id = f"{doc_id}_table_{page}_{table_idx}"
        
        return FullTable(
            table_id=table_id,
            doc_id=doc_id,
            page=int(page),
            headers=unique_headers,
            rows=rows,
            caption=None,
            source_file=source_file,
            extraction_method=f"camelot_{self.table_method}",
            bbox=bbox
        )
    
    def _clean_cell_value(self, value) -> Any:
        """Limpia valores de celdas."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            # Intenta convertir a número
            try:
                if '.' in value or ',' in value:
                    return float(value.replace(',', ''))
                return int(value)
            except ValueError:
                return value
        return value
    
    def extract_text_excluding_tables(
        self, 
        pdf_path: Path, 
        tables: List[FullTable]
    ) -> str:
        """
        Extrae texto excluyendo regiones de tablas.
        
        Args:
            pdf_path: Ruta al PDF
            tables: Tablas ya extraídas (para identificar regiones a excluir)
        
        Returns:
            Texto limpio sin tablas
        """
        reader = PdfReader(pdf_path)
        text_by_page = {}
        
        # Extraer texto por página
        for page_num, page in enumerate(reader.pages, 1):
            text_by_page[page_num] = page.extract_text()
        
        # Identificar páginas con tablas
        pages_with_tables = {t.page for t in tables}

        # Excluir páginas con tablas del texto plano
        clean_text = []
        for page_num, text in text_by_page.items():
            if page_num in pages_with_tables:
                # Estrategia conservadora: excluir página completa
                print(f"⚠️ Página {page_num} excluida (contiene tabla)")
                continue
            clean_text.append(text)
        
        return "\n\n".join(clean_text)
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ) -> List[TextChunk]:
        """
        Divide texto en chunks validados.
        
        Args:
            text: Texto a dividir
            doc_id: ID del documento
            source_file: Nombre del archivo fuente
            chunk_size: Tamaño objetivo de chunk
            chunk_overlap: Overlap entre chunks
        
        Returns:
            Lista de TextChunk validados
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        raw_chunks = splitter.split_text(text)
        validated_chunks = []
        
        for i, chunk_text in enumerate(raw_chunks):
            # Estimar página (heurística: 3000 chars por página)
            page = (i * chunk_size) // 3000 + 1
            
            chunk_id = f"{doc_id}_{page}_{i}"
            char_count = len(chunk_text)
            word_count = len(chunk_text.split())
            
            # Filtrar chunks muy cortos
            if char_count < 50:
                continue
            
            try:
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk_text,
                    page=page,
                    char_count=char_count,
                    word_count=word_count,
                    source_file=source_file
                )
                validated_chunks.append(chunk)
            except Exception as e:
                print(f"⚠️ Chunk {i} no válido: {e}")
                continue
        
        return validated_chunks
    
    def process_pdf(
        self,
        pdf_path: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ) -> Tuple[List[TextChunk], List[FullTable]]:
        """
        Procesa un PDF completo: extrae tablas y texto por separado.
        
        Returns:
            (text_chunks, tables)
        """
        print(f"\n{'='*80}")
        print(f"Procesando: {pdf_path.name}")
        print(f"{'='*80}")
        
        doc_id = self._generate_doc_id(pdf_path)
        
        # 1. Extraer tablas
        tables = self.extract_tables(pdf_path)
        
        # 2. Extraer texto excluyendo tablas
        clean_text = self.extract_text_excluding_tables(pdf_path, tables)
        
        # 3. Chunk de texto
        text_chunks = self.chunk_text(
            clean_text, doc_id, pdf_path.name, chunk_size, chunk_overlap
        )
        
        print(f"✓ Resultado: {len(text_chunks)} chunks de texto, {len(tables)} tablas")
        return text_chunks, tables

    def is_bibliography_section(self, text: str) -> bool:
        """Detecta si un texto pertenece a la sección de bibliografía."""
        text_lower = text.lower().strip()
        
        # Heurística 1: Headers de bibliografía
        bibliography_markers = [
            r'\breferences\b',
            r'\bbibliography\b',
            r'\bworks cited\b',
            r'\bliteratura citada\b',
            r'\bbibliografía\b',
            r'\breferencias\b'
        ]
        
        # Si el texto COMIENZA con un marker (primeros 200 chars)
        for marker in bibliography_markers:
            if re.search(marker, text_lower[:200]):
                return True
        
        # Heurística 2: Densidad de años entre paréntesis
        year_pattern = r'\((\d{4})\)'
        year_matches = re.findall(year_pattern, text)
        
        # Si hay >5 años en un chunk típico (1000 chars), probablemente es bib
        year_density = len(year_matches) / max(len(text), 1) * 1000
        if year_density > 5:  # >5 años por cada 1000 caracteres
            return True
        
        # Heurística 3: Estructura de citación
        citation_patterns = [
            r'[A-Z][a-z]+,\s+[A-Z]\.\s*\(\d{4}\)',  # Smith, J. (2020)
            r'[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)',  # Smith et al. (2020)
            r'\[\d+\]\s+[A-Z][a-z]+',                 # [1] Smith...
        ]
        
        citation_count = sum(
            len(re.findall(pattern, text)) 
            for pattern in citation_patterns
        )
        
        # Si >30% de las líneas parecen citas
        lines = text.split('\n')
        if lines and citation_count / len(lines) > 0.3:
            return True
        
        return False
    
    def classify_chunks_by_type(
        self,
        chunks: List[TextChunk]
    ) -> Tuple[List[TextChunk], List[TextChunk]]:
        """
        Clasifica chunks en contenido vs bibliografía.
        
        Returns:
            (content_chunks, bibliography_chunks)
        """
        content_chunks = []
        bibliography_chunks = []
        
        for chunk in chunks:
            if self.is_bibliography_section(chunk.content):
                # Crear nuevo chunk con type="bibliography"
                chunk_dict = chunk.dict()
                chunk_dict['type'] = 'bibliography'
                bib_chunk = TextChunk(**chunk_dict)
                bibliography_chunks.append(bib_chunk)
            else:
                content_chunks.append(chunk)
        
        print(f"  Clasificados: {len(content_chunks)} contenido, {len(bibliography_chunks)} bibliografía")
        
        return content_chunks, bibliography_chunks
    
    def process_pdf_with_bibliography_handling(
        self,
        pdf_path: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        filter_bibliography: bool = False
    ) -> Tuple[List[TextChunk], List[FullTable], List[TextChunk]]:
        """
        Procesa PDF con manejo explícito de bibliografías.
        
        Args:
            pdf_path: Ruta al PDF
            chunk_size: Tamaño de chunks
            chunk_overlap: Overlap entre chunks
            filter_bibliography: Si True, separa bibliografía; si False, incluye todo
        
        Returns:
            (text_chunks, tables, bibliography_chunks)
            Si filter_bibliography=False, bibliography_chunks estará vacío
        """
        print(f"\n{'='*80}")
        print(f"Procesando: {pdf_path.name}")
        print(f"Filtro de bibliografía: {'ACTIVO' if filter_bibliography else 'INACTIVO'}")
        print(f"{'='*80}")
        
        doc_id = self._generate_doc_id(pdf_path)
        
        # 1. Extraer tablas
        tables = self.extract_tables(pdf_path)
        
        # 2. Extraer texto excluyendo tablas
        clean_text = self.extract_text_excluding_tables(pdf_path, tables)
        
        # 3. Chunk de texto
        all_text_chunks = self.chunk_text(
            clean_text, doc_id, pdf_path.name, chunk_size, chunk_overlap
        )
        
        # 4. Clasificar chunks si filter_bibliography=True
        if filter_bibliography:
            content_chunks, bibliography_chunks = self.classify_chunks_by_type(all_text_chunks)
            print(f"✓ Resultado: {len(content_chunks)} chunks contenido, "
                  f"{len(tables)} tablas, {len(bibliography_chunks)} chunks bibliografía")
            return content_chunks, tables, bibliography_chunks
        else:
            # No filtrar: devolver todo como contenido
            print(f"✓ Resultado: {len(all_text_chunks)} chunks (sin filtrar), "
                  f"{len(tables)} tablas")
            return all_text_chunks, tables, []
    
    def _generate_doc_id(self, pdf_path: Path) -> str:
        """Genera ID único para documento basado en nombre de archivo."""
        name = pdf_path.stem
        path_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        return f"{name}_{path_hash}"
