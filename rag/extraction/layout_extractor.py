"""
NIVEL 2: Layout Extractor - Manejo de columnas por secciones
Hereda de BibliographyExtractor y añade detección inteligente de columnas.

MEJORA: Detecta cambios de columnas DENTRO de cada página.
Maneja: [1 col] → [2 cols] → [1 col]
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pdfplumber
import numpy as np
from dataclasses import dataclass
import sys
from rag.models import TextChunk
from rag.extraction.bibliography_extractor import BibliographyExtractor


@dataclass
class Section:
    """Representa una sección de página con número de columnas."""
    y_start: float
    y_end: float
    num_columns: int
    text: str


class LayoutExtractor(BibliographyExtractor):
    """
    Extractor con manejo inteligente de columnas por secciones.
    
    Detecta si una página tiene:
    - Título (1 columna)
    - Contenido (2 columnas)
    - Conclusión (1 columna)
    
    Y extrae texto en el orden correcto para cada sección.
    """
    
    def __init__(self, detect_sections: bool = False, section_height: float = 100.0):
        super().__init__(detect_sections=detect_sections)
        self.name = "LayoutExtractor"
        self.section_height = section_height

    def detect_sections_with_columns(self, page) -> List[Section]:
        """Divide página en secciones horizontales y detecta columnas en cada una."""
        words = page.extract_words()
        
        if not words:
            return [Section(
                y_start=0,
                y_end=page.height,
                num_columns=1,
                text=""
            )]
        
        page_height = page.height
        page_width = page.width
        
        # 1. Dividir página en bandas horizontales
        num_sections = max(1, int(page_height / self.section_height))
        sections = []
        
        for section_idx in range(num_sections):
            y_start = section_idx * self.section_height
            y_end = min((section_idx + 1) * self.section_height, page_height)
            
            # 2. Extraer palabras de esta sección
            section_words = [
                w for w in words 
                if y_start <= w['top'] < y_end
            ]
            
            if not section_words:
                # Sección vacía
                sections.append(Section(
                    y_start=y_start,
                    y_end=y_end,
                    num_columns=1,
                    text=""
                ))
                continue
            
            # 3. Detectar columnas en esta sección
            num_cols = self._count_columns_in_section(
                section_words, page_width
            )
            
            # 4. Extraer texto respetando columnas
            text = self._extract_section_text(
                page, y_start, y_end, num_cols, page_width
            )
            
            sections.append(Section(
                y_start=y_start,
                y_end=y_end,
                num_columns=num_cols,
                text=text
            ))
        
        # 5. Consolidar secciones consecutivas con mismo número de columnas
        consolidated = self._consolidate_sections(sections)
        
        return consolidated
    
    def _count_columns_in_section(self, words: List[Dict], page_width: float) -> int:
        """
        Cuenta el número de columnas en una sección analizando distribución de X.
        
        Estrategia:
        1. Agrupar palabras por posición X
        2. Si hay 2+ clusters → múltiples columnas
        3. Si hay 1 cluster → 1 columna
        """
        if not words:
            return 1
        
        x_positions = [w['x0'] for w in words]
        
        # Agrupar posiciones X cercanas
        x_clusters = self._cluster_positions(x_positions, threshold=50)
        
        num_clusters = len(x_clusters)
        
        # Heurística: mínimo 10% de ancho por columna
        min_col_width = page_width * 0.1
        
        # Filtrar clusters muy cercanos (posibles ruido)
        valid_clusters = []
        for cluster in x_clusters:
            cluster_width = max(cluster) - min(cluster)
            if cluster_width >= min_col_width or len(valid_clusters) == 0:
                valid_clusters.append(cluster)
        
        return min(len(valid_clusters), 3)  # Máximo 3 columnas
    
    def _cluster_positions(self, positions: List[float], threshold: float = 50) -> List[List[float]]:
        """
        Agrupa posiciones que estén cercanas (dentro de threshold).
        
        Args:
            positions: Lista de posiciones X
            threshold: Distancia máxima entre posiciones del mismo cluster
        
        Returns:
            Lista de clusters [[x1, x2, ...], [x4, x5, ...], ...]
        """
        if not positions:
            return []
        
        # Eliminar duplicados y ordenar
        positions = sorted(set(positions))
        
        if len(positions) == 1:
            return [positions]
        
        # Clustering jerárquico simple
        clusters = [[positions[0]]]
        
        for pos in positions[1:]:
            # Distancia al cluster más cercano
            last_cluster = clusters[-1]
            dist_to_last = pos - max(last_cluster)
            
            if dist_to_last < threshold:
                # Agregar al cluster actual
                last_cluster.append(pos)
            else:
                # Nuevo cluster
                clusters.append([pos])
        
        return clusters
    
    def _extract_section_text(
            self, 
            page, 
            y_start: float, 
            y_end: float, 
            num_columns: int,
            page_width: float
        ) -> str:
            """
            Extrae texto de una sección respetando columnas.
            CORREGIDO: Usa coordenadas absolutas para evitar el error de Bounding Box.
            """
            # Validar coordenadas para evitar errores de redondeo o fuera de límites
            y_start = max(0, float(y_start))
            y_end = min(float(page.height), float(y_end))
            
            if y_start >= y_end:
                return ""

            if num_columns == 1:
                # 1 columna: Crop directo de la sección
                # Coordenadas: (x0, top, x1, bottom)
                section_page = page.crop((0, y_start, page_width, y_end))
                return section_page.extract_text() or ""
            
            elif num_columns == 2:
                # 2 columnas: izquierda → derecha
                mid_x = page_width / 2
                
                # Columna izquierda (Usamos coordenadas ABSOLUTAS: y_start, no 0)
                try:
                    left_page = page.crop((0, y_start, mid_x, y_end))
                    left_text = left_page.extract_text() or ""
                except Exception:
                    left_text = ""
                
                # Columna derecha
                try:
                    right_page = page.crop((mid_x, y_start, page_width, y_end))
                    right_text = right_page.extract_text() or ""
                except Exception:
                    right_text = ""
                
                return f"{left_text}\n\n{right_text}"
            
            elif num_columns == 3:
                # 3 columnas: iteramos calculando coordenadas absolutas
                col_width = page_width / 3
                texts = []
                for col_idx in range(3):
                    x0 = col_idx * col_width
                    x1 = (col_idx + 1) * col_width
                    try:
                        col_page = page.crop((x0, y_start, x1, y_end))
                        text = col_page.extract_text() or ""
                        texts.append(text)
                    except Exception:
                        continue
                
                return "\n\n".join(texts)
            
            else:
                # Fallback seguro
                try:
                    section_page = page.crop((0, y_start, page_width, y_end))
                    return section_page.extract_text() or ""
                except Exception:
                    return ""
    
    def _consolidate_sections(self, sections: List[Section]) -> List[Section]:
        """
        Consolida secciones consecutivas que tengan el mismo número de columnas.
        
        Ejemplo:
            [1col, 1col, 2col, 2col, 2col, 1col]
            →
            [2col(0-200), 6col(200-600), 1col(600-700)]
        """
        if not sections:
            return []
        
        consolidated = [sections[0]]
        
        for section in sections[1:]:
            last = consolidated[-1]
            
            # Si misma cantidad de columnas, expandir sección anterior
            if section.num_columns == last.num_columns:
                last.y_end = section.y_end
                last.text += "\n\n" + section.text
            else:
                # Nueva sección
                consolidated.append(section)
        
        return consolidated
    
    def extract_text_with_layout(self, pdf_path: Path) -> str:
        """
        Extrae texto completo respetando columnas por sección.
        
        Maneja PDFs donde:
        - Portada: 1 columna
        - Contenido: 2 columnas
        - Conclusión: 1 columna
        
        Retorna en orden correcto de lectura.
        """
        full_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"  Página {page_num}...", end="", flush=True)
                
                # Detectar secciones con columnas
                sections = self.detect_sections_with_columns(page)
                
                # Extraer texto de cada sección
                page_text = []
                for section in sections:
                    if section.text.strip():
                        page_text.append(section.text)
                
                if page_text:
                    full_text.append("\n".join(page_text))
                
                # Log del layout
                layout = " → ".join([f"{s.num_columns}col" for s in sections])
                print(f" [{layout}]")
        
        return "\n\n".join(full_text)
    
    def process_pdf(
        self,
        pdf_path: Path,
        doc_id: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        use_layout_detection: bool = True,
        use_title_detection: bool = True,
        use_heuristics_fallback: bool = True
    ) -> Tuple[List[TextChunk], List[TextChunk], dict]:
        """
        Procesa PDF con detección inteligente de columnas por secciones.
        
        Args:
            pdf_path: Ruta al PDF
            doc_id: ID del documento
            chunk_size: Tamaño de chunks
            chunk_overlap: Overlap
            use_layout_detection: Detectar columnas por sección
            use_title_detection: Buscar título "References"
            use_heuristics_fallback: Usar heurísticas si falla detección
        
        Returns:
            (content_chunks, bibliography_chunks, metadata)
        """
        print(f"[NIVEL 2] Procesando {pdf_path.name} con detección de columnas por secciones")
        
        # 1. Detectar página de inicio de bibliografía (heredado)
        bib_start_page = None
        if use_title_detection:
            bib_start_page = self.find_bibliography_start_page(pdf_path)
        
        # 2. Extraer texto respetando columnas por sección
        if use_layout_detection:
            try:
                print("\n📄 Analizando estructura de columnas por secciones...")
                text = self.extract_text_with_layout(pdf_path)
                layout_method = "pdfplumber_sections"
            except Exception as e:
                print(f"\n  ⚠️ Error en detección de secciones, usando pypdf: {e}")
                text = super().extract_text(pdf_path)  # Fallback a BaseExtractor
                layout_method = "pypdf_fallback"
        else:
            text = super().extract_text(pdf_path)
            layout_method = "pypdf"
        
        # 3. Chunking (heredado)
        all_chunks = self.chunk_text(
            text=text,
            doc_id=doc_id,
            source_file=pdf_path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 4. Clasificar bibliografía (heredado)
        if bib_start_page is not None:
            content_chunks, bib_chunks = self.classify_chunks_by_page(
                all_chunks, bib_start_page
            )
            detection_method = "title"
        elif use_heuristics_fallback:
            content_chunks = []
            bib_chunks = []
            for chunk in all_chunks:
                if self.is_bibliography_heuristic(chunk.content):
                    chunk_dict = chunk.dict()
                    chunk_dict['type'] = 'bibliography'
                    bib_chunks.append(TextChunk(**chunk_dict))
                else:
                    content_chunks.append(chunk)
            detection_method = "heuristic"
        else:
            content_chunks = all_chunks
            bib_chunks = []
            detection_method = "none"
        
        # 5. Metadata
        metadata = {
            "level": 2,
            "extractor": self.name,
            "total_chars": len(text),
            "num_content_chunks": len(content_chunks),
            "num_bib_chunks": len(bib_chunks),
            "bib_start_page": bib_start_page,
            "detection_method": detection_method,
            "layout_method": layout_method,
            "section_height": self.section_height,
            "features": ["basic_text", "bibliography_detection", "layout_awareness", "section_columns"]
        }
        
        print(f"\n✓ {len(content_chunks)} chunks de contenido")
        print(f"✓ {len(bib_chunks)} chunks de bibliografía")
        
        return content_chunks, bib_chunks, metadata


# ========== TESTING ==========
if __name__ == "__main__":
    from pathlib import Path
    
    extractor = LayoutExtractor(section_height=120.0)
    
    test_pdf = Path("/mnt/user-data/uploads/cf6896f938cab26391a2541c2ecada647fd2.pdf")
    
    if test_pdf.exists():
        # TEST 1: Análisis de secciones
        print("\n" + "="*80)
        print("TEST 1: ANÁLISIS DE SECCIONES CON COLUMNAS")
        print("="*80)
        
        with pdfplumber.open(test_pdf) as pdf:
            for page_num, page in enumerate(pdf.pages[:2], 1):  # Primeras 2 páginas
                print(f"\n📄 Página {page_num}:")
                
                sections = extractor.detect_sections_with_columns(page)
                
                for i, section in enumerate(sections):
                    print(f"  Sección {i+1}: Y[{section.y_start:.0f}-{section.y_end:.0f}] "
                          f"→ {section.num_columns} columna(s)")
                    print(f"    Texto: {section.text[:80].replace(chr(10), ' ')}...")
        
        # TEST 2: Extracción completa
        print("\n" + "="*80)
        print("TEST 2: EXTRACCIÓN COMPLETA CON COLUMNAS")
        print("="*80)
        
        content_chunks, bib_chunks, metadata = extractor.process_pdf(
            pdf_path=test_pdf,
            doc_id="test_doc_001",
            use_layout_detection=True,
            use_title_detection=True,
            use_heuristics_fallback=True
        )
        
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        if content_chunks:
            print(f"\n📝 Primer chunk de contenido:")
            print(content_chunks[0].content[:300] + "...")
    else:
        print(f"⚠️ PDF no encontrado: {test_pdf}")