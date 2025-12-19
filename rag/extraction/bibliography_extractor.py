"""
NIVEL 1: Bibliography Extractor - Detección mejorada de bibliografía
Hereda de BaseExtractor y añade:
1. Detección de bibliografía por título
2. Detección de títulos de secciones (opcional)
"""

from pathlib import Path
from typing import List, Tuple, Optional
import re
from pypdf import PdfReader
import sys
from rag.models import TextChunk
from rag.extraction.base_extractor import BaseExtractor
from rag.extraction.section_detector import SectionTitleDetector, SectionTitle


class BibliographyExtractor(BaseExtractor):
    """
    Extractor con detección de bibliografía mejorada.
    
    Estrategia:
    1. Busca título "References", "Bibliography", etc. en cada página
    2. TODO después de esa página → bibliografía
    3. Fallback: heurísticas de contenido si no encuentra título
    """
    
    def __init__(self, detect_sections: bool = False):
        super().__init__()
        self.name = "BibliographyExtractor"
        self.detect_sections = detect_sections
        
        # Detector de títulos de secciones
        if detect_sections:
            self.section_detector = SectionTitleDetector()
        else:
            self.section_detector = None
        
        # Títulos de bibliografía (case-insensitive)
        self.bibliography_titles = [
            r'^references$',
            r'^bibliography$',
            r'^works\s+cited$',
            r'^referencias$',
            r'^bibliografía$',
            r'^literatura\s+citada$',
            r'^\d+\.\s*references$',  # "5. References"
            r'^\d+\.\s*bibliography$',
        ]
        
        # Compilar patrones
        self.title_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.bibliography_titles
        ]
    
    def find_bibliography_start_page(self, pdf_path: Path) -> Optional[int]:
        """
        Encuentra la página donde comienza la bibliografía.
        
        Busca títulos como "References", "Bibliography" en las páginas.
        
        Args:
            pdf_path: Ruta al PDF
        
        Returns:
            Número de página (1-indexed) o None si no se encuentra
        """
        reader = PdfReader(pdf_path)
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            
            # Dividir en líneas
            lines = text.split('\n')
            
            for line in lines:
                line_clean = line.strip()
                
                # Buscar título exacto
                for pattern in self.title_patterns:
                    if pattern.match(line_clean):
                        print(f"   Bibliografía detectada en página {page_num}: '{line_clean}'")
                        return page_num
                
                # También buscar títulos en mayúsculas (formato heading)
                if len(line_clean) < 50:  # Títulos son cortos
                    line_upper = line_clean.upper()
                    if any(marker in line_upper for marker in ['REFERENCES', 'BIBLIOGRAPHY', 'REFERENCIAS', 'BIBLIOGRAFÍA']):
                        # Verificar que no es parte de una oración
                        if not line_clean.endswith('.') or line_clean.count('.') == 1:
                            print(f"   Bibliografía detectada en página {page_num}: '{line_clean}'")
                            return page_num
        
        print("  Warning: No se detectó página de inicio de bibliografía")
        return None
    
    def is_bibliography_heuristic(self, text: str) -> bool:
        """
        Heurísticas de contenido (FALLBACK).
        Detecta si un chunk parece bibliografía por su contenido.
        
        Args:
            text: Texto a analizar
        
        Returns:
            True si parece bibliografía
        """
        text_lower = text.lower().strip()
        
        # Heurística 1: Headers de bibliografía
        bibliography_markers = [
            r'\breferences\b', r'\bbibliography\b',
            r'\breferencias\b', r'\bbibliografía\b'
        ]
        for marker in bibliography_markers:
            if re.search(marker, text_lower[:200]):
                return True
        
        # Heurística 2: Densidad de años (2020), (2021)
        year_matches = re.findall(r'\((\d{4})\)', text)
        year_density = len(year_matches) / max(len(text), 1) * 1000
        if year_density > 5:  # >5 años por cada 1000 caracteres
            return True
        
        # Heurística 3: Patrones de citación
        citation_patterns = [
            r'[A-Z][a-z]+,\s+[A-Z]\.\s*\(\d{4}\)',  # Smith, J. (2020)
            r'[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)',  # Smith et al. (2020)
            r'\[\d+\]\s+[A-Z][a-z]+'  # [1] Smith...
        ]
        citation_count = sum(len(re.findall(p, text)) for p in citation_patterns)
        lines = text.split('\n')
        if lines and citation_count / len(lines) > 0.3:
            return True
        
        return False
    
    def classify_chunks_by_page(
        self,
        chunks: List[TextChunk],
        bibliography_start_page: Optional[int]
    ) -> Tuple[List[TextChunk], List[TextChunk]]:
        """
        Clasifica chunks en contenido vs bibliografía por página.
        
        Args:
            chunks: Lista de TextChunk
            bibliography_start_page: Página donde comienza bibliografía
        
        Returns:
            (content_chunks, bibliography_chunks)
        """
        if bibliography_start_page is None:
            return chunks, []
        
        content_chunks = []
        bibliography_chunks = []
        
        for chunk in chunks:
            if chunk.page >= bibliography_start_page:
                # Chunk en página de bibliografía o posterior
                chunk_dict = chunk.dict()
                chunk_dict['type'] = 'bibliography'
                bib_chunk = TextChunk(**chunk_dict)
                bibliography_chunks.append(bib_chunk)
            else:
                content_chunks.append(chunk)
        
        print(f"  Clasificados por página: {len(content_chunks)} contenido, "
              f"{len(bibliography_chunks)} bibliografía")
        
        return content_chunks, bibliography_chunks
    
    def extract_sections(
        self,
        pdf_path: Path
    ) -> Optional[List[SectionTitle]]:
        """
        Extrae títulos de secciones del PDF.
        
        Args:
            pdf_path: Ruta al PDF
        
        Returns:
            Lista de SectionTitle o None si detect_sections=False
        """
        if not self.detect_sections or self.section_detector is None:
            return None
        
        try:
            sections = self.section_detector.extract_sections_from_pdf(str(pdf_path))
            print(f"   Detectadas {len(sections)} secciones")
            return sections
        except Exception as e:
            print(f"  Warning: Error detectando secciones: {e}")
            return None
    
    def process_pdf(
        self,
        pdf_path: Path,
        doc_id: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        use_title_detection: bool = True,
        use_heuristics_fallback: bool = True
    ) -> Tuple[List[TextChunk], List[TextChunk], dict]:
        """
        Procesa PDF con detección de bibliografía.
        
        Args:
            pdf_path: Ruta al PDF
            doc_id: ID del documento
            chunk_size: Tamaño de chunks
            chunk_overlap: Overlap
            use_title_detection: Si True, busca título "References"
            use_heuristics_fallback: Si True, usa heurísticas si falla detección
        
        Returns:
            (content_chunks, bibliography_chunks, metadata)
        """
        print(f"[NIVEL 1] Procesando {pdf_path.name} con detección de bibliografía")
        
        # 1. Detectar secciones (opcional)
        sections = None
        if self.detect_sections:
            sections = self.extract_sections(pdf_path)
        
        # 2. Detectar página de inicio de bibliografía
        bib_start_page = None
        if use_title_detection:
            bib_start_page = self.find_bibliography_start_page(pdf_path)
        
        # 2. Extraer texto y chunkear (heredado de BaseExtractor)
        text = self.extract_text(pdf_path)
        all_chunks = self.chunk_text(
            text=text,
            doc_id=doc_id,
            source_file=pdf_path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 3. Clasificar chunks
        if bib_start_page is not None:
            # Clasificación por página (MÉTODO PRINCIPAL)
            content_chunks, bib_chunks = self.classify_chunks_by_page(
                all_chunks, bib_start_page
            )
            detection_method = "title"
        
        elif use_heuristics_fallback:
            # Clasificación por heurísticas (FALLBACK)
            print("  Warning: Usando heurísticas de contenido (fallback)")
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
            # Sin clasificación
            content_chunks = all_chunks
            bib_chunks = []
            detection_method = "none"
        
        # 4. Metadata
        metadata = {
            "level": 1,
            "extractor": self.name,
            "total_chars": len(text),
            "num_content_chunks": len(content_chunks),
            "num_bib_chunks": len(bib_chunks),
            "bib_start_page": bib_start_page,
            "detection_method": detection_method,
            "features": ["basic_text", "bibliography_detection"]
        }
        
        # Añadir info de secciones si están disponibles
        if sections:
            metadata["num_sections"] = len(sections)
            metadata["features"].append("section_detection")
            metadata["sections"] = [
                {
                    "text": s.text,
                    "level": s.level,
                    "page": s.page,
                    "numbering": s.numbering
                }
                for s in sections
            ]
        
        print(f" {len(content_chunks)} chunks de contenido")
        print(f" {len(bib_chunks)} chunks de bibliografía")
        
        return content_chunks, bib_chunks, metadata


# ========== TESTING ==========
if __name__ == "__main__":
    extractor = BibliographyExtractor()
    
    test_pdf = Path("/mnt/user-data/uploads/cf6896f938cab26391a2541c2ecada647fd2.pdf")
    
    if test_pdf.exists():
        content_chunks, bib_chunks, metadata = extractor.process_pdf(
            pdf_path=test_pdf,
            doc_id="test_doc_001",
            use_title_detection=True,
            use_heuristics_fallback=True
        )
        
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\n📝 Primer chunk de contenido:")
        print(content_chunks[0].content[:200] + "...")
        
        if bib_chunks:
            print(f"\n📚 Primer chunk de bibliografía:")
            print(bib_chunks[0].content[:200] + "...")
    else:
        print(f"Warning: PDF no encontrado: {test_pdf}")
