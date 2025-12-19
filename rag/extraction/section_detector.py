"""
Section Title Detector - Detección de títulos de secciones en PDFs

Detecta:
1. Títulos numerados (1. Introduction, 2.1 Methods)
2. Títulos en mayúsculas (INTRODUCTION, METHODS)
3. Títulos con formato especial (negrita, tamaño grande)
4. Estructura jerárquica (sección → subsección → subsubsección)
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pdfplumber


@dataclass
class SectionTitle:
    """Representa un título de sección detectado."""
    text: str
    level: int  # 1 = sección, 2 = subsección, 3 = subsubsección
    page: int
    numbering: Optional[str] = None  # ej: "2.1"
    start_char: int = 0  # Posición en texto completo
    confidence: float = 1.0  # 0-1
    
    def __str__(self):
        prefix = "  " * (self.level - 1)
        number = f"{self.numbering} " if self.numbering else ""
        return f"{prefix}{number}{self.text}"


class SectionTitleDetector:
    """
    Detecta títulos de secciones en PDFs.
    
    Estrategias:
    1. Patrones de numeración (1., 2.1, etc.)
    2. Formato de texto (MAYÚSCULAS, negrita)
    3. Heurísticas de estructura (líneas cortas, sin punto final)
    """
    
    def __init__(self):
        # Patrones de numeración
        self.numbered_patterns = [
            r'^(\d+)\.\s+',                    # "1. Introduction"
            r'^(\d+\.\d+)\s+',                 # "2.1 Methods"
            r'^(\d+\.\d+\.\d+)\s+',            # "2.1.1 Details"
            r'^\d+\.\s*([A-Z][A-Za-z\s]+)$',   # "1. INTRODUCTION"
            r'^Chapter\s+(\d+)',               # "Chapter 1"
            r'^Section\s+(\d+)',               # "Section 1"
        ]
        
        # Palabras comunes de inicio de sección
        self.section_keywords = [
            'introduction', 'abstract', 'background', 'methods',
            'methodology', 'results', 'discussion', 'conclusion',
            'conclusions', 'references', 'bibliography', 'appendix',
            'materials', 'experimental', 'analysis', 'implementation',
            'evaluation', 'related work', 'future work', 'acknowledgments'
        ]
    
    def detect_numbered_title(self, line: str) -> Optional[Tuple[str, int, str]]:
        """
        Detecta títulos numerados (1., 2.1, etc.).
        
        Args:
            line: Línea de texto
        
        Returns:
            (text, level, numbering) o None
        """
        line_clean = line.strip()
        
        # Patrón: "1." o "2.1" o "3.2.1"
        match = re.match(r'^(\d+(?:\.\d+)*)\.\s+(.+)$', line_clean)
        if match:
            numbering = match.group(1)
            text = match.group(2).strip()
            
            # Nivel basado en puntos: "1" -> nivel 1, "2.1" -> nivel 2
            level = numbering.count('.') + 1
            
            return text, level, numbering
        
        return None
    
    def detect_uppercase_title(self, line: str) -> Optional[Tuple[str, int]]:
        """
        Detecta títulos en MAYÚSCULAS.
        
        Args:
            line: Línea de texto
        
        Returns:
            (text, level) o None
        """
        line_clean = line.strip()
        
        # Debe ser corta (<100 chars), mayúsculas, sin punto final
        if (len(line_clean) < 100 and
            line_clean.isupper() and
            not line_clean.endswith('.') and
            len(line_clean.split()) >= 1):
            
            # Verificar si es keyword conocida
            line_lower = line_clean.lower()
            for keyword in self.section_keywords:
                if keyword in line_lower:
                    return line_clean, 1  # Nivel 1 si es keyword
            
            # Si no es keyword pero cumple criterios
            return line_clean, 1
        
        return None
    
    def detect_capitalized_title(self, line: str) -> Optional[Tuple[str, int]]:
        """
        Detecta títulos con primera letra mayúscula en cada palabra.
        
        Ejemplo: "Materials and Methods"
        
        Args:
            line: Línea de texto
        
        Returns:
            (text, level) o None
        """
        line_clean = line.strip()
        
        # Debe ser corta, sin punto final
        if (len(line_clean) < 100 and
            not line_clean.endswith('.') and
            len(line_clean.split()) >= 2):
            
            words = line_clean.split()
            
            # Verificar que la mayoría de palabras empiezan con mayúscula
            capitalized_count = sum(1 for w in words if w and w[0].isupper())
            if capitalized_count / len(words) >= 0.7:  # 70% capitalizadas
                
                # Verificar keywords
                line_lower = line_clean.lower()
                for keyword in self.section_keywords:
                    if keyword in line_lower:
                        return line_clean, 1
        
        return None
    
    def is_likely_body_text(self, line: str) -> bool:
        """
        Heurística: ¿es texto del cuerpo (no título)?
        
        Args:
            line: Línea de texto
        
        Returns:
            True si parece texto de cuerpo
        """
        line_clean = line.strip()
        
        # Demasiado larga → cuerpo
        if len(line_clean) > 120:
            return True
        
        # Termina en punto → probablemente cuerpo
        if line_clean.endswith('.'):
            return True
        
        # Tiene minúsculas al inicio (excepto "a", "and") → cuerpo
        words = line_clean.split()
        if words and words[0] and words[0][0].islower():
            return True
        
        return False
    
    def extract_sections_from_text(
        self,
        text: str,
        page_num: int = 1
    ) -> List[SectionTitle]:
        """
        Extrae títulos de secciones de texto plano.
        
        Args:
            text: Texto a analizar
            page_num: Número de página
        
        Returns:
            Lista de SectionTitle detectadas
        """
        sections = []
        lines = text.split('\n')
        char_position = 0
        
        for line in lines:
            line_clean = line.strip()
            
            if not line_clean:
                char_position += len(line) + 1
                continue
            
            # Filtrar texto de cuerpo obvio
            if self.is_likely_body_text(line_clean):
                char_position += len(line) + 1
                continue
            
            # ESTRATEGIA 1: Títulos numerados
            result = self.detect_numbered_title(line_clean)
            if result:
                text, level, numbering = result
                section = SectionTitle(
                    text=text,
                    level=level,
                    page=page_num,
                    numbering=numbering,
                    start_char=char_position,
                    confidence=0.95
                )
                sections.append(section)
                char_position += len(line) + 1
                continue
            
            # ESTRATEGIA 2: MAYÚSCULAS
            result = self.detect_uppercase_title(line_clean)
            if result:
                text, level = result
                section = SectionTitle(
                    text=text,
                    level=level,
                    page=page_num,
                    start_char=char_position,
                    confidence=0.80
                )
                sections.append(section)
                char_position += len(line) + 1
                continue
            
            # ESTRATEGIA 3: Capitalizadas (Title Case)
            result = self.detect_capitalized_title(line_clean)
            if result:
                text, level = result
                section = SectionTitle(
                    text=text,
                    level=level,
                    page=page_num,
                    start_char=char_position,
                    confidence=0.70
                )
                sections.append(section)
            
            char_position += len(line) + 1
        
        return sections
    
    def extract_sections_from_pdf(
        self,
        pdf_path: str
    ) -> List[SectionTitle]:
        """
        Extrae títulos de secciones directamente de PDF.
        
        Usa pdfplumber para mejor detección de formato.
        
        Args:
            pdf_path: Ruta al PDF
        
        Returns:
            Lista de SectionTitle
        """
        all_sections = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                
                # Extraer secciones de esta página
                sections = self.extract_sections_from_text(text, page_num)
                all_sections.extend(sections)
        
        return all_sections
    
    def build_section_hierarchy(
        self,
        sections: List[SectionTitle]
    ) -> Dict[str, any]:
        """
        Construye jerarquía de secciones.
        
        Args:
            sections: Lista de secciones detectadas
        
        Returns:
            Dict con estructura jerárquica
        """
        hierarchy = {
            "title": "Document",
            "sections": []
        }
        
        current_l1 = None
        current_l2 = None
        
        for section in sections:
            section_dict = {
                "title": section.text,
                "level": section.level,
                "page": section.page,
                "numbering": section.numbering,
                "subsections": []
            }
            
            if section.level == 1:
                hierarchy["sections"].append(section_dict)
                current_l1 = section_dict
                current_l2 = None
            
            elif section.level == 2 and current_l1:
                current_l1["subsections"].append(section_dict)
                current_l2 = section_dict
            
            elif section.level == 3 and current_l2:
                current_l2["subsections"].append(section_dict)
        
        return hierarchy


# ========== TESTING ==========
if __name__ == "__main__":
    from pathlib import Path
    
    detector = SectionTitleDetector()
    
    # Test con texto de ejemplo
    sample_text = """
INTRODUCTION
This paper presents...

1. Background
The field of natural language processing...

2. Related Work
Previous studies have shown...

2.1 Traditional Approaches
Early methods relied on...

2.2 Modern Techniques
Recent advances in deep learning...

3. METHODOLOGY
Our approach consists of...

3.1. Data Collection
We collected data from...

CONCLUSION
In this work we have...
    """
    
    print("="*60)
    print("TEST 1: Detección en texto plano")
    print("="*60)
    
    sections = detector.extract_sections_from_text(sample_text, page_num=1)
    
    print(f"\n✓ Detectadas {len(sections)} secciones:\n")
    for section in sections:
        print(section)
    
    # Test con PDF real
    test_pdf = Path("/mnt/user-data/uploads/cf6896f938cab26391a2541c2ecada647fd2.pdf")
    
    if test_pdf.exists():
        print("\n" + "="*60)
        print("TEST 2: Detección en PDF real")
        print("="*60)
        
        sections = detector.extract_sections_from_pdf(str(test_pdf))
        
        print(f"\n✓ Detectadas {len(sections)} secciones:\n")
        for section in sections[:10]:  # Primeras 10
            print(section)
        
        # Construir jerarquía
        hierarchy = detector.build_section_hierarchy(sections)
        
        print("\n" + "="*60)
        print("TEST 3: Jerarquía de secciones")
        print("="*60)
        print(f"\n✓ {len(hierarchy['sections'])} secciones principales\n")
    else:
        print(f"\n⚠️ PDF no encontrado: {test_pdf}")
