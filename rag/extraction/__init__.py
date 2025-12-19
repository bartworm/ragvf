

# 1. Importamos primero los componentes base para evitar ciclos
from .base_extractor import BaseExtractor
from .bibliography_extractor import BibliographyExtractor
from .layout_extractor import LayoutExtractor
from .table_extractor import TableExtractor

# 2. Importamos el orquestador (pipeline.py)
# Asegúrate de haber renombrado tu archivo a 'pipeline.py'
from .pipeline import UnifiedPDFExtractor, ExtractionLevel

def extract_from_pdf(pdf_path, level=3, **kwargs):
    """
    Función helper para facilitar el uso desde fuera.
    """
    extractor = UnifiedPDFExtractor(level=level)
    return extractor.process_pdf(pdf_path, doc_id=pdf_path.stem, **kwargs)