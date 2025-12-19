"""
NIVEL 0: Base Extractor - Extracción de texto plano
Simplemente extrae texto sin procesamiento avanzado.
"""

from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
from rag.models import TextChunk


class BaseExtractor:
    """
    Extractor básico de texto plano.
    Sin manejo de columnas, sin detección de bibliografía.
    """
    
    def __init__(self):
        self.name = "BaseExtractor"
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extrae texto plano de PDF.
        
        Args:
            pdf_path: Ruta al PDF
        
        Returns:
            Texto completo del PDF
        """
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ) -> List[TextChunk]:
        """
        Divide texto en chunks.
        
        Args:
            text: Texto a dividir
            doc_id: ID del documento
            source_file: Nombre del archivo
            chunk_size: Tamaño objetivo del chunk
            chunk_overlap: Overlap entre chunks
        
        Returns:
            Lista de TextChunk
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
            
            chunk_id = f"{doc_id}_chunk_{i}"
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
                    source_file=source_file,
                    type="content"  # Siempre content en nivel 0
                )
                validated_chunks.append(chunk)
            except Exception as e:
                print(f"⚠️ Chunk {i} no válido: {e}")
                continue
        
        return validated_chunks
    
    def process_pdf(
        self,
        pdf_path: Path,
        doc_id: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ) -> Tuple[List[TextChunk], dict]:
        """
        Procesa PDF completo.
        
        Args:
            pdf_path: Ruta al PDF
            doc_id: ID del documento
            chunk_size: Tamaño de chunks
            chunk_overlap: Overlap
        
        Returns:
            (chunks, metadata)
        """
        print(f"[NIVEL 0] Extrayendo texto plano de {pdf_path.name}")
        
        # Extraer texto
        text = self.extract_text(pdf_path)
        
        # Chunking
        chunks = self.chunk_text(
            text=text,
            doc_id=doc_id,
            source_file=pdf_path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        metadata = {
            "level": 0,
            "extractor": self.name,
            "total_chars": len(text),
            "num_chunks": len(chunks),
            "features": ["basic_text"]
        }
        
        print(f"✓ {len(chunks)} chunks extraídos")
        
        return chunks, metadata


# ========== TESTING ==========
if __name__ == "__main__":
    extractor = BaseExtractor()
    
    test_pdf = Path("/mnt/user-data/uploads/cf6896f938cab26391a2541c2ecada647fd2.pdf")
    
    if test_pdf.exists():
        chunks, metadata = extractor.process_pdf(
            pdf_path=test_pdf,
            doc_id="test_doc_001"
        )
        
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\n📝 Primer chunk:")
        print(chunks[0].content[:200] + "...")
    else:
        print(f"⚠️ PDF no encontrado: {test_pdf}")
