import re
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from docx import Document

def _extract_from_pdf(file_path: Path) -> str:
    """Internal helper to read PDF."""
    text_parts = []
    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text_parts.append(content)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error reading PDF {file_path.name}: {e}")
        return ""

def _extract_from_docx(file_path: Path) -> str:
    """Internal helper to read DOCX."""
    try:
        doc = Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {file_path.name}: {e}")
        return ""

def _clean_text(text: str) -> str:
    """Normalizes whitespace."""
    if not text:
        return ""
    return re.sub(r'[ \t]+', ' ', text).strip()

def load_and_clean_document(file_path_str: str) -> str:
    """
    Public API: Loads a file and returns clean text.
    Handles file validation and format detection.
    """
    path = Path(file_path_str)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = path.suffix.lower()
    
    if ext == '.pdf':
        raw_text = _extract_from_pdf(path)
    elif ext == '.docx':
        raw_text = _extract_from_docx(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
        
    return _clean_text(raw_text)