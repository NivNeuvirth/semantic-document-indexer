import logging
import re
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

def _extract_from_pdf(file_path: Path) -> Optional[str]:
    """
    Internal helper to read PDF.
    Returns: extracted text (str) or None if extraction failed.
    """
    text_parts = []
    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text_parts.append(content)
        return "\n".join(text_parts)
    except Exception:
        logger.exception(f"Failed to parse PDF file: {file_path.name}")
        return None

def _extract_from_docx(file_path: Path) -> Optional[str]:
    """
    Internal helper to read DOCX.
    Returns: extracted text (str) or None if extraction failed.
    """
    try:
        doc = Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        logger.exception(f"Failed to parse DOCX file: {file_path.name}")
        return None

def _clean_text(text: str) -> str:
    """Normalizes whitespace."""
    if not text:
        return ""
    return re.sub(r'[ \t]+', ' ', text).strip()

def load_and_clean_document(file_path_str: str) -> str:
    """
    Public API: Loads a file and returns clean text.
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If format is unsupported.
        RuntimeError: If parsing failed.
    """
    path = Path(file_path_str)
    
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = path.suffix.lower()
    
    raw_text: Optional[str] = None
    
    if ext == '.pdf':
        raw_text = _extract_from_pdf(path)
    elif ext == '.docx':
        raw_text = _extract_from_docx(path)
    else:
        logger.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported format: {ext}")
    
    if raw_text is None:
        raise RuntimeError(f"Failed to extract text from {path.name}. Check logs for details.")
        
    cleaned_text = _clean_text(raw_text)
    
    if not cleaned_text:
        logger.warning(f"Document {path.name} was parsed successfully but is empty.")
        
    return cleaned_text