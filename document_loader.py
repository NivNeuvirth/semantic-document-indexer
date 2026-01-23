import logging
import re
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

def _extract_from_pdf(file_path: Path) -> Optional[str]:
    """
    Extracts text content from a PDF file.

    Args:
        file_path (Path): The pathlib.Path object pointing to the PDF file.

    Returns:
        Optional[str]: The extracted text combined into a single string, or None if extraction fails.
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
    Extracts text content from a DOCX file.

    Args:
        file_path (Path): The pathlib.Path object pointing to the DOCX file.

    Returns:
        Optional[str]: The extracted paragraphs joined by newlines, or None if extraction fails.
    """
    try:
        doc = Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        logger.exception(f"Failed to parse DOCX file: {file_path.name}")
        return None

def _clean_text(text: str) -> str:
    """
    Normalizes whitespace in the given text.

    Replaces multiple whitespace characters (including tabs) with a single space
    and strips leading/trailing whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned and normalized text.
    """
    if not text:
        return ""
    return re.sub(r'[ \t]+', ' ', text).strip()

def load_and_clean_document(file_path_str: str) -> str:
    """
    Loads a document from the filesystem and returns its cleaned text content.

    This function supports PDF and DOCX formats. It handles file reading,
    text extraction, and basic whitespace normalization.

    Args:
        file_path_str (str): The absolute or relative path to the document file.

    Returns:
        str: The extracted and cleaned text content of the document.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format (extension) is not supported.
        RuntimeError: If text extraction fails (e.g., empty result from parser).
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