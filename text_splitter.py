import logging
import nltk
import re
from typing import List

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

logger = logging.getLogger(__name__)


def _ensure_punkt():
    """
    Ensures that the NLTK 'punkt' tokenizer data is downloaded.

    Checks if 'tokenizers/punkt' is available; if not, attempts to download it.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            logger.exception("Failed to download NLTK punkt tokenizer.")
            raise

def split_by_fixed_size(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """
    Splits text into fixed-size character chunks with specified overlap.

    Args:
        text (str): The input text to split.
        chunk_size (int): The maximum size of each chunk in characters.
        overlap (int): The number of characters of overlap between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.

    Raises:
        ValueError: If `overlap` is greater than or equal to `chunk_size`.
    """
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    return chunks

def split_by_sentence(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Splits text by sentences using NLTK, ensuring chunks don't exceed a maximum character length.

    This strategy tries to keep sentences together. If a single sentence exceeds `max_chars`,
    it may still be included (depending on implementation specifics) or force a split.
    Fallbacks to fixed-size splitting if NLTK fails.

    Args:
        text (str): The input text to split.
        max_chars (int): The target maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []

    try:
        _ensure_punkt()
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        logger.exception(f"NLTK tokenization failed: {e}. Falling back to simple split.")
        return split_by_fixed_size(text, max_chars, 0)

    chunks = []
    current_chunk = []
    current_len = 0
    sep_len = 1 
    for sentence in sentences:
        sentence_len = len(sentence)
        extra = sep_len if current_chunk else 0
        if current_len + sentence_len + extra > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len + extra
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def split_by_paragraph(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Splits text by double newlines (paragraphs), ensuring chunks don't exceed a maximum character length.

    Args:
        text (str): The input text to split.
        max_chars (int): The target maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []
    
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0
    sep_len = 1 
    for para in paragraphs:
        para_len = len(para)
        extra = sep_len if current_chunk else 0
        if current_len + para_len + extra > max_chars:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len + extra
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks