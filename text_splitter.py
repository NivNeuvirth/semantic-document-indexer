import logging
import nltk
from typing import List

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def split_by_fixed_size(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """
    Splits text into fixed-size chunks with overlap.
    """
    if not text:
        return []
        
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
    Splits text by sentences using NLTK, ensuring chunks don't exceed max_chars.
    """
    if not text:
        return []

    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        logger.error(f"NLTK tokenization failed: {e}. Falling back to simple split.")
        return split_by_fixed_size(text, max_chars, 0)

    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        if current_len + sentence_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def split_by_paragraph(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Splits text by double newlines (paragraphs).
    """
    if not text:
        return []
        
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        para_len = len(para)
        
        if current_len + para_len > max_chars:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len + 1
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    return chunks