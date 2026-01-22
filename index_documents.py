import os
import logging
from document_loader import load_and_clean_document
from text_splitter import split_by_fixed_size, split_by_sentence, split_by_paragraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def process_document(file_path: str, strategy: str = 'fixed'):
    logger.info(f"Processing file: {file_path}")
    
    try:
        text = load_and_clean_document(file_path)
    except Exception as e:
        logger.error(f"Failed to load document: {e}")
        return

    logger.info(f"Splitting text using strategy: {strategy}")
    chunks = []
    
    if strategy == 'fixed':
        chunks = split_by_fixed_size(text)
    elif strategy == 'sentence':
        chunks = split_by_sentence(text)
    elif strategy == 'paragraph':
        chunks = split_by_paragraph(text)
    else:
        logger.error(f"Unknown strategy: {strategy}")
        return

    logger.info(f"Generated {len(chunks)} chunks.")
    
    output_filename = "debug_chunks.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"Source File: {file_path}\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write("=" * 40 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk)
                f.write("\n\n")
        
        logger.info(f"Chunks written to file: {output_filename}")
        
    except Exception as e:
        logger.error(f"Failed to write debug file: {e}")

if __name__ == "__main__":
    test_file = "sample.docx" 
    
    if os.path.exists(test_file):
        process_document(test_file, strategy='sentence')
    else:
        logger.warning(f"File {test_file} not found.")