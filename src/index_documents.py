import os
import logging
from pathlib import Path
from document_loader import load_and_clean_document
from text_splitter import split_by_fixed_size, split_by_sentence, split_by_paragraph
from embedding_client import embedding_client
from database_manager import db_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def _chunk_text(text: str, strategy: str) -> list[str]:
    """Splits text based on the selected strategy."""
    logger.info(f"Splitting text using strategy: {strategy}")
    if strategy == 'fixed':
        return split_by_fixed_size(text)
    elif strategy == 'sentence':
        return split_by_sentence(text)
    elif strategy == 'paragraph':
        return split_by_paragraph(text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _generate_embeddings(chunks: list[str]) -> list[list[float] | None]:
    """Generates embeddings for a list of text chunks."""
    logger.info("Starting embedding generation...")
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        vector = embedding_client.get_embedding(chunk)
        embeddings.append(vector)
        
        if vector:
            if i == 0: 
                logger.info("✅ Sample Check - Chunk #0 embedded successfully.")
                logger.info(f"   Vector dimensions: {len(vector)}")
        else:
            logger.warning(f"⚠️ Failed to embed chunk #{i} (Result: None)")
            
    return embeddings

def _save_to_db(file_name: str, strategy: str, chunks: list[str], embeddings: list[list[float] | None]):
    """Filters valid embeddings and saves them to the database."""
    valid_chunks = []
    valid_embeddings = []
    
    for chunk, vector in zip(chunks, embeddings):
        if vector is not None:
            valid_chunks.append(chunk)
            valid_embeddings.append(vector)

    logger.info(f"Embedding success rate: {len(valid_embeddings)}/{len(chunks)}")

    if valid_embeddings:
        logger.info(f"💾 Saving {len(valid_embeddings)} records to PostgreSQL...")
        try:
            db_manager.insert_chunks(file_name, strategy, valid_chunks, valid_embeddings)
            logger.info("✅ DONE! Document successfully indexed in Database.")
        except Exception as e:
            logger.error(f"❌ Failed to save to database: {e}")
    else:
        logger.warning("⚠️ No valid embeddings generated. Nothing to save to DB.")

def process_document(file_path: str, strategy: str = 'fixed'):
    """Orchestrates the document indexing process."""
    logger.info(f"🚀 Processing file: {file_path}")

    try:
        db_manager.setup_database()
        text = load_and_clean_document(file_path)
    except Exception as e:
        logger.error(f"❌ Process failed at setup/loading: {e}")
        return

    try:
        chunks = _chunk_text(text, strategy)
        logger.info(f"Generated {len(chunks)} chunks.")
        
        embeddings = _generate_embeddings(chunks)
        
        file_name = Path(file_path).name
        _save_to_db(file_name, strategy, chunks, embeddings)
        
        return embeddings # Return for testing/debug
        
    except Exception as e:
        logger.error(f"❌ Process failed during execution: {e}")
        return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index a document into the vector database.")
    parser.add_argument("--file", type=str, required=True, help="Path to the document file (PDF, DOCX).")
    parser.add_argument("--strategy", type=str, default="fixed", choices=["fixed", "sentence", "paragraph"],
                        help="Text splitting strategy (default: fixed).")
    
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        process_document(args.file, strategy=args.strategy)
    else:
        logger.error(f"❌ File not found: {args.file}")