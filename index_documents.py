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

def process_document(file_path: str, strategy: str = 'fixed'):
    """
    Orchestrates the document indexing process: loading, splitting, embedding, and saving.

    This function performs the following steps:
    1. Initializes the database.
    2. Loads and cleans the document text.
    3. Splits the text into chunks using the specified strategy.
    4. Generates embeddings for each chunk.
    5. Saves valid chunks and embeddings to the PostgreSQL database.
    6. Writes a debug file ('debug_chunks.txt') with chunking information.

    Args:
        file_path (str): The path to the document to process.
        strategy (str): The text splitting strategy to use ('fixed', 'sentence', 'paragraph').
                        Defaults to 'fixed'.

    Returns:
        list: A list of generated embeddings, or None/empty if process fails early.
    """
    logger.info(f"🚀 Processing file: {file_path}")

    try:
        db_manager.setup_database()
    except Exception as e:
        logger.critical(f"🛑 Database setup failed. Stopping process. Error: {e}")
        return
    
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

    logger.info("Starting embedding generation...")
    
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        vector = embedding_client.get_embedding(chunk)
        embeddings.append(vector)
        
        if vector:
            if i == 0: 
                logger.info("✅ Sample Check - Chunk #0 embedded successfully.")
                logger.info(f"   Vector dimensions: {len(vector)}")
                logger.info(f"   First 5 values: {vector[:5]}")
        else:
            logger.warning(f"⚠️ Failed to embed chunk #{i} (Result: None)")

    valid_chunks = []
    valid_embeddings = []
    
    for chunk, vector in zip(chunks, embeddings):
        if vector is not None:
            valid_chunks.append(chunk)
            valid_embeddings.append(vector)

    logger.info(f"Embedding success rate: {len(valid_embeddings)}/{len(chunks)}")

    if valid_embeddings:
        logger.info(f"💾 Saving {len(valid_embeddings)} records to PostgreSQL (Standard Array Mode)...")
        file_name = Path(file_path).name
        
        try:
            db_manager.insert_chunks(file_name, strategy, valid_chunks, valid_embeddings)
            logger.info("✅ DONE! Document successfully indexed in Database.")
        except Exception as e:
            logger.error(f"❌ Failed to save to database: {e}")
    else:
        logger.warning("⚠️ No valid embeddings generated. Nothing to save to DB.")

    output_filename = "debug_chunks.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"Source: {file_path}\nStrategy: {strategy}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write("="*40 + "\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")
        logger.info(f"Debug output written to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to write debug file: {e}")

    return embeddings

if __name__ == "__main__":
    test_file = "sample.docx" 
    
    if os.path.exists(test_file):
        process_document(test_file, strategy='sentence')
    else:
        logger.warning(f"File {test_file} not found.")