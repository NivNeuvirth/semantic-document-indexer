import os
import logging
from document_loader import load_and_clean_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    test_file = "sample.docx" 
    
    if not os.path.exists(test_file):
        logger.warning(f"Please place a file named '{test_file}' to test.")
        return

    try:
        logger.info(f"Processing {test_file}...")
        text = load_and_clean_document(test_file)
        
        logger.info(f"Successfully extracted {len(text)} characters.")
        print("-" * 20)
        print(text[:500])
        print("-" * 20)
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()