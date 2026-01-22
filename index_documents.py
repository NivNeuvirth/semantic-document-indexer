import os
from document_loader import load_and_clean_document

def main():
    test_file = "sample.docx" 
    
    if not os.path.exists(test_file):
        print(f"Please place a file named '{test_file}' to test.")
        return

    try:
        print(f"Processing {test_file}...")
        text = load_and_clean_document(test_file)
        
        print(f"Successfully extracted {len(text)} characters.")
        print("-" * 20)
        print(text[:500])
        print("-" * 20)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()