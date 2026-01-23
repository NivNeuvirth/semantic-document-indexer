import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from document_loader import load_and_clean_document

@pytest.fixture
def mock_path_exists():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        yield mock_exists

def test_load_and_clean_pdf(mock_path_exists):
    with patch("document_loader.PdfReader") as MockPdfReader:
        # Setup mock behavior
        mock_reader = MockPdfReader.return_value
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 content."
        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2 content."
        mock_reader.pages = [page1, page2]

        content = load_and_clean_document("dummy.pdf")
        
        assert "Page 1 content." in content
        assert "Page 2 content." in content
        assert "\n" in content # Should join with newlines

def test_load_and_clean_docx(mock_path_exists):
    with patch("document_loader.Document") as MockDocument:
        # Setup mock behavior
        mock_doc = MockDocument.return_value
        p1 = MagicMock()
        p1.text = "Paragraph 1."
        p2 = MagicMock()
        p2.text = "Paragraph 2."
        mock_doc.paragraphs = [p1, p2]

        content = load_and_clean_document("dummy.docx")
        
        assert "Paragraph 1." in content
        assert "Paragraph 2." in content

def test_file_not_found():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            load_and_clean_document("nonexistent.pdf")

def test_unsupported_format(mock_path_exists):
    with pytest.raises(ValueError, match="Unsupported format"):
        load_and_clean_document("image.png")

def test_empty_parsed_content(mock_path_exists):
    # Case where file exists and is valid format, but extraction returns nothing
    with patch("document_loader._extract_from_pdf") as mock_extract:
        mock_extract.return_value = None
        with pytest.raises(RuntimeError, match="Failed to extract text"):
            load_and_clean_document("empty.pdf")
