import pytest
from unittest.mock import patch, MagicMock
from text_splitter import split_by_fixed_size, split_by_sentence, split_by_paragraph

"""
Unit tests for text_splitter module, covering fixed-size, sentence-based, and
paragraph-based splitting strategies with mocked NLTK dependencies.
"""

# --- Fixed Size Splitter Tests ---

def test_fixed_size_basic():
    """Checks basic splitting into chunks of exact size with no overlap."""
    text = "abcdefghij"
    chunks = split_by_fixed_size(text, chunk_size=4, overlap=0)
    assert chunks == ["abcd", "efgh", "ij"]

def test_fixed_size_with_overlap():
    """Verifies overlapping chunk generation."""
    text = "abcdefghij"
    chunks = split_by_fixed_size(text, chunk_size=4, overlap=2)
    # Expected: 0-4(abcd), 2-6(cdef), 4-8(efgh), 6-10(ghij)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]

def test_fixed_size_empty():
    """Ensures empty input results in an empty list."""
    assert split_by_fixed_size("") == []

def test_fixed_size_invalid_overlap():
    """Checks that ValueError is raised if overlap >= chunk_size."""
    with pytest.raises(ValueError):
        split_by_fixed_size("text", chunk_size=5, overlap=5)

def test_fixed_size_smaller_than_chunk():
    """Verifies behavior when input text is shorter than the chunk size."""
    text = "abc"
    chunks = split_by_fixed_size(text, chunk_size=5, overlap=0)
    assert chunks == ["abc"]

# --- Sentence Splitter Tests ---

def test_sentence_split_basic():
    """Tests basic sentence splitting logic using a mock tokenizer."""
    text = "Hello world. This is a test."
    with patch("text_splitter._ensure_punkt"), \
         patch("text_splitter.nltk.sent_tokenize") as mock_tokenize:
        
        mock_tokenize.return_value = ["Hello world.", "This is a test."]
        
        chunks = split_by_sentence(text, max_chars=50)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world. This is a test."

def test_sentence_split_max_chars():
    """
    Verifies that sentences are grouped together but split when the cumulative length 
    exceeds max_chars.
    """
    text = "S1. S2. S3."
    with patch("text_splitter._ensure_punkt"), \
         patch("text_splitter.nltk.sent_tokenize") as mock_tokenize:
         
        # Mock behavior: return 3 sentences
        mock_tokenize.return_value = ["S1.", "S2.", "S3."]
        
        # "S1. " (4) + "S2." (3) = 7 chars
        # If we set max_chars=8, S1 and S2 fit (7 <= 8). S3 (3) would make it 10+ > 8.
        # Note: logic in text_splitter joins with " " (1 char).
        
        chunks = split_by_sentence(text, max_chars=8)
        
        # Expected: ["S1. S2.", "S3."]
        assert chunks == ["S1. S2.", "S3."]

def test_sentence_split_empty():
    """Ensures empty string input returns an empty list."""
    assert split_by_sentence("") == []

# --- Paragraph Splitter Tests ---

def test_paragraph_split_basic():
    """Tests basic paragraph splitting by double newlines."""
    text = "Para1.\n\nPara2.\n\nPara3."
    chunks = split_by_paragraph(text, max_chars=100)
    assert len(chunks) == 1 # Should fit in one chunk if max_chars is large
    assert "Para1." in chunks[0]
    assert "Para3." in chunks[0]

def test_paragraph_split_max_chars():
    """Verifies paragraph splitting respects max_chars limit."""
    text = "Para1.\n\nPara2."
    # Force split
    chunks = split_by_paragraph(text, max_chars=10)
    assert len(chunks) == 2
    assert chunks[0] == "Para1."
    assert chunks[1] == "Para2."

def test_paragraph_split_empty():
    """Ensures empty string input results in an empty list."""
    assert split_by_paragraph("") == []
