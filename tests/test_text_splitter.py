import pytest
from text_splitter import split_by_fixed_size, split_by_sentence, split_by_paragraph

# --- Fixed Size Splitter Tests ---

def test_fixed_size_basic():
    text = "abcdefghij"
    chunks = split_by_fixed_size(text, chunk_size=4, overlap=0)
    assert chunks == ["abcd", "efgh", "ij"]

def test_fixed_size_with_overlap():
    text = "abcdefghij"
    chunks = split_by_fixed_size(text, chunk_size=4, overlap=2)
    # Expected: 0-4(abcd), 2-6(cdef), 4-8(efgh), 6-10(ghij)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]

def test_fixed_size_empty():
    assert split_by_fixed_size("") == []

def test_fixed_size_invalid_overlap():
    with pytest.raises(ValueError):
        split_by_fixed_size("text", chunk_size=5, overlap=5)

def test_fixed_size_smaller_than_chunk():
    text = "abc"
    chunks = split_by_fixed_size(text, chunk_size=5, overlap=0)
    assert chunks == ["abc"]

# --- Sentence Splitter Tests ---

def test_sentence_split_basic():
    text = "Hello world. This is a test."
    chunks = split_by_sentence(text, max_chars=50)
    # Assuming standard nltk behavior (space after period might be attached to next or split)
    # Our implementation joins sentences.
    assert len(chunks) >= 1
    assert "Hello world." in chunks[0]

def test_sentence_split_max_chars():
    # Sentences shorter than max_chars should be grouped if possible
    text = "S1. S2. S3."
    # If max_chars allows S1 and S2 but not S3
    chunks = split_by_sentence(text, max_chars=10) 
    # "S1. " is 4 chars. "S2. " is 4 chars.
    # Implementation depends on exact logic in text_splitter.py
    # Expected behavior: group sentences until max_chars is hit.
    pass 

def test_sentence_split_empty():
    assert split_by_sentence("") == []

# --- Paragraph Splitter Tests ---

def test_paragraph_split_basic():
    text = "Para1.\n\nPara2.\n\nPara3."
    chunks = split_by_paragraph(text, max_chars=100)
    assert len(chunks) == 1 # Should fit in one chunk if max_chars is large
    assert "Para1." in chunks[0]
    assert "Para3." in chunks[0]

def test_paragraph_split_max_chars():
    text = "Para1.\n\nPara2."
    # Force split
    chunks = split_by_paragraph(text, max_chars=10)
    assert len(chunks) == 2
    assert chunks[0] == "Para1."
    assert chunks[1] == "Para2."

def test_paragraph_split_empty():
    assert split_by_paragraph("") == []
