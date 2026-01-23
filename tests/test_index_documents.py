import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.index_documents import process_document

"""
Integration tests for the process_document workflow, mocking all external
dependencies (DB, loading, embedding) to verify orchestration logic.
"""

@pytest.fixture
def mock_dependencies():
    """Patches all external dependencies for process_document and yields them."""
    with patch("src.index_documents.db_manager") as mock_db, \
         patch("src.index_documents.load_and_clean_document") as mock_load, \
         patch("src.index_documents.embedding_client") as mock_embed, \
         patch("src.index_documents.split_by_fixed_size") as mock_split:
         
        yield mock_db, mock_load, mock_embed, mock_split

def test_process_document_success(mock_dependencies):
    """Verifies the complete success flow: setup, load, split, embed, and insert."""
    mock_db, mock_load, mock_embed, mock_split = mock_dependencies
    
    # Setup mocks
    mock_load.return_value = "chunk1 chunk2"
    mock_split.return_value = ["chunk1", "chunk2"]
    mock_embed.get_embedding.side_effect = [[0.1], [0.2]] # Returns vector for each chunk
    
    with patch("builtins.open", mock_open()):
        embeddings = process_document("test.pdf", strategy="fixed")
        
        # Verify DB setup
        mock_db.setup_database.assert_called_once()
        
        # Verify splitting
        mock_split.assert_called_once()
        
        # Verify embedding
        assert mock_embed.get_embedding.call_count == 2
        
        # Verify DB insert
        mock_db.insert_chunks.assert_called_once()
        args = mock_db.insert_chunks.call_args[0]
        assert args[0] == "test.pdf"
        assert args[2] == ["chunk1", "chunk2"]
        assert args[3] == [[0.1], [0.2]]
        
        # Verify result
        assert len(embeddings) == 2

def test_process_document_db_setup_failure(mock_dependencies):
    """Checks that the process aborts immediately if database setup fails."""
    mock_db, _, _, _ = mock_dependencies
    mock_db.setup_database.side_effect = Exception("DB Fail")
    
    result = process_document("test.pdf")
    assert result is None
    # Should stop early
    mock_db.insert_chunks.assert_not_called()

def test_process_document_load_failure(mock_dependencies):
    """Checks that the process aborts if document loading fails."""
    mock_db, mock_load, _, _ = mock_dependencies
    mock_load.side_effect = Exception("Load Fail")
    
    result = process_document("test.pdf")
    assert result is None
    mock_db.insert_chunks.assert_not_called()

def test_process_document_invalid_strategy(mock_dependencies):
    """Checks that the process aborts if an unknown splitting strategy is provided."""
    mock_db, mock_load, _, _ = mock_dependencies
    mock_load.return_value = "text"
    
    result = process_document("test.pdf", strategy="unknown")
    assert result is None
    mock_db.insert_chunks.assert_not_called()

def test_process_document_no_valid_embeddings(mock_dependencies):
    """Ensures nothing is inserted into the DB if all embedding attempts fail."""
    mock_db, mock_load, mock_embed, mock_split = mock_dependencies
    
    mock_load.return_value = "text"
    mock_split.return_value = ["chunk1"]
    mock_embed.get_embedding.return_value = None # Failure
    
    with patch("builtins.open", mock_open()):
        process_document("test.pdf")
        
        # Should not insert anything
        mock_db.insert_chunks.assert_not_called()
