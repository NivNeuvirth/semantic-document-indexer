import pytest
import psycopg2
from unittest.mock import MagicMock, patch, call
from database_manager import DatabaseManager
import os

"""
Unit tests for the DatabaseManager class, covering connection, setup,
deletion, and insertion logic using mocked PostgreSQL interactions.
"""

@pytest.fixture
def mock_db_env():
    """Sets a mock POSTGRES_URL environment variable for testing."""
    with patch.dict(os.environ, {"POSTGRES_URL": "postgresql://user:pass@localhost/db"}):
        yield

def test_init_raises_without_url():
    """Verifies that ValueError is raised if POSTGRES_URL is not set."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Missing Configuration"):
            DatabaseManager()

@pytest.mark.usefixtures("mock_db_env")
def test_get_connection():
    """Checks that get_connection calls psycopg2.connect with the correct URL."""
    with patch("database_manager.psycopg2.connect") as mock_connect:
        db = DatabaseManager()
        db.get_connection()
        mock_connect.assert_called_once_with("postgresql://user:pass@localhost/db")

@pytest.mark.usefixtures("mock_db_env")
def test_setup_database():
    """Ensures setup_database executes the expected table creation SQL."""
    with patch("database_manager.psycopg2.connect") as mock_connect:
        mock_conn = mock_connect.return_value
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        
        db = DatabaseManager()
        db.setup_database()
        
        # Verify table creation query executed
        assert mock_cur.execute.call_count >= 1
        # First call should be table creation
        args, _ = mock_cur.execute.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS document_chunks" in args[0]
        
        mock_conn.commit.assert_called_once()

@pytest.mark.usefixtures("mock_db_env")
def test_delete_existing_chunks():
    """Verifies that delete_existing_chunks executes the correct DELETE SQL command."""
    with patch("database_manager.psycopg2.connect") as mock_connect:
        mock_conn = mock_connect.return_value
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        mock_cur.rowcount = 5
        
        db = DatabaseManager()
        db.delete_existing_chunks("file.txt", "fixed")
        
        mock_cur.execute.assert_called_once()
        args, _ = mock_cur.execute.call_args
        assert "DELETE FROM document_chunks" in args[0]
        assert args[1] == ("file.txt", "fixed")
        mock_conn.commit.assert_called_once()

@pytest.mark.usefixtures("mock_db_env")
def test_insert_chunks():
    """Tests that insert_chunks performs a delete followed by a bulk insert."""
    with patch("database_manager.psycopg2.connect") as mock_connect:
        mock_conn = mock_connect.return_value
        
        # We need to mock execute_values since it's imported from psycopg2.extras
        with patch("database_manager.execute_values") as mock_execute_values:
            db = DatabaseManager()
            chunks = ["chunk1", "chunk2"]
            embeddings = [[0.1], [0.2]]
            
            db.insert_chunks("file.txt", "fixed", chunks, embeddings)
            
            # Should delete first then insert
            # Actually implementation opens cursor once.
            mock_cur = mock_conn.cursor.return_value.__enter__.return_value
            
            # Check delete called
            mock_cur.execute.assert_called() 
            del_args = mock_cur.execute.call_args[0]
            assert "DELETE FROM" in del_args[0]
            
            # Check insert called via execute_values
            mock_execute_values.assert_called_once()
            args, _ = mock_execute_values.call_args
            assert args[0] == mock_cur # cursor
            assert "INSERT INTO" in args[1] # query
            assert len(args[2]) == 2 # data list length
            assert args[2][0] == ("file.txt", "fixed", "chunk1", [0.1])

            mock_conn.commit.assert_called_once()
