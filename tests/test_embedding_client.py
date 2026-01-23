import pytest
from unittest.mock import MagicMock, patch
import os
from embedding_client import EmbeddingClient

"""
Unit tests for the EmbeddingClient, verifying initialization, API key handling,
embedding generation, and retry logic with mocked Google GenAI calls.
"""

@pytest.fixture
def mock_env_api_key():
    """Sets a mock GEMINI_API_KEY in the environment for testing."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
        yield

def test_init_raises_without_api_key():
    """Ensures ValueError is raised when the API key is missing from environment variables."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API Key not set"):
            EmbeddingClient()

@pytest.mark.usefixtures("mock_env_api_key")
def test_init_success():
    """Verifies that the client initializes correctly with the default model name."""
    with patch("embedding_client.genai.Client"): # Note: file imports genai
        client = EmbeddingClient()
        assert client.model_name == "text-embedding-004"

@pytest.mark.usefixtures("mock_env_api_key")
def test_get_embedding_success():
    """Tests successful embedding generation using a mocked GenAI client."""
    with patch("embedding_client.genai.Client") as MockGenAI:
        # Setup mock
        mock_client_instance = MockGenAI.return_value
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
        mock_client_instance.models.embed_content.return_value = mock_response

        client = EmbeddingClient()
        embedding = client.get_embedding("test text")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_client_instance.models.embed_content.assert_called_once()

@pytest.mark.usefixtures("mock_env_api_key")
def test_get_embedding_empty_text():
    """Ensures get_embedding returns None immediately for empty input strings."""
    with patch("embedding_client.genai.Client"):
        client = EmbeddingClient()
        assert client.get_embedding("") is None

@pytest.mark.usefixtures("mock_env_api_key")
def test_get_embedding_retry_failure():
    """Verifies that None is returned after retries when the API fails repeatedly."""
    with patch("embedding_client.genai.Client") as MockGenAI:
        mock_client_instance = MockGenAI.return_value
        # Make it raise exception every time
        mock_client_instance.models.embed_content.side_effect = Exception("API Error")

        client = EmbeddingClient()
        
        # Should return None after retries
        result = client.get_embedding("text")
        assert result is None
        assert mock_client_instance.models.embed_content.call_count == 3
