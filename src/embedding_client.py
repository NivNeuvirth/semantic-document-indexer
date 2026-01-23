import os
import time
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingClient:
    """
    Client for interacting with Google's GenAI embedding models.

    This client handles the initialization of the Google GenAI service and provides
    methods to generate text embeddings safely with retry logic.

    Attributes:
        model_name (str): The name of the embedding model to use (default: "text-embedding-004").
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.error("API Key not found in environment variables")
            raise ValueError("API Key not set. Please set GEMINI_API_KEY in .env")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = "text-embedding-004"

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding vector for the given text using the Google GenAI SDK.

        Args:
            text (str): The input text to be embedded.

        Returns:
            Optional[List[float]]: A list of floating-point numbers representing the embedding,
            or None if the text is empty or generation fails after retries.
        """
        if not text:
            return None

        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                
                if response.embeddings:
                    return response.embeddings[0].values
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                time.sleep(1)
                
        logger.error("Failed to generate embedding.")
        return None
    
embedding_client = EmbeddingClient()