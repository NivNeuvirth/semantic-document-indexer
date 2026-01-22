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
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.error("API Key not found in environment variables")
            raise ValueError("API Key not set. Please set GEMINI_API_KEY in .env")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = "text-embedding-004"

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates embedding using the new google-genai SDK.
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