from typing import List
import logging
from .model import Model

logger = logging.getLogger(__name__)

class Embedding:
    
    def __init__(self, model: Model):
        """
        Args:
            model: An initialized Model instance
        """
        self._model = model

    def embed_text(self, text: str) -> List[float]:
        try:
            self._model._initialize_embeddings()
            st_model = self._model.client  # Use the client property
            vec = st_model.encode(text, convert_to_numpy=True).tolist()
            return vec
        except Exception as e:
            logger.error(f"Embedding text failed: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        try:
            self._model._initialize_embeddings()
            st_model = self._model.client
            vec = st_model.encode(query, convert_to_numpy=True).tolist()
            return vec
        except Exception as e:
            logger.error(f"Embedding query failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            self._model._initialize_embeddings()
            st_model = self._model.client
            return st_model.encode(texts, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise

    @staticmethod
    def convert_bytes_to_text(content: bytes, encoding: str = "utf-8") -> str:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode with {encoding}: {e}")
            raise