from typing import List
import logging
from .model import Model
logger = logging.getLogger(__name__)

class EmbeddingService(Model):
    
    def __init__(self, model: Model):
        self._model = model

    def embed_text(self, text: str) -> List[float]:
        try:
            self._model._initialize_embeddings()

            st_model = self._model._embeddings.client

            vec = st_model.encode(text, convert_to_numpy=True).tolist()

            return vec
        except Exception as e:
            logger.error(f"Embedding text failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            self._model._initialize_embeddings()
            st_model = self._model._embeddings.client
            return st_model.encode(texts, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise

    @staticmethod
    def convert_bytes_to_text(content: bytes, encoding: str = "utf-8") -> List[str]:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode with {encoding}")
            