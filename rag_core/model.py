from typing import Protocol, List
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code"):
        self._model_name = model_name
        self._embeddings = None

    def _initialize_embeddings(self):
        if self._embeddings is not None:
            return
            
        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embedding model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def model_info(self) -> dict:
        """Return useful metadata about the underlying SentenceTransformer."""
        self._initialize_embeddings()
        st_model = self._embeddings._client
        try:
            dim = st_model.get_sentence_embedding_dimension()
        except AttributeError:
            sample_vec = st_model.encode("test", convert_to_numpy=True)
            dim = int(sample_vec.shape[-1])

        try:
            max_len = st_model.get_max_seq_length()
        except AttributeError:
            max_len = getattr(st_model, "max_seq_length", None)

        try:
            num_params = sum(p.numel() for p in st_model.parameters())
        except Exception:
            num_params = None

        return {
            "model_name": self._model_name,
            "embedding_dim": dim,
            "max_seq_length": max_len,
            "device": str(getattr(st_model, "device", "unknown")),
            "num_parameters": num_params,
            "backend_type": type(st_model).__name__,
        }


if __name__ == "__main__":
    m = Model()
    print(m.model_info())