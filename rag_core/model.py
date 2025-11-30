from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


class Model:

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code", device: str = "cpu"):
        self._model_name: str = model_name
        self._target_device: str = device
        self._hf = None
        self.dim: int = None
        self.max_len: int = None
        self.params: int = None
        self.device: str = None
        self._inspected = False 

    @property
    def name(self):
        return self._model_name

    def _load(self):
        if self._hf is not None:
            return

        try:
            self._hf = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"trust_remote_code": True, "device": self._target_device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"Loaded embedding model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self._model_name}: {e}")
            raise

    def _initialize_embeddings(self):
        self._load()
    
    @property
    def client(self):
        self._load()
        return getattr(self._hf, "client", None) or self._hf._client
    
    def _inspect(self):
        if self._inspected:
            return

        self._load()
        backend = self.client

        try:
            self.dim = backend.get_sentence_embedding_dimension()
        except AttributeError:
            self.dim = int(backend.encode("test", convert_to_numpy=True).shape[-1])

        try:
            self.max_len = backend.get_max_seq_length()
        except AttributeError:
            self.max_len = getattr(backend, "max_seq_length", None)

        try:
            self.params = sum(p.numel() for p in backend.parameters())
        except Exception:
            self.params = None

        self.device = str(getattr(backend, "device", "unknown"))
        self._inspected = True

    def info(self) -> dict:
        self._inspect()
        return {
            "name": self._model_name,
            "dim": self.dim,
            "max_len": self.max_len,
            "params": self.params,
            "device": self.device,
        }
