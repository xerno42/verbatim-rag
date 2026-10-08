"""
Qwen3-Embedding as the dense provider for Hungarian retrieval.

Qwen3-Embedding is asymmetric: queries are prefixed with an instruction,
documents are embedded as-is. VerbatimIndex calls embed_batch() at ingest and
embed_text() at query time, so the split maps directly onto the interface:

    embed_batch -> documents, no prompt
    embed_text  -> queries, prompt_name="query" (or a custom instruction)

Qwen3-Embedding-0.6B outputs 1024-dim vectors, not the repo default of 384.
Create the store with LocalMilvusStore(dense_dim=provider.get_dimension()) in a
new db file; an existing collection's schema cannot be changed.
"""

import logging
from typing import Any, Dict, List, Optional

from verbatim_rag.embedding_providers import SentenceTransformersProvider

logger = logging.getLogger(__name__)


class Qwen3EmbeddingProvider(SentenceTransformersProvider):
    """Dense provider for Qwen/Qwen3-Embedding-* models via SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
        batch_size: int = 32,
        query_instruction: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        revision: Optional[str] = None,
    ):
        """
        Args:
            model_name: Qwen3-Embedding checkpoint (0.6B for CPU, 4B on GPU).
            device: Torch device.
            batch_size: Encode batch size for embed_batch.
            query_instruction: Task description replacing the model's built-in
                "query" prompt, formatted as "Instruct: {task}\\nQuery:".
                Qwen recommends English instructions for non-English text.
                None uses the model's own "query" prompt.
            model_kwargs: Extra kwargs for the HF model, merged over the
                defaults (e.g. {"torch_dtype": "bfloat16"} on GPU).
            revision: HF hub revision (commit sha) to pin the checkpoint, so eval
                runs are reproducible. None uses the latest.
        """
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.model_kwargs = model_kwargs or {}
        self.revision = revision

        super().__init__(model_name=model_name, device=device)

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            revision=self.revision,
            model_kwargs={"low_cpu_mem_usage": True, "device_map": None, **self.model_kwargs},
            # Qwen3 uses last-token pooling; left padding keeps the last token in place.
            # sentence-transformers 5.7 renamed this to processor_kwargs (old name still
            # accepted); keep tokenizer_kwargs while pyproject allows >=5.0.0.
            tokenizer_kwargs={"padding_side": "left"},
        )
        if self.query_instruction is None and "query" not in self.model.prompts:
            raise ValueError(
                f"{self.model_name} has no 'query' prompt; pass query_instruction explicitly"
            )
        logger.info(f"Loaded Qwen3 embedding model: {self.model_name}")

    def _query_kwargs(self) -> Dict[str, str]:
        if self.query_instruction is not None:
            return {"prompt": f"Instruct: {self.query_instruction}\nQuery:"}
        return {"prompt_name": "query"}

    def embed_text(self, text: str) -> List[float]:
        """Embed a query (instruction-prefixed)."""
        return self.model.encode([text], normalize_embeddings=True, **self._query_kwargs())[
            0
        ].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed documents. prompt="" blocks any default_prompt_name from applying."""
        return self.model.encode(
            texts, prompt="", batch_size=self.batch_size, normalize_embeddings=True
        ).tolist()

    def get_dimension(self) -> int:
        # get_sentence_embedding_dimension is deprecated in newer sentence-transformers.
        getter = getattr(self.model, "get_embedding_dimension", None)
        if getter is None:
            getter = self.model.get_sentence_embedding_dimension
        return getter()
