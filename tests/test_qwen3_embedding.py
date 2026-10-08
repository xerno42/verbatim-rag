"""Tests for Qwen3EmbeddingProvider with a mocked SentenceTransformer (network-free)."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# verbatim_rag pulls in numpy, pymilvus etc., which the core-only CI install lacks.
np = pytest.importorskip("numpy")
qwen3_embedding = pytest.importorskip("verbatim_rag.qwen3_embedding")
Qwen3EmbeddingProvider = qwen3_embedding.Qwen3EmbeddingProvider


def _fake_model(prompts=None, dim=1024):
    model = MagicMock()
    model.prompts = {"query": "Instruct: q\nQuery:", "document": ""} if prompts is None else prompts
    model.get_embedding_dimension.return_value = dim
    model.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), dim))
    return model


@pytest.fixture
def fake_st():
    model = _fake_model()
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = MagicMock(return_value=model)
    with patch.dict(sys.modules, {"sentence_transformers": module}):
        yield module.SentenceTransformer, model


class TestQwen3EmbeddingProvider:
    def test_loads_with_left_padding(self, fake_st):
        cls, _ = fake_st
        Qwen3EmbeddingProvider(device="cpu")
        args, kwargs = cls.call_args
        assert args[0] == "Qwen/Qwen3-Embedding-0.6B"
        assert kwargs["tokenizer_kwargs"] == {"padding_side": "left"}
        assert kwargs["revision"] is None

    def test_revision_passed_through(self, fake_st):
        cls, _ = fake_st
        Qwen3EmbeddingProvider(revision="abc123")
        assert cls.call_args.kwargs["revision"] == "abc123"

    def test_query_uses_query_prompt(self, fake_st):
        _, model = fake_st
        vec = Qwen3EmbeddingProvider().embed_text("Mi a munkaszerződés?")
        assert len(vec) == 1024
        kwargs = model.encode.call_args.kwargs
        assert kwargs["prompt_name"] == "query"
        assert "prompt" not in kwargs

    def test_documents_get_no_prompt(self, fake_st):
        _, model = fake_st
        vecs = Qwen3EmbeddingProvider(batch_size=4).embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        kwargs = model.encode.call_args.kwargs
        assert kwargs["prompt"] == ""
        assert "prompt_name" not in kwargs
        assert kwargs["batch_size"] == 4

    def test_custom_query_instruction(self, fake_st):
        _, model = fake_st
        provider = Qwen3EmbeddingProvider(query_instruction="Retrieve Hungarian legal passages")
        provider.embed_text("kérdés")
        kwargs = model.encode.call_args.kwargs
        assert kwargs["prompt"] == "Instruct: Retrieve Hungarian legal passages\nQuery:"
        assert "prompt_name" not in kwargs

    def test_missing_query_prompt_raises(self, fake_st):
        cls, _ = fake_st
        cls.return_value = _fake_model(prompts={})
        with pytest.raises(ValueError, match="query"):
            Qwen3EmbeddingProvider()

    def test_dimension(self, fake_st):
        assert Qwen3EmbeddingProvider().get_dimension() == 1024

    def test_dimension_fallback_for_older_sentence_transformers(self, fake_st):
        _, model = fake_st
        del model.get_embedding_dimension
        model.get_sentence_embedding_dimension.return_value = 1024
        assert Qwen3EmbeddingProvider().get_dimension() == 1024
