"""VerbatimIndex.query with an empty sparse query vector (mocked store, network-free)."""

from unittest.mock import MagicMock

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
index_module = pytest.importorskip("verbatim_rag.index")
VerbatimIndex = index_module.VerbatimIndex

DENSE = [0.1, 0.2]


def make_index(sparse_vec, with_dense=True):
    store = MagicMock()
    store.query.return_value = ["hit"]
    dense = MagicMock()
    dense.embed_text.return_value = DENSE
    sparse = MagicMock()
    sparse.embed_text.return_value = sparse_vec
    index = VerbatimIndex(
        vector_store=store,
        dense_provider=dense if with_dense else None,
        sparse_provider=sparse,
        chunker_provider=MagicMock(),
    )
    return index, store


class TestEmptySparseQuery:
    def test_auto_hybrid_falls_back_to_dense(self):
        index, store = make_index({})
        assert index.query("Mi ez?") == ["hit"]
        kwargs = store.query.call_args.kwargs
        assert kwargs["search_type"] == "dense"
        assert kwargs["dense_query"] == DENSE
        assert kwargs["sparse_query"] is None

    def test_hybrid_weights_drop_sparse(self):
        index, store = make_index({})
        index.query("Mi ez?", hybrid_weights={"dense": 0.5, "sparse": 0.5})
        kwargs = store.query.call_args.kwargs
        assert kwargs["hybrid_weights"] == {"dense": 0.5}
        assert kwargs["sparse_query"] is None

    def test_sparse_only_returns_no_results(self):
        index, store = make_index({}, with_dense=False)
        assert index.query("Mi ez?") == []
        assert index.query("Mi ez?", hybrid_weights={"sparse": 1.0}) == []
        store.query.assert_not_called()

    def test_non_empty_sparse_unchanged(self):
        index, store = make_index({7: 1.5})
        index.query("törvény")
        kwargs = store.query.call_args.kwargs
        assert kwargs["search_type"] == "hybrid"
        assert kwargs["sparse_query"] == {7: 1.5}

        index.query("törvény", hybrid_weights={"dense": 0.5, "sparse": 0.5})
        kwargs = store.query.call_args.kwargs
        assert kwargs["hybrid_weights"] == {"dense": 0.5, "sparse": 0.5}
        assert kwargs["sparse_query"] == {7: 1.5}
