"""Metric helpers of the Hungarian retrieval eval runner (network-free)."""

import pytest

from evaluation.hu_legal.run_retrieval import QueryCache, breakdown, first_hit_rank, summarize


class TestFirstHitRank:
    def test_first_gold_wins(self):
        assert first_hit_rank(["a", "g2", "g1"], ["g1", "g2"]) == 2

    def test_miss(self):
        assert first_hit_rank(["a", "b"], ["g"]) is None

    def test_empty_results(self):
        assert first_hit_rank([], ["g"]) is None


class TestSummarize:
    def test_recall_and_mrr(self):
        s = summarize([1, 4, 50, None])
        assert s["n"] == 4
        assert s["recall@10"] == pytest.approx(0.5)
        assert s["recall@100"] == pytest.approx(0.75)
        assert s["mrr@100"] == pytest.approx((1 + 1 / 4 + 1 / 50) / 4)

    def test_rank_beyond_top_k_is_a_miss(self):
        assert summarize([101])["recall@100"] == 0

    def test_empty(self):
        assert summarize([]) == {"n": 0}


def test_breakdown_groups_by_field():
    rows = [
        {"difficulty": "easy", "ranks": {"dense": 1}},
        {"difficulty": "hard", "ranks": {"dense": None}},
        {"difficulty": "easy", "ranks": {"dense": 20}},
    ]
    out = breakdown(rows, "dense", "difficulty")
    assert out["easy"]["n"] == 2 and out["easy"]["recall@10"] == 0.5
    assert out["hard"]["recall@100"] == 0


def test_query_cache_embeds_once_and_counts_empty_sparse():
    calls = []

    class Provider:
        def embed_text(self, text):
            calls.append(text)
            return {}

    cache = QueryCache(Provider())
    cache.embed_text("q")
    cache.embed_text("q")
    assert calls == ["q"]
    assert cache.empty == 1
