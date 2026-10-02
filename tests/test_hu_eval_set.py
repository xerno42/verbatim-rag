import random

import pytest

from evaluation.hu_legal.build_eval_set import assign_splits, pick_holdout
from evaluation.hu_legal.overlap import equivalent_chunks, overlap_clusters, shared_shingle_counts
from evaluation.hu_legal.spans import (
    SpanError,
    chunks_containing,
    resolve_annotation,
    resolve_quote,
)

PASSAGE = "1. cikk\nA tagállamok biztosítják,  hogy a díj\nmegfizethető legyen. A díj nem emelhető."


class TestResolveQuote:
    def test_offsets_point_into_original_passage(self):
        span = resolve_quote(PASSAGE, "A tagállamok biztosítják")
        assert PASSAGE[span["start"] : span["end"]] == span["text"] == "A tagállamok biztosítják"

    def test_whitespace_differences_are_ignored(self):
        span = resolve_quote(PASSAGE, "biztosítják, hogy a díj megfizethető legyen.")
        assert span["text"] == "biztosítják,  hogy a díj\nmegfizethető legyen."

    def test_missing_quote_raises(self):
        with pytest.raises(SpanError, match="not found"):
            resolve_quote(PASSAGE, "nem szerepel")

    def test_ambiguous_quote_needs_occurrence(self):
        with pytest.raises(SpanError, match="occurs 2 times"):
            resolve_quote(PASSAGE, "díj")
        second = resolve_quote(PASSAGE, {"text": "díj", "occurrence": 2})
        assert second["start"] == PASSAGE.rindex("díj")


class TestResolveAnnotation:
    def record(self, **kw):
        return {"qid": "q1", "passage": PASSAGE, **kw}

    def test_spans_sorted_by_start(self):
        out = resolve_annotation(
            self.record(answerable="yes", quotes=["A díj nem emelhető.", "1. cikk"])
        )
        assert [s["text"] for s in out["gold_spans"]] == ["1. cikk", "A díj nem emelhető."]

    @pytest.mark.parametrize(
        "kw",
        [
            {"answerable": None, "quotes": []},
            {"answerable": "no", "quotes": ["1. cikk"]},
            {"answerable": "yes", "quotes": []},
        ],
    )
    def test_inconsistent_records_raise(self, kw):
        with pytest.raises(SpanError):
            resolve_annotation(self.record(**kw))


def test_chunks_containing_requires_every_span():
    corpus = [
        {"chunk_id": "a", "text": "A díj nem emelhető. Más szöveg."},
        {"chunk_id": "b", "text": "1. cikk A díj  nem\nemelhető."},
    ]
    spans = [{"text": "1. cikk"}, {"text": "A díj nem emelhető."}]
    assert chunks_containing(spans, corpus) == ["b"]


WORDS = [f"w{i}" for i in range(60)]


def test_overlap_equivalents_and_clusters():
    texts = [
        " ".join(WORDS[0:20]),  # 0: contained in 1
        " ".join(WORDS[0:40]),  # 1
        " ".join(WORDS[35:60]),  # 2: shares a run shorter than one shingle with 1
        "teljesen más szöveg ami semmivel sem fedi át a többit itt",  # 3
    ]
    sets, pairs = shared_shingle_counts(texts)
    eq = equivalent_chunks(sets, pairs, 0.9)
    assert eq[0] == {1}
    assert 0 not in eq.get(1, set())

    assert overlap_clusters(len(texts), pairs, min_shared=1) == [[0, 1], [2], [3]]


def test_holdout_takes_whole_clusters_and_respects_size_cap():
    clusters = [[0, 1], [2], [3], [4, 5, 6, 7]]
    held = pick_holdout(clusters, n_rows=8, frac=0.5, max_cluster=2, rng=random.Random(0))
    assert not held & {4, 5, 6, 7}
    for c in clusters:
        assert set(c) <= held or not set(c) & held


def test_splits_are_stratified():
    rows = [
        {"answerable": a, "source": "s", "bucket": "A", "difficulty": "easy", "query_type": "t"}
        for a in [True] * 50 + [False] * 10
    ]
    assign_splits(rows, test_frac=0.1, dev_frac=0.2, rng=random.Random(0))
    for answerable, n in [(True, 50), (False, 10)]:
        group = [r["split"] for r in rows if r["answerable"] is answerable]
        assert group.count("test") == round(n * 0.1)
        assert group.count("dev") == round(n * 0.2)
