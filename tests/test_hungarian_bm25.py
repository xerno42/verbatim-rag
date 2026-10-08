"""Tests for HungarianBM25Provider with a fake spaCy pipeline (network-free).

TestRealModel runs only where spacy and hu_core_news_lg are installed.
"""

import json
import math
from dataclasses import dataclass

import pytest

# verbatim_rag pulls in numpy, pymilvus etc., which the core-only CI install lacks.
hungarian_bm25 = pytest.importorskip("verbatim_rag.hungarian_bm25")
HungarianBM25Provider = hungarian_bm25.HungarianBM25Provider

STOP = {"a", "az", "és"}


@dataclass
class FakeToken:
    text: str

    @property
    def lemma_(self):
        return self.text

    @property
    def is_alpha(self):
        return self.text.isalpha()

    @property
    def is_stop(self):
        return self.text.lower() in STOP


class FakeNLP:
    """Whitespace tokenizer; lemma == text."""

    meta = {"lang": "hu", "name": "fake", "version": "0"}

    def pipe(self, texts, batch_size=64):
        for text in texts:
            yield [FakeToken(w) for w in text.split()]


CORPUS = [
    "a törvény hatálya",
    "a törvény és a rendelet szerint a törvény",
    "szerződés felmondás munkáltató munkavállaló bér",
]


def make(tmp_path, **kwargs):
    return HungarianBM25Provider(stats_path=str(tmp_path / "stats.json"), nlp=FakeNLP(), **kwargs)


def bm25(query_terms, doc_terms, corpus_terms, k1=1.2, b=0.75):
    n = len(corpus_terms)
    avgdl = sum(map(len, corpus_terms)) / n
    score = 0.0
    for q in set(query_terms):
        df = sum(q in d for d in corpus_terms)
        if not df:
            continue
        tf = doc_terms.count(q)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc_terms) / avgdl))
    return score


def dot(q, d):
    return sum(w * d.get(i, 0.0) for i, w in q.items())


class TestScoring:
    def test_inner_product_equals_bm25(self, tmp_path):
        p = make(tmp_path)
        p.fit(CORPUS)
        corpus_terms = [[w for w in c.split() if w not in STOP] for c in CORPUS]
        docs = p.embed_batch(CORPUS)
        for query in ["törvény hatálya", "rendelet törvény", "munkáltató bér törvény"]:
            q = p.embed_text(query)
            for d_vec, d_terms in zip(docs, corpus_terms):
                assert dot(q, d_vec) == pytest.approx(bm25(query.split(), d_terms, corpus_terms))

    def test_idf_positive_for_term_in_every_doc(self, tmp_path):
        p = make(tmp_path)
        p.fit(["törvény egy", "törvény kettő"])
        assert all(w > 0 for w in p.embed_text("törvény").values())

    def test_unseen_query_returns_empty(self, tmp_path):
        p = make(tmp_path)
        p.fit(CORPUS)
        assert p.embed_text("ismeretlen szavak") == {}
        assert p.embed_text("a az és") == {}

    def test_stop_word_only_chunk_gets_empty_vector(self, tmp_path):
        p = make(tmp_path)
        p.fit(CORPUS)
        assert p.embed_batch(["a az"]) == [{}]

    def test_term_id_is_stable(self):
        assert HungarianBM25Provider._term_id("törvény") == 742698282


class TestFitState:
    def test_embed_requires_fit(self, tmp_path):
        p = make(tmp_path)
        with pytest.raises(RuntimeError, match="call fit"):
            p.embed_text("törvény")
        with pytest.raises(RuntimeError, match="call fit"):
            p.embed_batch(["törvény"])

    def test_fit_rejects_empty_corpus(self, tmp_path):
        with pytest.raises(ValueError):
            make(tmp_path).fit([])

    def test_corpus_without_terms(self, tmp_path):
        p = make(tmp_path)
        p.fit(["a az", "és"])
        with pytest.raises(RuntimeError, match="none had indexable terms"):
            p.embed_text("törvény")

    def test_save_load_round_trip(self, tmp_path):
        p = make(tmp_path)
        p.fit(CORPUS)
        loaded = make(tmp_path)
        assert loaded.n_docs == 3
        assert loaded.embed_text("törvény rendelet") == p.embed_text("törvény rendelet")
        assert loaded.fitted_on(reversed(CORPUS))
        assert not loaded.fitted_on(CORPUS[:2])
        assert not (tmp_path / "stats.json.tmp").exists()

    def test_config_mismatch_blocks_until_refit(self, tmp_path):
        make(tmp_path).fit(CORPUS)
        p = make(tmp_path, k1=1.5)
        with pytest.raises(RuntimeError, match="re-index"):
            p.embed_text("törvény")
        p.fit(CORPUS)
        assert p.embed_text("törvény")

    def test_analyzer_version_mismatch_blocks(self, tmp_path):
        make(tmp_path).fit(CORPUS)
        path = tmp_path / "stats.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["config"]["analyzer_version"] = 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RuntimeError, match="re-index"):
            make(tmp_path).embed_text("törvény")


class TestTermRule:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Törvény", "törvény"),
            ("Ptk.", "ptk"),
            ("6:519.", "6:519"),
            ("1990.", "1990"),
            ("2016/679", "2016/679"),
            ("EU-s", "eu"),
            ("GDPR-t", "gdpr"),
            ("2020-ban", "2020"),
            ("Európa-bajnokság", "európa-bajnokság"),
            ("§-a", None),
            ("§", None),
            ("?", None),
            ("az", None),
        ],
    )
    def test_term(self, text, expected):
        assert HungarianBM25Provider._term(FakeToken(text)) == expected


class TestRealModel:
    @pytest.fixture(scope="class")
    def provider(self, tmp_path_factory):
        pytest.importorskip("spacy")
        pytest.importorskip("hu_core_news_lg")
        return HungarianBM25Provider(stats_path=str(tmp_path_factory.mktemp("bm25") / "s.json"))

    def test_legal_citations_kept(self, provider):
        terms = provider._analyze(["Mit mond a Ptk. 6:519. §-a a kártérítésről?"])[0]
        assert terms == ["mond", "ptk", "6:519", "kártérítés"]

    def test_inflected_forms_match(self, provider):
        doc, query = provider._analyze(
            ["A munkaszerződések felmondása a munkáltatónál.", "munkaszerződés felmondás"]
        )
        assert set(query) <= set(doc)
