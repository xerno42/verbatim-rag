"""
Build the Hungarian eval set from the JudicialMind legal training dataset.

    python -m evaluation.hu_legal.build_eval_set [--revision SHA] [--out DIR]

Reads the six Hungarian parquet files (eurlex_hu, lemur_hu x buckets A/B/C)
and writes, under evaluation/hu_legal/data/:

    corpus.jsonl               chunks to index, one per unique passage, keyed by chunk_id
    queries.jsonl              every query with split, gold chunks and breakdown fields
    annotation_template.jsonl  queries picked for hand-annotated answer spans
    manifest.json              dataset revision, parameters and counts

The build is deterministic for a given revision and seed. If
annotations/gold_spans.jsonl exists, its spans are merged into queries.jsonl.

Two facts about the dataset shape this build (see README.md):
- Buckets A/B/C are passage-length buckets (<=256, <=512, <=1024 tokens), not
  train/test splits, so splits are drawn here, stratified over all fields.
- Passages overlap heavily, so gold chunks are expanded to chunks that contain
  the gold passage, and unanswerable questions hold out whole overlap clusters.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from evaluation.hu_legal.overlap import equivalent_chunks, overlap_clusters, shared_shingle_counts
from evaluation.hu_legal.spans import SpanError, chunks_containing, resolve_annotation

REPO_ID = "judicialmind/legal-training-dataset"
SOURCES = ["eurlex_hu", "lemur_hu"]
BUCKETS = ["A", "B", "C"]

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "data"
ANNOTATIONS = HERE / "annotations" / "gold_spans.jsonl"
OUT_OF_SCOPE = HERE / "out_of_scope.jsonl"

STRATA = ["answerable", "source", "bucket", "difficulty", "query_type"]
BREAKDOWNS = ["query_type", "legal_domain", "difficulty", "bucket", "source"]


def load_dataset(revision: str) -> pd.DataFrame:
    frames = []
    for source in SOURCES:
        for bucket in BUCKETS:
            path = hf_hub_download(
                REPO_ID,
                f"hungarian/eu/{source}_{bucket}.parquet",
                repo_type="dataset",
                revision=revision,
            )
            frames.append(pd.read_parquet(path).assign(source=source))
    df = pd.concat(frames, ignore_index=True)

    for col in ["chunk_id", "positive", "query"]:
        dupes = df[col].duplicated().sum()
        if dupes:
            raise ValueError(
                f"{dupes} duplicate values in '{col}'; the build assumes one query per passage"
            )
    return df.sort_values("chunk_id", ignore_index=True)


def pick_holdout(
    clusters: List[List[int]], n_rows: int, frac: float, max_cluster: int, rng: random.Random
) -> set:
    """Choose whole overlap clusters to drop from the index until `frac` of queries lose their evidence."""
    eligible = [c for c in clusters if len(c) <= max_cluster]
    rng.shuffle(eligible)
    held: set = set()
    target = round(frac * n_rows)
    for cluster in eligible:
        if len(held) >= target:
            break
        held.update(cluster)
    return held


def assign_splits(rows: List[Dict], test_frac: float, dev_frac: float, rng: random.Random) -> None:
    by_stratum: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        by_stratum[tuple(r[k] for k in STRATA)].append(r)
    for key in sorted(by_stratum):
        members = by_stratum[key]
        rng.shuffle(members)
        n_test = round(len(members) * test_frac)
        n_dev = round(len(members) * dev_frac)
        for i, r in enumerate(members):
            r["split"] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"


def pick_for_annotation(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """Round-robin over query_type x difficulty cells so rare cells are still covered."""
    cells: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == "test" and r["answerable"]:
            cells[(r["query_type"], r["difficulty"])].append(r)
    for key in cells:
        rng.shuffle(cells[key])
    picked: List[Dict] = []
    keys = sorted(cells)
    while len(picked) < n and any(cells.values()):
        for key in keys:
            if cells[key] and len(picked) < n:
                picked.append(cells[key].pop())
    return sorted(picked, key=lambda r: r["qid"])


def merge_annotations(rows: List[Dict], corpus: List[Dict], path: Path) -> Dict[str, int]:
    by_qid = {r["qid"]: r for r in rows}
    counts: Counter = Counter()
    errors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("answerable") is None:
            counts["pending"] += 1
            continue
        row = by_qid.get(record["qid"])
        if row is None:
            errors.append(f"{record['qid']}: unknown qid")
            continue
        record = {**record, "passage": row["passage"]}
        try:
            resolved = resolve_annotation(record)
        except SpanError as e:
            errors.append(str(e))
            continue

        row["annotation"] = resolved["answerable"]
        row["annotation_notes"] = record.get("notes", "")
        if resolved["answerable"] == "no":
            # The synthetic query is not answered by its own passage; its true
            # gold is unknown, so it is excluded rather than scored.
            row["excluded"] = True
        else:
            row["gold_spans"] = resolved["gold_spans"]
            row["gold_chunk_ids"] = chunks_containing(resolved["gold_spans"], corpus)
        counts[resolved["answerable"]] += 1

    if errors:
        raise SystemExit("Annotation errors:\n  " + "\n  ".join(errors))
    return dict(counts)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--revision",
        default=None,
        help="dataset commit sha (default: current main, recorded in the manifest)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--dev-frac", type=float, default=0.10)
    p.add_argument("--unanswerable-frac", type=float, default=0.12)
    p.add_argument("--n-annotate", type=int, default=100)
    p.add_argument("--equiv-containment", type=float, default=0.9)
    p.add_argument("--cluster-min-shared", type=int, default=3)
    p.add_argument("--cluster-max-size", type=int, default=10)
    args = p.parse_args()

    revision = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    df = load_dataset(revision)
    texts = df["positive"].tolist()
    ids = df["chunk_id"].tolist()
    print(f"Loaded {len(df)} rows from {REPO_ID}@{revision[:10]}")

    sets, pairs = shared_shingle_counts(texts)
    equivalents = equivalent_chunks(sets, pairs, args.equiv_containment)
    clusters = overlap_clusters(len(texts), pairs, args.cluster_min_shared)

    rng = random.Random(args.seed)
    held_out = pick_holdout(clusters, len(df), args.unanswerable_frac, args.cluster_max_size, rng)

    corpus = [
        {
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "source": r.source,
            "bucket": r.bucket,
            "token_count": int(r.token_count),
            "text": r.positive,
        }
        for i, r in enumerate(df.itertuples())
        if i not in held_out
    ]

    rows = []
    for i, r in enumerate(df.itertuples()):
        answerable = i not in held_out
        rows.append(
            {
                "qid": f"q_{r.chunk_id}",
                "query": r.query,
                "answerable": answerable,
                "unanswerable_reason": None if answerable else "gold_removed",
                "source_chunk_id": r.chunk_id,
                "gold_chunk_ids": sorted([r.chunk_id] + [ids[j] for j in equivalents.get(i, ())])
                if answerable
                else [],
                "gold_spans": None,
                "doc_id": r.doc_id,
                "source": r.source,
                "bucket": r.bucket,
                "query_type": r.query_type,
                "legal_domain": r.legal_domain,
                "difficulty": r.difficulty,
                "passage": r.positive,
            }
        )
    assign_splits(rows, args.test_frac, args.dev_frac, rng)

    for line in OUT_OF_SCOPE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            rows.append(
                {
                    "qid": q["qid"],
                    "query": q["query"],
                    "answerable": False,
                    "unanswerable_reason": "out_of_scope",
                    "source_chunk_id": None,
                    "gold_chunk_ids": [],
                    "gold_spans": None,
                    "doc_id": None,
                    "source": None,
                    "bucket": None,
                    "query_type": "out_of_scope",
                    "legal_domain": q.get("legal_domain", "other"),
                    "difficulty": None,
                    "passage": None,
                    "split": q["split"],
                }
            )

    to_annotate = pick_for_annotation(rows, args.n_annotate, rng)
    template = [
        {
            "qid": r["qid"],
            "query_type": r["query_type"],
            "difficulty": r["difficulty"],
            "query": r["query"],
            "passage": r["passage"],
            "answerable": None,
            "quotes": [],
            "notes": "",
        }
        for r in to_annotate
    ]

    annotation_counts = None
    if ANNOTATIONS.exists():
        annotation_counts = merge_annotations(rows, corpus, ANNOTATIONS)
        print(f"Merged annotations from {ANNOTATIONS}: {annotation_counts}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "corpus.jsonl", corpus)
    write_jsonl(args.out / "queries.jsonl", rows)
    write_jsonl(args.out / "annotation_template.jsonl", template)

    split_counts = {
        split: {
            "queries": sum(r["split"] == split for r in rows),
            "unanswerable": sum(r["split"] == split and not r["answerable"] for r in rows),
        }
        for split in ["train", "dev", "test"]
    }
    manifest = {
        "dataset": REPO_ID,
        "revision": revision,
        "files": [f"hungarian/eu/{s}_{b}.parquet" for s in SOURCES for b in BUCKETS],
        "licence": "CC BY-NC-ND 4.0",
        "params": {k: v for k, v in vars(args).items() if k != "out"},
        "counts": {
            "rows": len(df),
            "corpus_chunks": len(corpus),
            "held_out_chunks": len(held_out),
            "overlap_clusters": len(clusters),
            "queries_with_equivalent_gold": sum(
                1 for r in rows if r["answerable"] and len(r["gold_chunk_ids"]) > 1
            ),
            "splits": split_counts,
            "annotation_template": len(template),
            "annotations_merged": annotation_counts,
        },
        "breakdowns": {
            k: dict(Counter(r[k] for r in rows if r["split"] == "test" and r[k] is not None))
            for k in BREAKDOWNS
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Corpus: {len(corpus)} chunks ({len(held_out)} held out)")
    for split, c in split_counts.items():
        print(f"{split:>5}: {c['queries']} queries, {c['unanswerable']} unanswerable")
    print(
        f"Annotation template: {len(template)} test queries -> {args.out / 'annotation_template.jsonl'}"
    )


if __name__ == "__main__":
    main()
