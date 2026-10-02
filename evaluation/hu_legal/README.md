# Hungarian legal eval set

Eval set for verbatim-rag-hu, built from the Hungarian slice of the
[JudicialMind Legal Training Dataset](https://huggingface.co/datasets/judicialmind/legal-training-dataset)
(`hungarian/eu/{eurlex_hu,lemur_hu}_{A,B,C}.parquet`, 10,023 query–passage pairs from 10 EU documents).

```bash
python -m evaluation.hu_legal.build_eval_set              # current dataset revision
python -m evaluation.hu_legal.build_eval_set --revision 8ed53a3527c78823ba12e979134dcdac152f8e55
```

The build is deterministic for a given revision and `--seed`. The output goes to `data/`, which is
gitignored because the source licence is CC BY-NC-ND 4.0.

## Outputs

| File | Contents |
| --- | --- |
| `corpus.jsonl` | Chunks to index: `chunk_id`, `doc_id`, `source`, `bucket`, `token_count`, `text`. Index each as one chunk keyed by `chunk_id`, bypassing `DocumentProcessor`. BM25 `fit()` runs on exactly these texts. |
| `queries.jsonl` | One row per query: `qid`, `query`, `split`, `answerable`, `unanswerable_reason`, `gold_chunk_ids`, `gold_spans`, and the breakdown fields `query_type`, `legal_domain`, `difficulty`, `bucket`, `source`. |
| `annotation_template.jsonl` | 100 answerable test queries for hand annotation of answer spans. |
| `manifest.json` | Dataset revision, parameters, counts and test-split breakdowns. |

Split sizes at revision `8ed53a35`: test 1,005 queries (125 unanswerable), dev 1,005 (125), train
8,033 (973). The corpus has 8,820 chunks after the holdout.

## Decisions that differ from the spec

**Buckets are passage lengths, not splits.** A, B and C hold passages of ≤256, ≤512 and ≤1,024
tokens. Bucket C covers only 2 of the 10 documents and only long passages, so it would be a biased
test set. Splits are drawn instead, stratified over answerability, source, bucket, difficulty and
query type: 10% test and 10% dev for tuning thresholds and RRF weights. The remaining 80% is kept
as train for later fine-tuning. `bucket` stays as a breakdown field.

**Gold chunks include overlapping chunks.** Passages are windows over the same 10 documents at
three sizes, plus `_sN` sub-splits. Half of them share most of their text with another passage.
A query's `gold_chunk_ids` holds its own passage plus every chunk containing ≥90% of that
passage's 8-word shingles, so retrieving a superset chunk counts as a hit. This applies to about
a third of the queries. `source_chunk_id` keeps the dataset's original single gold.

**Unanswerable questions remove whole overlap clusters.** Dropping only the gold chunk would leave
copies of the answer in the index. Chunks linked by ≥3 shared 8-word shingles form clusters, and
randomly chosen clusters of ≤10 chunks are left out of the corpus. Their queries become
unanswerable (`unanswerable_reason: "gold_removed"`), about 12% of every split. No such query's
passage shares more than 7% of its shingles with any chunk left in the index.
`out_of_scope.jsonl` adds 20 hand-written questions on topics the corpus does not cover
(`"out_of_scope"`, split between dev and test).

## Annotating gold spans

1. Copy `data/annotation_template.jsonl` to `annotations/gold_spans.jsonl`.
2. For each record, set `answerable` and fill `quotes`:
   - `"yes"`: the passage answers the query. Paste each answer sentence or phrase into `quotes`
     exactly as it appears in `passage`. Whitespace differences don't matter.
   - `"partial"`: the passage answers part of the query. Quote that part and say what is missing
     in `notes`.
   - `"no"`: the passage does not answer the query. Leave `quotes` empty. The query is then
     excluded from scoring, since its true gold is unknown.
   - If a quote occurs more than once in the passage, write `{"text": "...", "occurrence": 2}`.
   - Leave `answerable` as `null` to skip a record for now.
3. Re-run the build. It resolves quotes to character offsets in `gold_spans`, and resets
   `gold_chunk_ids` to every corpus chunk that contains all the spans. It stops with a list of
   errors if a quote is not found or is ambiguous.

`annotations/` is gitignored too. Check the licence before committing or sharing annotations.

## Caveats

- Queries are synthetic and some don't match their passage. For example, a query may ask about
  two articles while the passage ends partway through the first. Annotation catches these as
  `"no"` or `"partial"`. Report how often this happens before trusting retrieval numbers.
- All splits share one corpus, and overlapping passages mean a train query and a test query can
  target nearly the same text. This doesn't matter for phase 1, which trains nothing. It matters
  for the phase 3 extractor: split by overlap cluster before training on silver labels.
- The out-of-scope questions are first drafts. Have a native speaker review them.
