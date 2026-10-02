"""
Turn hand-annotated answer quotes into character-offset gold spans.

Annotators copy the answer text out of the passage instead of typing offsets.
A quote is matched against the passage with whitespace runs treated as equal,
so line breaks lost or added by copy-paste do not matter; the offsets always
point into the original passage. A quote that occurs more than once must say
which occurrence it means: {"text": "...", "occurrence": 2} (1-based).
"""

import re
from typing import Dict, Iterable, List, Union

Quote = Union[str, Dict]

ANSWERABLE_VALUES = {"yes", "partial", "no"}


class SpanError(ValueError):
    pass


def _quote_pattern(text: str) -> re.Pattern:
    words = text.split()
    if not words:
        raise SpanError("empty quote")
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def resolve_quote(passage: str, quote: Quote) -> Dict:
    """Return {"start", "end", "text"} for a quote, with text taken from the passage."""
    if isinstance(quote, str):
        text, occurrence = quote, None
    else:
        text, occurrence = quote["text"], quote.get("occurrence")

    matches = list(_quote_pattern(text).finditer(passage))
    if not matches:
        raise SpanError(f"quote not found in passage: {text[:80]!r}")
    if occurrence is None:
        if len(matches) > 1:
            raise SpanError(f"quote occurs {len(matches)} times, add 'occurrence': {text[:80]!r}")
        m = matches[0]
    else:
        if not 1 <= occurrence <= len(matches):
            raise SpanError(f"occurrence {occurrence} of {len(matches)}: {text[:80]!r}")
        m = matches[occurrence - 1]
    return {"start": m.start(), "end": m.end(), "text": m.group(0)}


def resolve_annotation(record: Dict) -> Dict:
    """Validate one annotation record and return its resolved gold fields.

    Returns {"answerable", "gold_spans"}; gold_spans are sorted by start offset.
    """
    answerable = record.get("answerable")
    if answerable not in ANSWERABLE_VALUES:
        raise SpanError(f"{record['qid']}: answerable must be one of {sorted(ANSWERABLE_VALUES)}")
    quotes = record.get("quotes") or []
    if answerable == "no" and quotes:
        raise SpanError(f"{record['qid']}: answerable is 'no' but quotes are given")
    if answerable != "no" and not quotes:
        raise SpanError(f"{record['qid']}: answerable is {answerable!r} but no quotes are given")

    try:
        spans = [resolve_quote(record["passage"], q) for q in quotes]
    except SpanError as e:
        raise SpanError(f"{record['qid']}: {e}") from None
    return {"answerable": answerable, "gold_spans": sorted(spans, key=lambda s: s["start"])}


def chunks_containing(spans: List[Dict], corpus: Iterable[Dict]) -> List[str]:
    """Ids of corpus chunks that contain every span verbatim (whitespace-insensitive).

    Once a query has gold spans, this replaces the shingle-based gold chunk set:
    any chunk that holds the whole answer is a correct retrieval.
    """
    patterns = [_quote_pattern(s["text"]) for s in spans]
    return [c["chunk_id"] for c in corpus if all(p.search(c["text"]) for p in patterns)]
