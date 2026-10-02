"""
Text overlap between passages of the JudicialMind Hungarian slice.

The dataset's passages overlap heavily: they are cut from 10 EU documents at
three window sizes (buckets A/B/C) and split further into `_sN` sub-chunks,
so about half of all passages share most of their text with another one.
Two eval decisions depend on this:

- Retrieval gold: a chunk that contains the gold passage is as good a hit as
  the gold passage itself (`equivalent_chunks`).
- Unanswerable questions: removing only the gold chunk from the index leaves
  copies of the answer behind, so whole overlap clusters are held out
  (`overlap_clusters`).

Overlap is measured on word 8-gram shingles, lowercased.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

SHINGLE_SIZE = 8

_WORD = re.compile(r"\w+")


def shingles(text: str, n: int = SHINGLE_SIZE) -> Set[str]:
    words = _WORD.findall(text.lower())
    return {" ".join(words[i : i + n]) for i in range(max(len(words) - n + 1, 1))}


def shared_shingle_counts(texts: List[str]) -> Tuple[List[Set[str]], Dict[Tuple[int, int], int]]:
    """Shingle sets per text, and shared-shingle counts for every overlapping pair (i < j)."""
    sets = [shingles(t) for t in texts]
    postings: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for g in s:
            postings[g].append(i)

    pairs: Dict[Tuple[int, int], int] = {}
    for i, s in enumerate(sets):
        counts = Counter(j for g in s for j in postings[g] if j > i)
        for j, n in counts.items():
            pairs[(i, j)] = n
    return sets, pairs


def equivalent_chunks(
    sets: List[Set[str]], pairs: Dict[Tuple[int, int], int], min_containment: float = 0.9
) -> Dict[int, Set[int]]:
    """For each text i, the texts j that contain at least `min_containment` of i's shingles."""
    out: Dict[int, Set[int]] = defaultdict(set)
    for (i, j), n in pairs.items():
        if n / len(sets[i]) >= min_containment:
            out[i].add(j)
        if n / len(sets[j]) >= min_containment:
            out[j].add(i)
    return out


def overlap_clusters(
    n_texts: int, pairs: Dict[Tuple[int, int], int], min_shared: int = 3
) -> List[List[int]]:
    """Connected components of texts linked by at least `min_shared` shared shingles."""
    parent = list(range(n_texts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), n in pairs.items():
        if n >= min_shared:
            parent[find(i)] = find(j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n_texts):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])
