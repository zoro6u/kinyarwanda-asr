"""
asr_metrics.py — WER/CER with sentence-level bootstrap confidence intervals.

Pure Python (no torch / transformers), so it can be unit-tested anywhere.

normalize_text() is byte-for-byte the normalization used in the original
Kaggle notebook (lowercase, strip everything except word chars / whitespace /
apostrophe, collapse whitespace), so numbers computed here are comparable
with the committed README table.

Corpus-level WER is sum(edits) / sum(reference words) — not the mean of
per-sentence rates — which is also what `evaluate`/`jiwer` report.
"""

import random
import re
from typing import List, Sequence, Tuple

Stat = Tuple[int, int]  # (edit_count, reference_length)


def normalize_text(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance (substitution/insertion/deletion each cost 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sentence_stat(pred: str, ref: str, unit: str = "word") -> Stat:
    """(edits, reference_length) for one sentence. unit: 'word' or 'char'."""
    if unit == "word":
        p, r = pred.split(), ref.split()
    elif unit == "char":
        p, r = list(pred), list(ref)
    else:
        raise ValueError("unit must be 'word' or 'char'")
    return _levenshtein(p, r), len(r)


def per_sentence(preds: List[str], refs: List[str], unit: str = "word") -> List[Stat]:
    if len(preds) != len(refs):
        raise ValueError(f"length mismatch: {len(preds)} preds vs {len(refs)} refs")
    return [sentence_stat(normalize_text(p), normalize_text(r), unit) for p, r in zip(preds, refs)]


def corpus_rate(stats: List[Stat]) -> float:
    total = sum(n for _, n in stats)
    if total == 0:
        raise ValueError("reference length is zero")
    return sum(e for e, _ in stats) / total


def bootstrap_ci(stats: List[Stat], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI for the corpus rate, resampling SENTENCES with replacement."""
    rng = random.Random(seed)
    n = len(stats)
    rates = []
    for _ in range(n_boot):
        sample = [stats[rng.randrange(n)] for _ in range(n)]
        if sum(x for _, x in sample) == 0:
            continue
        rates.append(corpus_rate(sample))
    rates.sort()
    lo = rates[int((alpha / 2) * len(rates))]
    hi = rates[min(len(rates) - 1, int((1 - alpha / 2) * len(rates)))]
    return lo, hi


def paired_bootstrap_diff(stats_a: List[Stat], stats_b: List[Stat],
                          n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """
    rate(A) - rate(B) with a paired sentence-level bootstrap: the SAME sentence
    indices are drawn for both systems on every resample, which is what makes
    the interval much tighter than comparing two independent intervals.
    Returns (diff, lo, hi). If the interval excludes 0 the difference is
    unlikely to be sampling noise *on this sample of sentences*.
    """
    if len(stats_a) != len(stats_b):
        raise ValueError("systems must be scored on the same sentences")
    rng = random.Random(seed)
    n = len(stats_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [stats_a[i] for i in idx]
        b = [stats_b[i] for i in idx]
        if sum(x for _, x in a) == 0:
            continue
        diffs.append(corpus_rate(a) - corpus_rate(b))
    diffs.sort()
    lo = diffs[int((alpha / 2) * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int((1 - alpha / 2) * len(diffs)))]
    return corpus_rate(stats_a) - corpus_rate(stats_b), lo, hi
