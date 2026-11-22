"""CI checks for the WER pipeline and the committed results (run: python3 test_wer_ci.py).

No model, no audio, no network. Two parts:
  1. WER on a few hand-written sentence pairs with hand-counted edits. They are
     written for this test, not taken from FLEURS: the FLEURS reference text is
     not redistributed, and per-sentence predictions were not saved in results/.
  2. results/run_meta.json is internally consistent, and every WER/CER figure in
     README.md and MODEL_CARD.md matches it after rounding to one decimal.
"""
import json
import re
from pathlib import Path

from asr_metrics import corpus_rate, per_sentence

ROOT = Path(__file__).parent

# (reference, hypothesis, expected word edits after normalization, reference words)
PAIRS = [
    ("Muraho, amakuru yawe?", "muraho amakuru yanyu", 1, 3),   # 1 substitution
    ("Abana bagiye ku ishuri.", "abana bagiye ishuri", 1, 4),   # 1 deletion
    ("Ndagukunda cyane", "nda gukunda cyane", 2, 2),            # 1 substitution + 1 insertion
    ("Ejo tuzahura", "Ejo tuzahura.", 0, 2),                    # case/punctuation ignored
    ("Yaje n'abana", "yaje nabana", 1, 2),                      # apostrophe is kept
]

refs = [r for r, _, _, _ in PAIRS]
hyps = [h for _, h, _, _ in PAIRS]
stats = per_sentence(hyps, refs, "word")
assert stats == [(e, n) for _, _, e, n in PAIRS], stats
assert abs(corpus_rate(stats) - 5 / 13) < 1e-12     # corpus WER, not mean of sentence WERs
try:
    import jiwer
    from asr_metrics import normalize_text
    jw = jiwer.wer([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps])
    assert abs(jw - 5 / 13) < 1e-12
    print("jiwer cross-check: OK")
except ImportError:
    print("jiwer not installed: cross-check skipped")

meta = json.loads((ROOT / "results" / "run_meta.json").read_text())
for subset in ("full", "first30"):
    r = meta["results"][subset]
    assert abs(r["paired_wer_diff"] - (r["ft_wer"] - r["base_wer"])) < 1e-9, subset
    lo, hi = r["paired_wer_diff_ci95"]
    assert lo <= r["paired_wer_diff"] <= hi, subset


def norm(text):
    return re.sub(r"\s+", " ", text.replace("−", "-"))


def pct(x):
    return f"{x * 100:.1f}%"


def points(x):
    return f"{x * 100:.1f}"


def expected_strings(r):
    lo, hi = r["paired_wer_diff_ci95"]
    return [pct(r[k]) for k in ("ft_wer", "base_wer", "ft_cer", "base_cer")] + [
        f"{points(r['paired_wer_diff'])} points, 95% CI [{points(lo)}, {points(hi)}]",
    ]


readme = norm((ROOT / "README.md").read_text())
card = norm((ROOT / "MODEL_CARD.md").read_text())
full, first30 = meta["results"]["full"], meta["results"]["first30"]
for doc_name, doc, wanted in (
    ("README.md", readme, expected_strings(full) + expected_strings(first30)),
    ("MODEL_CARD.md", card, expected_strings(full)),
):
    missing = [s for s in wanted if s not in doc]
    assert not missing, f"{doc_name} does not match run_meta.json: {missing}"

print("ALL WER CI CHECKS PASS")
