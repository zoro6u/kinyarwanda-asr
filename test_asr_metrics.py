"""Plain-script checks for asr_metrics.py (run: python3 test_asr_metrics.py).
The jiwer cross-check runs only if jiwer is installed (it is in the README's setup)."""
import random
from asr_metrics import *

assert normalize_text("Hello,  World!  It's ok.") == "hello world it's ok"
assert sentence_stat("a b c", "a b c") == (0, 3)
assert sentence_stat("a x c", "a b c") == (1, 3)      # substitution
assert sentence_stat("a c", "a b c") == (1, 3)        # deletion
assert sentence_stat("a b b c", "a b c") == (1, 3)    # insertion
assert sentence_stat("", "a b") == (2, 2)
assert sentence_stat("abc", "abd", "char") == (1, 3)

rng = random.Random(1)
vocab = ["ni", "mwiza", "umuntu", "inzu", "ejo", "kuri", "nta", "cyane", "abana", "ikaze"]
refs, preds = [], []
for _ in range(60):
    r = [rng.choice(vocab) for _ in range(rng.randint(3, 12))]
    p = [w if rng.random() > 0.3 else rng.choice(vocab + [""]) for w in r]
    if rng.random() < 0.2:
        p = p[:-1]
    refs.append(" ".join(r)); preds.append(" ".join(x for x in p if x))
st = per_sentence(preds, refs, "word")
try:
    import jiwer
    assert abs(corpus_rate(st) - jiwer.wer(refs, preds)) < 1e-9
    assert abs(corpus_rate(per_sentence(preds, refs, "char")) - jiwer.cer(refs, preds)) < 1e-9
    print("jiwer cross-check: OK")
except ImportError:
    print("jiwer not installed: cross-check skipped")

lo, hi = bootstrap_ci(st, 1000)
assert lo <= corpus_rate(st) <= hi
d, dlo, dhi = paired_bootstrap_diff(st, st, 500)
assert (d, dlo, dhi) == (0, 0, 0)
st_b = [(max(0, e - 1) if i % 5 == 0 else e, n) for i, (e, n) in enumerate(st)]
d, dlo, dhi = paired_bootstrap_diff(st, st_b, 1000)
la, ha = bootstrap_ci(st, 1000); lb, hb = bootstrap_ci(st_b, 1000)
assert (dhi - dlo) < (ha - la) + (hb - lb)      # pairing tightens the interval

for bad in (lambda: per_sentence(["a"], ["a", "b"]),
            lambda: paired_bootstrap_diff([(0, 1)], [(0, 1), (0, 1)]),
            lambda: corpus_rate([(0, 0)])):
    try:
        bad(); raise SystemExit("expected ValueError")
    except ValueError:
        pass
print("ALL asr_metrics CHECKS PASS")
