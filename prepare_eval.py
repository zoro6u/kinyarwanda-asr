"""
Stage 0: Unified evaluation harness for Kinyarwanda ASR comparison.
Loads mbazaNLP/fleurs-kinyarwanda (community FLEURS reconstruction) and
provides a single normalize + WER/CER function to use across all models
(MMS zero-shot, MMS adapter fine-tune, Whisper proxy-token, Whisper new-token).
"""

import re
import pandas as pd
from huggingface_hub import hf_hub_download

try:
    import evaluate
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
except ImportError:
    raise SystemExit("Run: pip install evaluate jiwer --break-system-packages")

COLUMNS = [
    "id",
    "filename",
    "raw_transcription",
    "normalized_transcription",
    "phonemic",
    "num_samples",
    "gender",
]

SPLITS = ["train", "dev", "test"]
REPO_ID = "mbazaNLP/fleurs-kinyarwanda"


def load_split(split_name: str) -> pd.DataFrame:
    """Download and parse one .tsv split, skipping malformed rows."""
    path = hf_hub_download(repo_id=REPO_ID, filename=f"{split_name}.tsv", repo_type="dataset")
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip", engine="python", header=None)
    df.columns = COLUMNS[: df.shape[1]]
    df["split"] = split_name
    return df


def load_all_splits() -> dict:
    return {s: load_split(s) for s in SPLITS}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation (keep apostrophes), collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer_cer(predictions: list[str], references: list[str]) -> dict:
    """Compute normalized WER and CER for a list of predictions vs references."""
    norm_preds = [normalize_text(p) for p in predictions]
    norm_refs = [normalize_text(r) for r in references]
    wer = wer_metric.compute(predictions=norm_preds, references=norm_refs)
    cer = cer_metric.compute(predictions=norm_preds, references=norm_refs)
    return {"wer": wer, "cer": cer}


if __name__ == "__main__":
    splits = load_all_splits()
    for name, df in splits.items():
        print(f"{name}: {df.shape[0]} rows")
    print()
    print("Sample row (dev):")
    print(splits["dev"][["filename", "normalized_transcription"]].iloc[0])

    # Sanity check: WER of a reference against itself should be 0
    sample_refs = splits["dev"]["normalized_transcription"].head(5).tolist()
    sanity = compute_wer_cer(sample_refs, sample_refs)
    print()
    print("Sanity check (predictions == references, expect wer=0, cer=0):", sanity)
