"""
train_mms_adapter.py — retrain the MMS-1B Kinyarwanda adapter, KEEP the result,
and evaluate it on the full FLEURS-rw dev set with bootstrap confidence intervals.

Why this exists: the original Kaggle notebook trained the adapter into
/kaggle/temp (not persisted) and never called save/push, so the weights that
produced the committed 33.1% WER no longer exist. This script reproduces that
setup (same data slices, same hyperparameters) and fixes the three gaps:
  1. seeds are set explicitly,
  2. the adapter is saved to /kaggle/working and pushed to the Hub (PRIVATE),
  3. evaluation covers all FLEURS dev rows, scores the pretrained-'kin'-adapter
     baseline and the fine-tuned adapter on the SAME sentences, and reports
     paired-bootstrap CIs.

Run on a Kaggle GPU session with internet on and two secrets set:
MDC_API_KEY (Mozilla Data Collective) and HF_TOKEN (write access).

STATUS: the metrics (asr_metrics.py) are unit-tested. This script itself has NOT
been run — it was written without GPU / Hugging Face access. Treat the first
run as a test of this file, not as a result.
"""

import os

# one GPU only: avoids the DataParallel OOM the notebook worked around with
# `trainer.args._n_gpu = 1`. Must be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import csv
import io
import json
import random
import tarfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import numpy as np
import pandas as pd
import requests
import torch
import transformers
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, Trainer, TrainingArguments, Wav2Vec2ForCTC, set_seed

from asr_metrics import (bootstrap_ci, corpus_rate, normalize_text,
                         paired_bootstrap_diff, per_sentence)

SEED = 42
N_TRAIN, N_VAL = 3000, 200            # first N rows of CV train.tsv / dev.tsv, as in the notebook
MMS_ID = "facebook/mms-1b-all"
MDC_URL = "https://mozilladatacollective.com/api/datasets/cmu5k4i0f00p2mi0771dl1mg8/download"
HF_REPO = "zoro6u/mms-1b-kinyarwanda-adapter"
WORK, OUT = "/kaggle/temp", "/kaggle/working/mms-rw-adapter"


# ---------------------------------------------------------------- data ----
def get_download_url() -> str:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("MDC_API_KEY")
    r = requests.post(MDC_URL, headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()["downloadUrl"]          # presigned: never print or log it


def _open_stream(url):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    resp.raw.decode_content = True
    return resp, tarfile.open(fileobj=resp.raw, mode="r|gz")


def read_tsvs(url):
    found = {"rw/train.tsv": None, "rw/dev.tsv": None}
    resp, tar = _open_stream(url)
    with tar:
        for m in tar:
            for name in found:
                if m.name.endswith(name):
                    found[name] = tar.extractfile(m).read()
            if all(v is not None for v in found.values()):
                break
    resp.close()
    missing = [k for k, v in found.items() if v is None]
    assert not missing, f"not found in archive: {missing}"
    parse = lambda b: pd.read_csv(io.BytesIO(b), sep="\t", quoting=csv.QUOTE_NONE)
    return parse(found["rw/train.tsv"]), parse(found["rw/dev.tsv"])


def fetch_audio(url, filenames, dest):
    os.makedirs(dest, exist_ok=True)
    need = {f for f in filenames if not os.path.exists(os.path.join(dest, f))}
    if need:
        resp, tar = _open_stream(url)
        with tar:
            for m in tar:
                base = m.name.split("/")[-1]
                if base in need:
                    with open(os.path.join(dest, base), "wb") as f:
                        f.write(tar.extractfile(m).read())
                    need.discard(base)
                    if not need:
                        break
        resp.close()
    assert not need, f"{len(need)} audio files not found in archive"


def load_fleurs_dev() -> pd.DataFrame:
    repo = "mbazaNLP/fleurs-kinyarwanda"
    tsv = hf_hub_download(repo, "dev.tsv", repo_type="dataset")
    df = pd.read_csv(tsv, sep="\t", on_bad_lines="skip", engine="python", header=None)
    cols = ["id", "filename", "raw_transcription", "normalized_transcription",
            "phonemic", "num_samples", "gender"]
    df.columns = cols[:df.shape[1]]
    z = hf_hub_download(repo, "audio/dev.tar.xz", repo_type="dataset")
    with tarfile.open(z, "r:xz") as t:
        t.extractall(f"{WORK}/fleurs_dev")
    df["local_audio_path"] = df["filename"].apply(lambda f: f"{WORK}/fleurs_dev/dev_data/{f}")
    assert df["local_audio_path"].apply(os.path.exists).all(), "some FLEURS audio missing"
    print(f"FLEURS dev rows loaded: {len(df)}  (README states 323)")
    return df


# --------------------------------------------------------------- model ----
def make_dataset(df, processor):
    def gen():
        for _, row in df.iterrows():
            audio, _ = librosa.load(row["local_audio_path"], sr=16000)
            yield {"input_values": processor(audio, sampling_rate=16000).input_values[0],
                   "labels": processor.tokenizer(row["normalized_sentence"]).input_ids}
    return Dataset.from_generator(gen)


@dataclass
class CTCCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        batch = self.processor.pad([{"input_values": f["input_values"]} for f in features],
                                   return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features],
                                              return_tensors="pt")
        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        return batch


def load_mms():
    processor = AutoProcessor.from_pretrained(MMS_ID, target_lang="kin")
    model = Wav2Vec2ForCTC.from_pretrained(MMS_ID, target_lang="kin", ignore_mismatched_sizes=True)
    model.load_adapter("kin")       # NOTE: warm start from MMS's own pretrained 'kin' adapter
    return processor, model


@torch.no_grad()
def transcribe(model, processor, paths) -> List[str]:
    model.eval()
    out = []
    for p in paths:
        audio, _ = librosa.load(p, sr=16000)
        x = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(model.device)
        out.append(processor.decode(torch.argmax(model(x).logits, dim=-1)[0]))
    return out


# ---------------------------------------------------------------- main ----
def summarize(name, preds, refs):
    w, c = per_sentence(preds, refs, "word"), per_sentence(preds, refs, "char")
    (wl, wh), (cl, ch) = bootstrap_ci(w), bootstrap_ci(c)
    print(f"{name:<34} N={len(refs):<4} WER {100*corpus_rate(w):5.1f}% [{100*wl:.1f}, {100*wh:.1f}]"
          f"  CER {100*corpus_rate(c):5.1f}% [{100*cl:.1f}, {100*ch:.1f}]")
    return w, c


def main():
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED); set_seed(SEED)
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")

    # data -----------------------------------------------------------------
    url = get_download_url()
    train_df, dev_df = read_tsvs(url)
    tr, va = train_df.head(N_TRAIN).copy(), dev_df.head(N_VAL).copy()
    fetch_audio(url, set(tr["path"]) | set(va["path"]), f"{WORK}/train_audio")
    for d, col in ((tr, "sentence"), (va, "sentence")):
        d["local_audio_path"] = d["path"].apply(lambda p: f"{WORK}/train_audio/{p}")
        d["normalized_sentence"] = d[col].apply(normalize_text)
    fleurs = load_fleurs_dev()
    refs = fleurs["normalized_transcription"].tolist()
    paths = fleurs["local_audio_path"].tolist()

    # baseline: MMS with its own pretrained 'kin' adapter (before any training here)
    processor, model = load_mms()
    model.to("cuda:0")
    base_preds = transcribe(model, processor, paths)

    # freeze everything except adapter tensors (same rule as the notebook) ----
    for n, p in model.named_parameters():
        p.requires_grad = "adapter" in n.lower()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in trainable)
    print(f"trainable tensors: {len(trainable)} | trainable params: {n_train_params:,} "
          f"({100 * n_train_params / total:.4f}% of {total:,})")

    args = TrainingArguments(
        output_dir=f"{WORK}/mms-rw-adapter-ckpt",
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,                      # effective batch 16
        num_train_epochs=3, learning_rate=1e-4, warmup_steps=100,
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=200,
        save_total_limit=2, logging_steps=25, fp16=True, gradient_checkpointing=True,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        seed=SEED, data_seed=SEED, report_to="none",
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=make_dataset(tr, processor),
                      eval_dataset=make_dataset(va, processor),
                      data_collator=CTCCollator(processor))
    trainer.train()
    best_eval_loss = trainer.state.best_metric

    # save the adapter (the thing the notebook lost) -------------------------
    os.makedirs(OUT, exist_ok=True)
    adapter_state = {k: v.detach().cpu().contiguous()
                     for k, v in model.state_dict().items() if "adapter" in k.lower()}
    adapter_path = f"{OUT}/adapter.kin.safetensors"
    save_file(adapter_state, adapter_path, metadata={"format": "pt"})
    print(f"saved {len(adapter_state)} tensors -> {adapter_path}")

    # evaluate fine-tuned vs baseline on the SAME sentences ------------------
    ft_preds = transcribe(model, processor, paths)
    sub = slice(0, 30)   # the 30-sample subset the committed README numbers used
    results = {}
    for tag, sl in (("full", slice(None)), ("first30", sub)):
        print(f"--- FLEURS dev: {tag} ---")
        bw, bc = summarize("pretrained 'kin' adapter (baseline)", base_preds[sl], refs[sl])
        fw, fc = summarize("fine-tuned adapter (this run)", ft_preds[sl], refs[sl])
        d, lo, hi = paired_bootstrap_diff(fw, bw)
        print(f"paired WER diff (ft - baseline): {100*d:+.1f} pts  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")
        results[tag] = {"ft_wer": corpus_rate(fw), "base_wer": corpus_rate(bw),
                        "ft_cer": corpus_rate(fc), "base_cer": corpus_rate(bc),
                        "paired_wer_diff": d, "paired_wer_diff_ci95": [lo, hi]}

    with open(f"{OUT}/predictions.jsonl", "w") as f:
        for i in range(len(refs)):
            f.write(json.dumps({"id": i, "ref": refs[i], "baseline": base_preds[i],
                                "finetuned": ft_preds[i]}, ensure_ascii=False) + "\n")

    # round-trip: prove the SAVED file reproduces the in-memory model ---------
    del trainer, model
    torch.cuda.empty_cache()
    _, fresh = load_mms()
    fresh.to("cuda:0")
    res = fresh.load_state_dict(load_file(adapter_path), strict=False)
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:3]}"
    assert not any("adapter" in k.lower() for k in res.missing_keys), "adapter tensors missing"
    again = transcribe(fresh, processor, paths[:5])
    assert again == ft_preds[:5], "saved adapter does NOT reproduce fine-tuned predictions"
    print("round-trip check: OK (reloaded adapter reproduces the first 5 predictions)")

    meta = {"seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL, "base_model": MMS_ID,
            "warm_start": "facebook/mms-1b-all pretrained 'kin' adapter",
            "trainable_tensors": len(trainable), "trainable_params": n_train_params,
            "total_params": total, "best_eval_loss": best_eval_loss,
            "hyperparameters": {k: getattr(args, k) for k in (
                "per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs",
                "learning_rate", "warmup_steps", "fp16", "gradient_checkpointing")},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "fleurs_rows": len(refs), "results": results,
            "wall_seconds": round(time.time() - t0)}
    with open(f"{OUT}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # push PRIVATE; flip to public manually after checking the base-model licence
    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=HF_REPO, repo_type="model")
    print(f"pushed (private) -> https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
