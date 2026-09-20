# Kinyarwanda ASR: Whisper Fine-Tuning vs Meta MMS

Comparing Whisper fine-tuning (proxy-language token vs. new language token)
against Meta MMS for Kinyarwanda automatic speech recognition — a controlled
benchmark on a low-resource African language, built as a CMU-Africa AI
portfolio project.

## Evaluation set

[`mbazaNLP/fleurs-kinyarwanda`](https://huggingface.co/datasets/mbazaNLP/fleurs-kinyarwanda) —
a community-built FLEURS-style benchmark (Google's official FLEURS does not
currently include Kinyarwanda). Created by 29 linguists; dev split used below
(323 samples total, 30-sample subset for early-stage results).

Training data: [Common Voice 27.0 Kinyarwanda](https://mozilladatacollective.com)
via Mozilla Data Collective (CC0-1.0), streamed directly from the archive
(no full 57GB download required).

## Results so far

| System | Train data | N (eval) | WER | CER |
|---|---|---|---|---|
| MMS zero-shot (`facebook/mms-1b-all`, no fine-tuning) | — | 30 | 40.4% | 8.8% |
| Whisper-small + Swahili proxy token (fine-tuned) | 300 samples | 30 | 92.1% | 25.7% |
| Whisper-small + Swahili proxy token (fine-tuned) | 3,000 samples | 30 | 79.7% | 20.8% |

*More rows to come: Whisper + new `<rw>` token, MMS adapter fine-tune,
larger Whisper training runs.*

## Notes / known issues

- **MMS zero-shot currently outperforms Whisper fine-tuning by a wide margin**,
  even at 3,000 training samples. This is a meaningful negative result: a
  massively multilingual model with no Kinyarwanda-specific training beats a
  targeted proxy-token fine-tune on a small dataset. The likely explanation is
  that MMS's cross-lingual pretraining scale outweighs targeted fine-tuning on
  a dataset this small.
- Increasing training data 10x (300 → 3,000 samples) reduced WER by ~12
  points, suggesting more data is the most likely path to closing the gap
  with MMS — not yet tested at larger scale (10,000+ samples).
- CTC-style errors dominate MMS output (word-splitting) while Whisper's
  errors are more phonetic/spelling-level, consistent with their different
  architectures (CTC vs. sequence-to-sequence generation).
- Code-switched loanwords (e.g. English "gypsy") are mistranscribed by MMS.
- All fine-tuning done via streaming (no full dataset download): a
  presigned URL fetched from the Mozilla Data Collective API, read directly
  into `tarfile` in streaming mode (`r|gz`), extracting only matching
  filenames rather than downloading and unpacking the full 57GB archive.
- Full dev/test set evaluation deferred; current numbers are on a 30-sample
  subset and will be refreshed on the full set before final reporting.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip install torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip install transformers soundfile huggingface_hub pandas evaluate jiwer --break-system-packages
```

## Reproduce

```bash
python3 mms_zeroshot.py
```

Whisper fine-tuning was run on Kaggle (2×T4 GPU); notebook code available on
request — training script is being cleaned up for inclusion in this repo.
