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

## Results so far

| System | N | WER | CER |
|---|---|---|---|
| MMS zero-shot (`facebook/mms-1b-all`, no fine-tuning) | 30 | 40.4% | 8.8% |

*More rows to come: MMS adapter fine-tune, Whisper + Swahili proxy token,
Whisper + new `<rw>` token.*

## Notes / known issues

- CTC-style errors dominate (word-splitting, e.g. "cyanecyane" → "cyane cyane")
  rather than semantic errors — CER is much lower than WER.
- Code-switched loanwords (e.g. English "gypsy") are mistranscribed.
- Full dev/test set evaluation deferred to GPU (Kaggle) for speed; current
  numbers are on a 30-sample subset and will be refreshed on the full set
  before final reporting.

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
