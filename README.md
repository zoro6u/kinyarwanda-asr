# Kinyarwanda ASR: Whisper Fine-Tuning vs Meta MMS

Comparing Whisper fine-tuning (proxy-language token vs. new language token)
against Meta MMS (zero-shot and adapter fine-tuning) for Kinyarwanda
automatic speech recognition — a controlled benchmark on a low-resource
African language, built as a CMU-Africa AI portfolio project.

## Evaluation set

[`mbazaNLP/fleurs-kinyarwanda`](https://huggingface.co/datasets/mbazaNLP/fleurs-kinyarwanda) —
a community-built FLEURS-style benchmark (Google's official FLEURS does not
currently include Kinyarwanda). Created by 29 linguists; dev split used below
(323 samples total, 30-sample subset for early-stage results).

Training data: [Common Voice 27.0 Kinyarwanda](https://mozilladatacollective.com)
via Mozilla Data Collective (CC0-1.0), streamed directly from the archive
(no full 57GB download required).

## Results

| System | Train data | N (eval) | WER | CER |
|---|---|---|---|---|
| **MMS-1B + adapter fine-tune** | **3,000 samples** | **30** | **33.1%** | **7.4%** |
| MMS-1B zero-shot (`facebook/mms-1b-all`, no fine-tuning) | — | 30 | 40.4% | 8.8% |
| Whisper-small + new `<rw>` token (warm-started from Swahili) | 3,000 samples | 30 | 77.9% | 20.2% |
| Whisper-small + Swahili proxy token | 3,000 samples | 30 | 79.7% | 20.8% |
| Whisper-small + Swahili proxy token | 300 samples | 30 | 92.1% | 25.7% |

## Key findings

- **MMS adapter fine-tuning is the clear winner**, beating even MMS
  zero-shot despite fine-tuning only 288 parameters (~0.03% of the
  964M-parameter model) on the same 3,000 samples used for Whisper. This
  suggests a small amount of targeted adaptation on top of MMS's massively
  multilingual pretraining is far more data-efficient than fine-tuning
  Whisper from a single proxy language.
- **Architecture choice mattered far more than the proxy-vs-new-token
  decision.** Whisper's two fine-tuning variants (proxy token vs.
  warm-started new token) differ by only ~2 WER points from each other, but
  both trail MMS's adapter fine-tune by ~45 points. The original research
  question ("does a new language token beat reusing a related language's
  token?") turned out to be secondary to a bigger factor: CTC + lightweight
  adapters (MMS) generalize far better than seq2seq fine-tuning via language
  token substitution (Whisper) at this data scale.
- Increasing Whisper's training data 10x (300 → 3,000 samples) reduced WER
  by ~12 points, the largest single improvement in the Whisper track —
  suggesting Whisper might close the gap with much more data (10,000+
  samples), though this was not tested due to time/compute constraints.
- CTC-style errors (MMS) tend toward word-boundary splitting; Whisper's
  errors are more phonetic/spelling-level, consistent with their different
  decoding mechanisms (CTC vs. autoregressive generation).
- Code-switched loanwords (e.g. English "gypsy") are mistranscribed by MMS
  zero-shot; not systematically evaluated after fine-tuning.
- All fine-tuning done via streaming (no full dataset download): a
  presigned URL fetched from the Mozilla Data Collective API, read directly
  into `tarfile` in streaming mode (`r|gz`), extracting only matching
  filenames rather than downloading and unpacking the full 57GB archive.
- Full dev/test set evaluation deferred; current numbers are on a 30-sample
  subset and would need to be refreshed on the full 323-sample set for a
  publication-grade result.

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

Fine-tuning (Whisper proxy-token, Whisper new-token, and MMS adapter) was
run on Kaggle (2×T4 GPU) via streaming from Mozilla Data Collective; training
notebook code is being cleaned up for inclusion in this repo.
