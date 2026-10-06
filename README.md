# Kinyarwanda ASR: Whisper fine-tuning vs Meta MMS

A small comparison of Whisper fine-tuning (reusing a related language's token
vs. adding a new language token) against Meta MMS (pretrained adapter and
adapter fine-tuning) for Kinyarwanda automatic speech recognition. Built as a
portfolio project; the numbers are small-scale and exploratory, and this README
says where they stop being trustworthy.

> **Read "What is and isn't held constant" before drawing any conclusion from
> the Whisper-vs-MMS gap.**

## Evaluation set

[`mbazaNLP/fleurs-kinyarwanda`](https://huggingface.co/datasets/mbazaNLP/fleurs-kinyarwanda),
dev split: **323 sentences**. It is a community-built FLEURS-style benchmark
(Google's official FLEURS does not include Kinyarwanda); the dataset is gated on
Hugging Face and requires accepting its access conditions.

- **Text normalization** (hypotheses and references alike): lowercase, remove
  everything except word characters, whitespace and apostrophes, collapse
  whitespace. References are the dataset's `normalized_transcription` column.
- **Metrics:** corpus-level WER/CER (sum of edits / sum of reference length;
  cross-checked against `jiwer`, see `test_asr_metrics.py`).
- **Uncertainty:** 95% percentile bootstrap over *sentences* (2,000 resamples).
  Comparisons between two systems on the same sentences use a *paired*
  bootstrap.

**Training / validation data:** [Common Voice 27.0 Kinyarwanda](https://mozilladatacollective.com)
via Mozilla Data Collective (CC0-1.0). Fine-tuning uses the **first 3,000 rows**
of `rw/train.tsv`; checkpoint selection uses the **first 200 rows** of
`rw/dev.tsv`. FLEURS is never used for selection. The audio is streamed from the
archive (no full 57 GB download).

## Results

### A. Full FLEURS dev set (n = 323), MMS only

| System | WER (95% CI) | CER (95% CI) |
|---|---|---|
| MMS-1B, pretrained `kin` adapter, no further training | 49.4% (45.5–53.5) | 19.8% (16.4–23.2) |
| MMS-1B + `kin` adapter fine-tuned on 3,000 Common Voice clips | **45.5%** (41.6–49.6) | **18.9%** (15.5–22.3) |

Paired WER difference (fine-tuned − pretrained): **−3.9 points, 95% CI
[−4.7, −3.0]**. The CER difference was not tested.

### B. First 30 FLEURS dev sentences, all systems

This is the subset used in the earlier stages of the project. The Whisper
checkpoints were not saved, so the Whisper systems were **not** re-evaluated on
the full set.

| System | Train data | WER | CER |
|---|---|---|---|
| MMS-1B + adapter fine-tune | 3,000 clips | 33.1% (26.7–40.6) | 7.4% (4.3–11.9) |
| MMS-1B, pretrained `kin` adapter | none | 40.6% (33.5–48.6) ¹ | 8.8% (5.6–13.3) |
| Whisper-small + new `<rw>` token (warm-started from Swahili) | 3,000 clips | 77.9% | 20.2% |
| Whisper-small + Swahili proxy token | 3,000 clips | 79.7% | 20.8% |
| Whisper-small + Swahili proxy token | 300 clips | 92.1% | 25.7% |

Paired WER difference for MMS on these 30 sentences: −7.5 points, 95% CI
[−10.5, −4.6]. Confidence intervals for the Whisper rows were not computed
(their predictions were not kept).

¹ An earlier run of `mms_zeroshot.py` reported 40.4%. The 0.2-point difference
is unexplained; the likely cause is the environment (library versions and
hardware differ), but that was not verified.

**The first 30 sentences are easier than the full set.** The pretrained MMS
baseline scores 40.6% WER on them but 49.4% on all 323, and its CER is 8.8% on
the subset versus 19.8% overall (outside the subset's own interval of
5.6–13.3%). Table B's absolute numbers should not be compared with Table A's.

## What is and isn't held constant

| | MMS | Whisper-small |
|---|---|---|
| Fine-tuning data | same 3,000 Common Voice clips | same 3,000 clips (and a 300-clip run) |
| Evaluation sentences and normalization | same | same |
| Already trained on Kinyarwanda? | **Yes**: MMS-1B-all ships a Kinyarwanda adapter, which is the starting point here | **No**: Whisper has no Kinyarwanda token, hence the proxy-token / new-token setups |
| Size | ~965M parameters | ~244M parameters (published figure) |
| Objective / decoding | CTC, greedy | autoregressive seq2seq |
| What was trained | 2,151,168 adapter parameters (288 tensors, 0.223% of the model) | full model |
| Hyperparameters | listed below | not tuned |

So the Whisper-vs-MMS comparison confounds architecture with prior exposure to
the language, model size and training recipe. It cannot say which architecture
is better for this language.

## Findings, and how far they go

1. **Fine-tuning the MMS adapter helps, modestly.** On the full 323-sentence set
   WER drops from 49.4% to 45.5% (paired CI excludes zero). An earlier version of
   this README reported a 7-point gain; that came from the easier 30-sentence
   subset. The CER change is within the noise of what was tested (CER was not
   compared in a paired test).
2. **MMS is far ahead of Whisper-small on the 30-sentence subset** (33–41% vs
   78–92% WER). The gap (more than 35 points) is large compared with the roughly
   ±7-point intervals around the MMS estimates, so the ranking on this subset is
   unlikely to be sampling noise. Whisper intervals were not computed. Because of
   the confounds above, this is not evidence that CTC-plus-adapter beats
   seq2seq fine-tuning in general.
3. **Proxy token vs new token is unresolved.** 77.9% vs 79.7% on 30 sentences
   cannot be told apart, and a paired test is not possible without the
   predictions. The original research question remains open.
4. **Whisper with 10x more data improved by about 12 points** (92.1% → 79.7%,
   300 → 3,000 clips). That is two points on one curve, measured on 30
   sentences; it does not support an extrapolation to larger data.

## MMS adapter training details

Seed 42; first 3,000 Common Voice rows for training, first 200 for validation;
learning rate 1e-4; per-device batch 2 × 8 gradient-accumulation steps
(effective 16); 3 epochs; 100 warm-up steps; fp16; gradient checkpointing;
evaluation every 100 steps; best checkpoint by validation loss (0.2546). Only
tensors whose name contains `adapter` are trainable. Run on one Kaggle GPU in
about 50 minutes (torch 2.11.0+cu128, transformers 5.16.1). All values are
recorded in [`results/run_meta.json`](results/run_meta.json).

## Limitations and not yet done

- One training run, one seed; no variation across seeds.
- Whisper systems: evaluated on 30 sentences only, checkpoints not saved.
  Re-running them with checkpoints and predictions saved would allow
  full-set numbers and paired tests.
- Training audio duration (hours) was not measured, only the number of clips.
- Overlap between Common Voice training sentences and FLEURS reference
  sentences was not checked.
- No error analysis yet: the large CER gap between the first 30 sentences and
  the rest of the set is unexplained.
- The CER improvement was not tested for significance.
- A single anecdotal observation (English loanwords such as "gypsy" being
  mis-transcribed by the pretrained MMS model) was not evaluated systematically.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers soundfile huggingface_hub pandas evaluate jiwer safetensors librosa datasets
```

## Reproduce

```bash
python3 test_asr_metrics.py      # metric code checks (no GPU needed)
python3 mms_zeroshot.py          # pretrained-adapter baseline on the first 30 sentences
```

The MMS adapter was trained and evaluated with `train_mms_adapter.py` on a
Kaggle GPU session. It needs two Kaggle secrets, `MDC_API_KEY` (Mozilla Data
Collective) and `HF_TOKEN` (a token from an account that has been granted
access to the gated FLEURS dataset, with write permission if you want the
script to upload the adapter). The Whisper fine-tuning notebooks are not yet
cleaned up for this repo.

## Artifacts and licence

The fine-tuned adapter (8.6 MB, 2,151,168 trainable parameters) is currently in a
private Hugging Face repository. When published it will carry **CC-BY-NC-4.0**,
the licence of its base model `facebook/mms-1b-all` (Meta): attribution is
required and use is **non-commercial only**. The adapter is adapted material
derived from that model and should be treated as non-commercial; this is not
legal advice. The published repository will not contain predictions or any
FLEURS reference text. See [`MODEL_CARD.md`](MODEL_CARD.md).

Common Voice is CC0-1.0. The FLEURS-style evaluation set has its own access
conditions, and its reference text is not redistributed here.
