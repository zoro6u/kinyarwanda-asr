---
license: cc-by-nc-4.0
base_model: facebook/mms-1b-all
language:
- rw
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- kinyarwanda
- mms
- adapter
metrics:
- wer
- cer
---

# MMS-1B Kinyarwanda adapter (fine-tuned on 3,000 Common Voice clips)

A small adapter file (`adapter.kin.safetensors`, 2,151,168 parameters, 8.6 MB)
for [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all).
It is **not** a standalone model: load the base model first, then apply this file.

**Licence and intended use: CC-BY-NC-4.0, research and non-commercial use only.**
This adapter is adapted material derived from `facebook/mms-1b-all` (Meta,
CC-BY-NC-4.0). Changes made: the adapter layers (288 tensors, 0.223% of the
964,829,197 parameters) were further trained on Kinyarwanda speech. The model
is not suitable for production use at its current error rate (see below).

## Starting point

Training did not start from scratch. MMS-1B-all already includes a pretrained
Kinyarwanda (`kin`) adapter, and that adapter was the initialization here. The
meaningful comparison is therefore this adapter vs. that pretrained adapter,
not vs. a model that had never seen the language.

## Results

Evaluation set: `mbazaNLP/fleurs-kinyarwanda` dev split (323 sentences; the
dataset is gated and is not redistributed here). Corpus-level WER/CER after
lowercasing and removing punctuation; 95% percentile bootstrap over sentences.

| System | WER (95% CI) | CER (95% CI) |
|---|---|---|
| MMS-1B-all, pretrained `kin` adapter | 49.4% (45.5-53.5) | 19.8% (16.4-23.2) |
| + this adapter | **45.5%** (41.6-49.6) | **18.9%** (15.5-22.3) |

Paired WER difference: -3.9 points, 95% CI [-4.7, -3.0]. The CER difference was
not tested for significance. A WER near 45% means roughly every second word is
wrong.

## Training

Common Voice 27.0 Kinyarwanda (CC0-1.0), first 3,000 rows of `rw/train.tsv`;
first 200 rows of `rw/dev.tsv` for validation. Seed 42, learning rate 1e-4,
effective batch 16, 3 epochs, 100 warm-up steps, fp16, best checkpoint by
validation loss (0.2546). About 50 minutes on one Kaggle GPU (torch
2.11.0+cu128, transformers 5.16.1). Training code, evaluation code and the full
run record are in the project repository: https://github.com/zoro6u/kinyarwanda-asr

## Usage

This is the same loading sequence the training script uses in its round-trip
check (the reloaded adapter reproduced the fine-tuned model's predictions on
the first five test sentences).

```python
import torch, librosa
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Wav2Vec2ForCTC

BASE = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(BASE, target_lang="kin")
model = Wav2Vec2ForCTC.from_pretrained(BASE, target_lang="kin", ignore_mismatched_sizes=True)
model.load_adapter("kin")                       # the pretrained starting point

path = hf_hub_download("<this-repo-id>", "adapter.kin.safetensors")
result = model.load_state_dict(load_file(path), strict=False)
assert not result.unexpected_keys               # only adapter tensors were saved

audio, _ = librosa.load("clip.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = torch.argmax(model(**inputs).logits, dim=-1)[0]
print(processor.decode(ids))
```

## Limitations

- Evaluated on one read-speech benchmark; behaviour on spontaneous, noisy or
  accented speech is unknown.
- Single training run and seed; no variation across seeds measured.
- Training audio duration was not measured (only the clip count).
- Overlap between Common Voice training sentences and the evaluation sentences
  was not checked.
- Errors were not analysed by category.

## Attribution

Base model: Pratap et al., *Scaling Speech Technology to 1,000+ Languages*
(arXiv:2305.13516), `facebook/mms-1b-all`, CC-BY-NC-4.0.
