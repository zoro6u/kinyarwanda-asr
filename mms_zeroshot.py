"""
Stage 1: MMS zero-shot baseline for Kinyarwanda ASR.
Runs facebook/mms-1b-all on a small sample of the dev set (no training),
using the unified evaluation harness from prepare_eval.py.
"""

import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, AutoProcessor
from huggingface_hub import hf_hub_download
import zipfile
import io

from prepare_eval import load_split, compute_wer_cer

MODEL_ID = "facebook/mms-1b-all"
LANG_CODE = "kin"  # ISO 639-3 for Kinyarwanda
N_SAMPLES = 30       # keep small on CPU

print("Loading MMS model and processor (first run downloads ~4-5GB)...")
processor = AutoProcessor.from_pretrained(MODEL_ID, target_lang=LANG_CODE)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, target_lang=LANG_CODE, ignore_mismatched_sizes=True)
model.load_adapter(LANG_CODE)

print(f"Loading {N_SAMPLES} samples from dev split...")
dev_df = load_split("dev").head(N_SAMPLES)

predictions = []
references = []

for _, row in dev_df.iterrows():
    audio_path = f"audio_dev/dev_data/{row['filename']}"
    speech, sr = sf.read(audio_path)
    if sr != 16000:
        import torchaudio
        speech_tensor = torch.tensor(speech).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech_tensor).squeeze(0).numpy()
        sr = 16000

    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    transcription = processor.decode(ids)

    predictions.append(transcription)
    references.append(row["normalized_transcription"])
    print(f"REF:  {row['normalized_transcription']}")
    print(f"PRED: {transcription}")
    print()

results = compute_wer_cer(predictions, references)
print("=== MMS zero-shot results (N=%d) ===" % N_SAMPLES)
print(results)
