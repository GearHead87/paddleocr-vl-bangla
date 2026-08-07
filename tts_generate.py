#!/usr/bin/env python
"""
Generate Bengali speech audio from the OCR text (output.txt) using TWO models:

  1. facebook/mms-tts-ben            -> output_mms.wav       (VITS, 16 kHz)
  2. ehzawad/orpheus-bangla-emotional-tts -> output_orpheus.wav (Orpheus+SNAC, 24 kHz)

The page separators ("===== Page N =====") in output.txt are stripped at runtime.
Text is split into sentence-sized chunks; each chunk is synthesized and written as a
part file (resumable), then all parts are concatenated into one wav per model.

Usage:
    .venv/bin/python tts_generate.py --model both
    .venv/bin/python tts_generate.py --model mms --limit-chars 400        # quick test
    .venv/bin/python tts_generate.py --model orpheus --emotion normal

Notes:
  * A full book is many hours of audio. Orpheus is a 3B autoregressive model and is
    SLOW (roughly real-time-ish per chunk); use --limit-chars / --max-chunks to sample.
  * Re-running resumes: existing part files are skipped.
"""
import argparse
import glob
import os
import re

import numpy as np
import soundfile as sf
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAGE_RE = re.compile(r"^\s*=====\s*Page\s+\d+\s*=====\s*$", re.M)

# ---- Orpheus / Llama special tokens (canonical Orpheus TTS scheme) ----
SOH, EOT, EOH = 128259, 128009, 128260   # start-of-human, end-of-text, end-of-human
SOA, EOA = 128257, 128258                 # start-of-audio, end-of-audio
CODE_OFFSET = 128266                       # first audio-code token id


# --------------------------- text handling ---------------------------
def load_clean_text(path, limit_chars=None):
    txt = open(path, encoding="utf-8").read()
    txt = PAGE_RE.sub("", txt)                 # drop page markers
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    if limit_chars:
        txt = txt[:limit_chars]
    return txt


def split_sentences(text):
    parts = re.split(r"(?<=[।!?\.])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(text, max_chars):
    chunks, cur = [], ""
    for s in split_sentences(text):
        if len(s) > max_chars:                 # hard-split an over-long sentence
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(s), max_chars):
                chunks.append(s[i:i + max_chars])
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def concat_parts(parts_dir, out_path, sr, gap_s=0.3):
    files = sorted(glob.glob(os.path.join(parts_dir, "chunk_*.wav")))
    if not files:
        print("  [concat] no parts to concatenate")
        return
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    with sf.SoundFile(out_path, "w", samplerate=sr, channels=1, subtype="PCM_16") as f:
        for i, fp in enumerate(files):
            data, _ = sf.read(fp, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            f.write(data)
            if i != len(files) - 1:
                f.write(gap)
    print(f"  [concat] {len(files)} parts -> {out_path}")


def synth_over_chunks(chunks, parts_dir, sr, synth_fn):
    """Call synth_fn(text)->float32 mono ndarray for each chunk, skipping done parts."""
    os.makedirs(parts_dir, exist_ok=True)
    for i, ch in enumerate(chunks):
        part = os.path.join(parts_dir, f"chunk_{i:05d}.wav")
        if os.path.exists(part):
            continue
        try:
            wav = synth_fn(ch)
        except Exception as e:
            print(f"  [chunk {i}] ERROR: {e}")
            wav = np.zeros(int(sr * 0.2), dtype=np.float32)
        if wav is None or len(wav) == 0:
            wav = np.zeros(int(sr * 0.2), dtype=np.float32)
        sf.write(part, wav, sr, subtype="PCM_16")
        print(f"  [chunk {i+1}/{len(chunks)}] {len(wav)/sr:.1f}s ({len(ch)} chars)", flush=True)


# ------------------------------ MMS-TTS ------------------------------
def run_mms(text, outdir, max_chars=200):
    from transformers import AutoTokenizer, VitsModel
    print("[mms] loading facebook/mms-tts-ben ...", flush=True)
    model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    sr = model.config.sampling_rate

    def synth(t):
        inputs = tok(t, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            wav = model(**inputs).waveform
        return wav.squeeze().detach().cpu().numpy().astype(np.float32)

    chunks = make_chunks(text, max_chars)
    print(f"[mms] {len(chunks)} chunks, sr={sr}", flush=True)
    parts_dir = os.path.join(outdir, "parts_mms")
    synth_over_chunks(chunks, parts_dir, sr, synth)
    concat_parts(parts_dir, os.path.join(outdir, "output_mms.wav"), sr)


# ------------------------------ Orpheus ------------------------------
def _snac_from_tokens(new_tokens, snac_model):
    toks = new_tokens
    soa = (toks == SOA).nonzero()
    if len(soa) > 0:
        toks = toks[soa[-1].item() + 1:]
    toks = toks[toks != EOA]
    n = (toks.shape[0] // 7) * 7
    toks = toks[:n]
    if n == 0:
        return None
    codes = [int(t) - CODE_OFFSET for t in toks]
    l1, l2, l3 = [], [], []
    for i in range(len(codes) // 7):
        c = codes[7 * i:7 * i + 7]
        l1.append(c[0])
        l2.append(c[1] - 4096)
        l3.append(c[2] - 2 * 4096)
        l3.append(c[3] - 3 * 4096)
        l2.append(c[4] - 4 * 4096)
        l3.append(c[5] - 5 * 4096)
        l3.append(c[6] - 6 * 4096)

    def ok(layer):
        return all(0 <= v < 4096 for v in layer)

    if not (ok(l1) and ok(l2) and ok(l3)):
        # drop frames with out-of-range codes
        good1, good2, good3 = [], [], []
        for i in range(len(codes) // 7):
            c = codes[7 * i:7 * i + 7]
            a = [c[0], c[1] - 4096, c[2] - 2 * 4096, c[3] - 3 * 4096,
                 c[4] - 4 * 4096, c[5] - 5 * 4096, c[6] - 6 * 4096]
            if all(0 <= v < 4096 for v in a):
                good1.append(a[0])
                good2.extend([a[1], a[4]])
                good3.extend([a[2], a[3], a[5], a[6]])
        l1, l2, l3 = good1, good2, good3
        if not l1:
            return None
    dev = next(snac_model.parameters()).device
    layers = [torch.tensor(l1, device=dev).unsqueeze(0),
              torch.tensor(l2, device=dev).unsqueeze(0),
              torch.tensor(l3, device=dev).unsqueeze(0)]
    with torch.no_grad():
        audio = snac_model.decode(layers)
    return audio.squeeze().detach().cpu().numpy().astype(np.float32)


def run_orpheus(text, outdir, emotion="normal", max_chars=180, max_new_tokens=1200):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    # prefer a local copy of the weights (./orpheus_model) if present
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orpheus_model")
    model_id = local if os.path.isdir(local) else "ehzawad/orpheus-bangla-emotional-tts"
    print(f"[orpheus] loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE).eval()
    sr = 24000

    def synth(t):
        prompt = f"<{emotion}>{t}</{emotion}>" if emotion else t
        ids = tok(prompt, return_tensors="pt").input_ids
        soh = torch.tensor([[SOH]], dtype=torch.long)
        end = torch.tensor([[EOT, EOH]], dtype=torch.long)
        ids = torch.cat([soh, ids, end], dim=1).to(model.device)
        attn = torch.ones_like(ids)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.6, top_p=0.9, repetition_penalty=1.1,
                eos_token_id=EOA, pad_token_id=128004)
        new = out[0][ids.shape[1]:]
        return _snac_from_tokens(new, snac_model)

    chunks = make_chunks(text, max_chars)
    print(f"[orpheus] {len(chunks)} chunks, emotion={emotion}, sr={sr}", flush=True)
    parts_dir = os.path.join(outdir, "parts_orpheus")
    synth_over_chunks(chunks, parts_dir, sr, synth)
    concat_parts(parts_dir, os.path.join(outdir, "output_orpheus.wav"), sr)


# -------------------------------- main --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output.txt")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--model", choices=["mms", "orpheus", "both"], default="both")
    ap.add_argument("--emotion", default="normal",
                    help="Orpheus emotion tag: normal, happy, sad, angry, whisper, ...")
    ap.add_argument("--limit-chars", type=int, default=None,
                    help="Only synthesize the first N characters (for quick tests)")
    args = ap.parse_args()

    text = load_clean_text(args.input, args.limit_chars)
    print(f"[text] {len(text)} chars after cleaning (device={DEVICE})", flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    if args.model in ("mms", "both"):
        run_mms(text, args.outdir)
    if args.model in ("orpheus", "both"):
        run_orpheus(text, args.outdir, emotion=args.emotion)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
