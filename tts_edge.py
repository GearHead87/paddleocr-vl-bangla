#!/usr/bin/env python
"""
Generate Bengali speech audio from the OCR text (output.txt) using edge-tts
(Microsoft Edge's free online TTS -> MP3). No GPU / no model download; needs internet.

Same scaffolding as tts_generate.py: page separators ("===== Page N =====") are
stripped at runtime, text is split into sentence-sized chunks, each chunk is
synthesized to a part file (resumable), then all parts are concatenated into a
single output_edge.mp3.

Usage:
    .venv/bin/python tts_edge.py                                   # full book, default voice
    .venv/bin/python tts_edge.py --limit-chars 400 --outdir tts_test   # quick test
    .venv/bin/python tts_edge.py --voice bn-BD-PradeepNeural --rate=-10%

Bengali voices: bn-BD-NabanitaNeural (F), bn-BD-PradeepNeural (M),
                bn-IN-BashkarNeural (M), bn-IN-TanishaaNeural (F)
"""
import argparse
import asyncio
import glob
import os
import re

import edge_tts

PAGE_RE = re.compile(r"^\s*=====\s*Page\s+\d+\s*=====\s*$", re.M)


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


def concat_parts(parts_dir, out_path):
    """Concatenate per-chunk MP3 part files into one MP3 (frames concatenate cleanly)."""
    files = sorted(glob.glob(os.path.join(parts_dir, "chunk_*.mp3")))
    if not files:
        print("  [concat] no parts to concatenate")
        return
    with open(out_path, "wb") as out:
        for fp in files:
            with open(fp, "rb") as f:
                out.write(f.read())
    print(f"  [concat] {len(files)} parts -> {out_path}")


# ------------------------------ edge-tts ------------------------------
async def synth_chunk(text, voice, rate, volume, pitch, part, retries=5):
    for attempt in range(1, retries + 1):
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
            tmp = part + ".tmp"
            await comm.save(tmp)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("empty audio returned")
            os.replace(tmp, part)
            return True
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"    retry {attempt}/{retries} after error: {e} (waiting {wait}s)", flush=True)
            await asyncio.sleep(wait)
    return False


async def run_edge(text, outdir, voice, rate, volume, pitch, max_chars, delay):
    chunks = make_chunks(text, max_chars)
    parts_dir = os.path.join(outdir, "parts_edge")
    os.makedirs(parts_dir, exist_ok=True)
    print(f"[edge] {len(chunks)} chunks, voice={voice}, rate={rate}, volume={volume}, pitch={pitch}", flush=True)

    for i, ch in enumerate(chunks):
        part = os.path.join(parts_dir, f"chunk_{i:05d}.mp3")
        if os.path.exists(part) and os.path.getsize(part) > 0:
            continue
        ok = await synth_chunk(ch, voice, rate, volume, pitch, part)
        if not ok:
            print(f"  [chunk {i+1}/{len(chunks)}] FAILED after retries; leaving gap", flush=True)
        else:
            print(f"  [chunk {i+1}/{len(chunks)}] ok ({len(ch)} chars)", flush=True)
        if delay:
            await asyncio.sleep(delay)

    concat_parts(parts_dir, os.path.join(outdir, "output_edge.mp3"))


# -------------------------------- main --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output.txt")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--voice", default="bn-BD-NabanitaNeural",
                    help="bn-BD-NabanitaNeural(F), bn-BD-PradeepNeural(M), "
                         "bn-IN-BashkarNeural(M), bn-IN-TanishaaNeural(F)")
    ap.add_argument("--rate", default="+0%", help="speech rate, e.g. -10% or +20%")
    ap.add_argument("--volume", default="+0%", help="volume, e.g. -10% or +20%")
    ap.add_argument("--pitch", default="+0Hz", help="pitch, e.g. -5Hz or +10Hz")
    ap.add_argument("--max-chars", type=int, default=1500, help="max characters per chunk")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between chunks")
    ap.add_argument("--limit-chars", type=int, default=None,
                    help="Only synthesize the first N characters (for quick tests)")
    args = ap.parse_args()

    text = load_clean_text(args.input, args.limit_chars)
    print(f"[text] {len(text)} chars after cleaning", flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    asyncio.run(run_edge(text, args.outdir, args.voice, args.rate, args.volume,
                         args.pitch, args.max_chars, args.delay))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
