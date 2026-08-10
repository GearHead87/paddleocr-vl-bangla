#!/usr/bin/env python
"""
Bengali TTS from OCR text (output.txt) using edge-tts, processed in PAGE BATCHES
run in PARALLEL, then merged into one final MP3.

Pipeline:
  1. Parse output.txt by "===== Page N =====" markers (markers are NOT spoken;
     they only delimit pages).
  2. Group pages into batches of --batch-size pages (e.g. 10 -> pages 1-10, 11-20, ...).
  3. Process up to --parallel batches concurrently. For each batch: its pages' text
     is cleaned, split into sentence chunks, each chunk synthesized (resumable part
     files), then concatenated into ONE batch MP3 (e.g. pages_0001-0010.mp3).
  4. Merge all batch MP3s in page order into a single --output (default output_edge.mp3).

Resume: finished batch MP3s are skipped; within a batch, finished chunk parts are skipped.

Usage:
  .venv/bin/python tts_edge_batched.py --batch-size 10 --parallel 10
  .venv/bin/python tts_edge_batched.py --pages 1-50 --batch-size 10 --parallel 5
  .venv/bin/python tts_edge_batched.py --merge-only

Voices: bn-BD-NabanitaNeural(F), bn-BD-PradeepNeural(M),
        bn-IN-BashkarNeural(M), bn-IN-TanishaaNeural(F)
"""
import argparse
import asyncio
import glob
import os
import re

import edge_tts

PAGE_RE = re.compile(r"^\s*=====\s*Page\s+(\d+)\s*=====\s*$", re.M)


# --------------------------- text handling ---------------------------
def parse_pages(path):
    """Return dict {page_number: page_text} using the page markers as delimiters."""
    raw = open(path, encoding="utf-8").read()
    ms = list(PAGE_RE.finditer(raw))
    pages = {}
    for i, m in enumerate(ms):
        num = int(m.group(1))
        start = m.end()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(raw)
        pages[num] = raw[start:end].strip()
    return pages


def clean_text(txt):
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt


def split_sentences(text):
    parts = re.split(r"(?<=[।!?\.])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(text, max_chars):
    chunks, cur = [], ""
    for s in split_sentences(text):
        if len(s) > max_chars:
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


def concat_mp3(files, out_path):
    with open(out_path, "wb") as out:
        for fp in files:
            with open(fp, "rb") as f:
                out.write(f.read())


# --------------------------- batch planning ---------------------------
def make_batches(page_nums, batch_size):
    """Group pages into fixed NUMERIC windows (1..batch_size, next window, ...).

    Windows are anchored at page 1 so they align to page-number decades
    (1-10, 11-20, ...) and are robust to missing page numbers (only existing
    pages are included; empty windows are skipped).
    """
    if not page_nums:
        return []
    have = set(page_nums)
    hi = max(page_nums)
    batches = []
    start = 1
    while start <= hi:
        end = start + batch_size - 1
        grp = [n for n in range(start, end + 1) if n in have]
        if grp:
            batches.append((start, min(end, hi), grp))  # (window_start, window_end, [pages])
        start = end + 1
    return batches


def batch_label(first, last):
    return f"pages_{first:04d}-{last:04d}"


# ------------------------------ edge-tts ------------------------------
async def synth_chunk(text, voice, rate, volume, pitch, part, retries=5):
    for attempt in range(1, retries + 1):
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
            tmp = part + ".tmp"
            await comm.save(tmp)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("empty audio")
            os.replace(tmp, part)
            return True
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"      [{os.path.basename(part)}] retry {attempt}/{retries}: {e} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
    return False


async def process_batch(batch, pages, workdir, opts, sem):
    first, last, nums = batch
    label = batch_label(first, last)
    batch_mp3 = os.path.join(workdir, label + ".mp3")
    if os.path.exists(batch_mp3) and os.path.getsize(batch_mp3) > 0:
        print(f"[batch {label}] already done, skipping", flush=True)
        return batch_mp3

    async with sem:
        text = clean_text("\n".join(pages.get(n, "") for n in nums))
        chunks = make_chunks(text, opts.max_chars)
        parts_dir = os.path.join(workdir, label)
        os.makedirs(parts_dir, exist_ok=True)
        print(f"[batch {label}] {len(chunks)} chunks", flush=True)

        part_files = []
        for i, ch in enumerate(chunks):
            part = os.path.join(parts_dir, f"chunk_{i:05d}.mp3")
            part_files.append(part)
            if os.path.exists(part) and os.path.getsize(part) > 0:
                continue
            ok = await synth_chunk(ch, opts.voice, opts.rate, opts.volume, opts.pitch, part,
                                   retries=opts.retries)
            if not ok:
                print(f"[batch {label}] chunk {i+1}/{len(chunks)} FAILED", flush=True)
            if opts.delay:
                await asyncio.sleep(opts.delay)

        have = [p for p in part_files if os.path.exists(p) and os.path.getsize(p) > 0]
        if len(have) != len(part_files):
            print(f"[batch {label}] INCOMPLETE ({len(have)}/{len(part_files)} chunks) "
                  f"- not writing batch mp3 so it retries next run", flush=True)
            return None
        concat_mp3(have, batch_mp3)
        print(f"[batch {label}] -> {batch_mp3}", flush=True)
        return batch_mp3


def merge_all(all_batches, workdir, out_path):
    ordered, missing = [], []
    for first, last, _ in all_batches:
        f = os.path.join(workdir, batch_label(first, last) + ".mp3")
        (ordered if os.path.exists(f) and os.path.getsize(f) > 0 else missing).append(f)
    if missing:
        print(f"[merge] {len(missing)} batch file(s) missing; NOT merging yet:")
        for m in missing[:10]:
            print("   -", os.path.basename(m))
        return False
    concat_mp3(ordered, out_path)
    print(f"[merge] {len(ordered)} batches -> {out_path}")
    return True


# -------------------------------- main --------------------------------
async def run(opts):
    pages = parse_pages(opts.input)
    all_nums = sorted(pages)
    print(f"[text] parsed {len(all_nums)} pages (from {all_nums[0]} to {all_nums[-1]})", flush=True)

    # optional page-range filter
    if opts.pages:
        lo, hi = opts.pages
        sel = [n for n in all_nums if lo <= n <= hi]
    else:
        sel = all_nums

    all_batches = make_batches(all_nums, opts.batch_size)         # for final merge ordering
    sel_batches = [b for b in make_batches(sel, opts.batch_size)] # batches to process now

    os.makedirs(opts.workdir, exist_ok=True)

    if not opts.merge_only:
        print(f"[run] {len(sel_batches)} batch(es) of up to {opts.batch_size} pages, "
              f"parallel={opts.parallel}, voice={opts.voice}", flush=True)
        sem = asyncio.Semaphore(opts.parallel)
        await asyncio.gather(*(process_batch(b, pages, opts.workdir, opts, sem) for b in sel_batches))

    # merge only makes sense over the FULL set of batches (in page order)
    if opts.merge_only or not opts.pages:
        merge_all(all_batches, opts.workdir, opts.output)
    else:
        print("[merge] processed a page subset; run with --merge-only after all "
              "batches are done to build the final file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output.txt")
    ap.add_argument("--workdir", default="edge_batches", help="where batch parts/mp3s go")
    ap.add_argument("--output", default="output_edge.mp3", help="final merged file")
    ap.add_argument("--batch-size", type=int, default=10, help="pages per batch")
    ap.add_argument("--parallel", type=int, default=10, help="batches processed concurrently")
    ap.add_argument("--pages", type=str, default=None, help="limit to a page range, e.g. 1-50")
    ap.add_argument("--merge-only", action="store_true", help="only merge existing batch mp3s")
    ap.add_argument("--voice", default="bn-BD-NabanitaNeural")
    ap.add_argument("--rate", default="+0%")
    ap.add_argument("--volume", default="+0%")
    ap.add_argument("--pitch", default="+0Hz")
    ap.add_argument("--max-chars", type=int, default=1500, help="max characters per chunk")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between chunks in a batch")
    opts = ap.parse_args()

    if opts.pages:
        lo, hi = opts.pages.split("-")
        opts.pages = (int(lo), int(hi))

    asyncio.run(run(opts))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
