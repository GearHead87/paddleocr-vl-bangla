#!/usr/bin/env python
"""
Bengali OCR of a scanned PDF -> plain text, using PaddleOCR-VL v1.6 (transformers engine, GPU).

Renders one page at a time to a temp PNG, runs the VL pipeline, and appends the
recognized text to the output .txt with a checkpoint file, so an interruption at
page N does not lose earlier pages (re-running resumes from the checkpoint).

Usage:
    .venv/bin/python ocr_to_txt.py --input input.pdf --output output.txt \
        [--dpi 150] [--start 0] [--end N]
"""
import argparse
import os
import tempfile

import pymupdf  # PyMuPDF


def render_page_png(page, dpi, path):
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    pix.save(path)


def extract_text(res):
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        return md.get("markdown_texts", "") or ""
    if isinstance(md, str):
        return md
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.pdf")
    ap.add_argument("--output", default="output.txt")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--version", default="v1.6")
    args = ap.parse_args()

    ckpt = args.output + ".progress"
    done = 0
    if os.path.exists(ckpt) and os.path.exists(args.output):
        try:
            done = int(open(ckpt).read().strip())
        except ValueError:
            done = 0
    start = max(args.start, done)

    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL(pipeline_version=args.version, engine="transformers")

    doc = pymupdf.open(args.input)
    n = doc.page_count if args.end is None else min(args.end, doc.page_count)

    mode = "a" if start > 0 else "w"
    out = open(args.output, mode, encoding="utf-8")
    print(f"[ocr] pages {start}..{n-1} (of {doc.page_count}), dpi={args.dpi}, resume_from={start}", flush=True)

    tmp = os.path.join(tempfile.gettempdir(), "ocr_page.png")
    for i in range(start, n):
        render_page_png(doc[i], args.dpi, tmp)
        text = ""
        try:
            for res in pipeline.predict(tmp):
                text += extract_text(res)
        except Exception as e:
            text = f"[[OCR ERROR on page {i+1}: {e}]]"
        out.write(f"\n\n===== Page {i+1} =====\n\n")
        out.write(text.strip() + "\n")
        out.flush()
        with open(ckpt, "w") as c:
            c.write(str(i + 1))
        print(f"[ocr] page {i+1}/{n} done ({len(text)} chars)", flush=True)

    out.close()
    print("[ocr] complete ->", args.output, flush=True)


if __name__ == "__main__":
    main()
