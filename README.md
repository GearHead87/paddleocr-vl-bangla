# pdfocrbangla

Extract selectable text from a **scanned Bengali (Bangla) PDF** using
[**PaddleOCR-VL-1.6**](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) — a
0.9B vision-language OCR model with genuine Bengali support. Runs locally on an
NVIDIA GPU and writes the recognized text to a plain `.txt` file, one section
per page.

## Why PaddleOCR-VL

The source PDF is image-only (no text layer), so its Bengali text can't be
selected or searched. Classic engines fall short for Bengali:

- **Tesseract `ben`** — works, but weaker on compound characters (যুক্তাক্ষর).
- **PaddleOCR (stock)** — no Bengali model shipped.
- **bbOCR / chitrolipi** — deprecated and needs CUDA 11.6 (incompatible with
  modern Blackwell GPUs).

PaddleOCR-VL-1.6 gives clean, coherent Bengali output and has an official path
for NVIDIA Blackwell (RTX 50-series) GPUs.

## Requirements

- Linux, NVIDIA GPU (tested on an RTX 5080, Compute Capability 12.0)
- NVIDIA driver supporting CUDA ≥ 12.9
- Python 3.10 (3.9–3.13 supported)

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# GPU PaddlePaddle (layout model) — cu129 build for Blackwell
.venv/bin/python -m pip install paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# PaddleOCR pipeline + document parser
.venv/bin/python -m pip install -U "paddleocr[doc-parser]"

# Transformers backend (VLM) — torch cu128 supports Blackwell (sm_120)
.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install "transformers>=5.0.0" pymupdf
```

Models download automatically on first run (~1.8 GB VL weights + ~128 MB layout
model), cached under `~/.cache/huggingface` and `~/.paddlex`.

## Usage

```bash
.venv/bin/python ocr_to_txt.py --input input.pdf --output output.txt --dpi 150
```

Options:

| Flag        | Default     | Description                                  |
|-------------|-------------|----------------------------------------------|
| `--input`   | `input.pdf` | Source PDF                                   |
| `--output`  | `output.txt`| Destination text file                        |
| `--dpi`     | `150`       | Page render resolution (raise for hard scans)|
| `--start`   | `0`         | First page index (0-based) to process        |
| `--end`     | last page   | Stop before this page index                  |
| `--version` | `v1.6`      | PaddleOCR-VL pipeline version                 |

### Resume

Progress is checkpointed to `<output>.progress` after every page. If the run is
interrupted, re-run the **same command** and it continues from the last
completed page. To start fresh, delete `output.txt` and `output.txt.progress`.

## Output format

```
===== Page 1 =====

<recognized Bengali text for page 1>

===== Page 2 =====

...
```

## Performance

~2–4 s/page on an RTX 5080 (≈30 min for 554 pages). Model load adds ~10–15 s at
startup.

## Files

- `ocr_to_txt.py` — the OCR script (page-by-page, resumable)
- `input.pdf` — source scanned PDF
- `output.txt` — generated text (git-ignored by default)
