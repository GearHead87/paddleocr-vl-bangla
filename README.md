# PDF OCR Bangla

Turn a scanned Bengali (Bangla) PDF into searchable plain text, then optionally
convert that text into an audiobook.

The project uses
[PaddleOCR-VL 1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
for OCR and provides three text-to-speech (TTS) workflows:

- `tts_edge.py` — simple online TTS with Microsoft Edge voices
- `tts_edge_batched.py` — faster, parallel Edge TTS for long documents
- `tts_generate.py` — local GPU synthesis with MMS-TTS or Orpheus

## How it works

```text
scanned PDF  ->  ocr_to_txt.py  ->  UTF-8 text  ->  a TTS script  ->  MP3/WAV
```

OCR output is written one page at a time and checkpointed, so an interrupted
run can continue without losing completed pages. The TTS scripts also retain
completed chunks and skip them when re-run.

## Requirements

### OCR and local TTS

- Linux
- Python 3.10 (the PaddleOCR stack supports Python 3.9–3.13)
- An NVIDIA GPU with a compatible CUDA driver
- Enough disk space for model downloads and generated audio

The OCR setup below was tested with an RTX 5080 and a CUDA 12.9-compatible
driver. Other GPUs may require a different PaddlePaddle or PyTorch wheel.

### Edge TTS

Edge TTS does not require a GPU, but it does require an internet connection.

## Installation

Create a virtual environment:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Install the OCR dependencies:

```bash
# PaddlePaddle cu129 build (appropriate for the tested Blackwell setup)
.venv/bin/python -m pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

.venv/bin/python -m pip install -U "paddleocr[doc-parser]"
.venv/bin/python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install "transformers>=5.0.0" pymupdf
```

Models are downloaded automatically on the first run and cached under
`~/.cache/huggingface` and `~/.paddlex`.

For online Edge TTS, install:

```bash
.venv/bin/python -m pip install edge-tts
```

For local MMS/Orpheus TTS, install the additional audio packages:

```bash
.venv/bin/python -m pip install numpy soundfile accelerate snac
```

`soundfile` may also require the system package `libsndfile` (for example,
`sudo apt install libsndfile1` on Ubuntu/Debian).

## 1. Extract Bengali text from a PDF

Place the scanned document in the project directory as `input.pdf`, or pass a
different path with `--input`:

```bash
.venv/bin/python ocr_to_txt.py \
  --input input.pdf \
  --output output.txt \
  --dpi 150
```

OCR options:

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `input.pdf` | Source PDF |
| `--output` | `output.txt` | Destination UTF-8 text file |
| `--dpi` | `150` | Page render resolution; increase for difficult scans |
| `--start` | `0` | First page index to process (zero-based) |
| `--end` | last page | Stop before this zero-based page index |
| `--version` | `v1.6` | PaddleOCR-VL pipeline version |

The text file uses page markers that the TTS scripts recognize and remove:

```text
===== Page 1 =====

<recognized Bengali text>
```

Progress is saved to `<output>.progress` after every page. Run the same command
again to resume. To restart from page 1, remove both the output text and its
`.progress` file.

## 2. Generate speech

### Recommended for long documents: batched Edge TTS

This version groups pages into batches, processes several batches concurrently,
and merges them in page order:

```bash
.venv/bin/python tts_edge_batched.py \
  --input output.txt \
  --batch-size 10 \
  --parallel 5 \
  --output output_edge.mp3
```

Useful commands:

```bash
# Generate only pages 1 through 50
.venv/bin/python tts_edge_batched.py --pages 1-50 --batch-size 10 --parallel 5

# Merge completed batches without synthesizing new audio
.venv/bin/python tts_edge_batched.py --merge-only
```

Important options:

| Option | Default | Description |
| --- | --- | --- |
| `--workdir` | `edge_batches` | Batch audio and resumable chunk directory |
| `--output` | `output_edge.mp3` | Final merged MP3 |
| `--batch-size` | `10` | Pages in each numeric batch |
| `--parallel` | `10` | Batches synthesized concurrently |
| `--pages` | all | Inclusive page range such as `1-50` |
| `--merge-only` | off | Merge already completed batches only |
| `--max-chars` | `1500` | Maximum characters sent per request |
| `--retries` | `5` | Attempts per failed chunk |
| `--delay` | `0` | Delay in seconds between requests in a batch |

A very high `--parallel` value may trigger throttling. Reduce it or add a small
`--delay` if requests repeatedly fail.

### Simple Edge TTS

For a short document or quick test:

```bash
.venv/bin/python tts_edge.py --input output.txt --outdir audio

# Synthesize only the first 400 characters
.venv/bin/python tts_edge.py --limit-chars 400 --outdir audio_test
```

The default voice is `bn-BD-NabanitaNeural`. Other Bengali voices include:

- `bn-BD-PradeepNeural`
- `bn-IN-BashkarNeural`
- `bn-IN-TanishaaNeural`

Voice characteristics can also be adjusted:

```bash
.venv/bin/python tts_edge.py \
  --voice bn-BD-PradeepNeural \
  --rate=-10% \
  --pitch=-5Hz
```

The result is written to `<outdir>/output_edge.mp3`; resumable chunks are kept
in `<outdir>/parts_edge/`.

### Local GPU TTS

`tts_generate.py` supports
[`facebook/mms-tts-ben`](https://huggingface.co/facebook/mms-tts-ben) and
[`ehzawad/orpheus-bangla-emotional-tts`](https://huggingface.co/ehzawad/orpheus-bangla-emotional-tts):

```bash
# Faster local Bengali synthesis
.venv/bin/python tts_generate.py --model mms --input output.txt --outdir audio

# Emotional Orpheus synthesis
.venv/bin/python tts_generate.py \
  --model orpheus \
  --emotion normal \
  --input output.txt \
  --outdir audio

# Run both models
.venv/bin/python tts_generate.py --model both --outdir audio
```

Use `--limit-chars 400` for a quick test before processing a full book.
Depending on the selected model, results are written as `output_mms.wav` or
`output_orpheus.wav`. Orpheus is a much larger autoregressive model and can be
substantially slower than MMS.

## Resume and cache behavior

- OCR resumes from `<output>.progress`.
- Simple Edge TTS resumes from `<outdir>/parts_edge/`.
- Batched Edge TTS resumes from `--workdir`.
- Local TTS resumes from `<outdir>/parts_mms/` or `<outdir>/parts_orpheus/`.

Existing audio chunks are reused based on their filenames. If you change the
input text, voice, emotion, speed, pitch, or chunk size, use a new output/work
directory (or remove the old generated chunks) to avoid mixing settings.

## Project files

| File | Purpose |
| --- | --- |
| `ocr_to_txt.py` | Resumable, page-by-page PDF OCR |
| `tts_edge.py` | Sequential online Bengali TTS |
| `tts_edge_batched.py` | Parallel page-batched online Bengali TTS |
| `tts_generate.py` | Local MMS and Orpheus Bengali TTS |

Large inputs, model weights, OCR output, audio chunks, and final audio should
remain local and are not source files.

## Notes

- OCR quality depends heavily on scan clarity, orientation, and render DPI.
- Review OCR text before generating a long audiobook; synthesis will preserve
  recognition errors.
- Edge TTS relies on an online service, so availability and voices may change.
- A full book can produce many hours of audio and require substantial storage.
