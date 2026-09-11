# Deepfake Face Swap

**Swap a source face onto a target video and get back both the swapped
result and a side-by-side comparison against the original — as a CLI, a
reusable Python package, or a notebook that runs the same pipeline locally
or on Google Colab.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-headless-5C3EE8?logo=opencv&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white">
  <img alt="Google Colab" src="https://img.shields.io/badge/Google%20Colab-compatible-F9AB00?logo=googlecolab&logoColor=white">
  <img alt="insightface" src="https://img.shields.io/badge/face%20swap-insightface-2E2E2E">
</p>

Face detection and swapping run directly through
[insightface](https://github.com/deepinsight/insightface)'s own `FaceAnalysis`
and `inswapper_128` model — no external CLI, no subprocess. This project's
own code is the glue around it — model setup, a typed pipeline, and a
comparison-video renderer — organized as an installable package instead of a
single Colab-only notebook.

## Contents

- [Preview](#preview)
- [Technologies](#technologies)
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Notes](#notes)

## Preview

<table>
  <tr>
    <td align="center" width="35%">
      <img src="assets/faces/elon_musk.jpeg" alt="Sample source face" width="100%"><br>
      <sub>Sample source face — <code>assets/faces/elon_musk.jpeg</code></sub>
    </td>
    <td align="center" width="65%">
      <img src="assets/preview.gif" alt="Side-by-side face swap preview" width="100%"><br>
      <sub>Original vs. swapped, side by side — <code>assets/preview.gif</code>
      (rendered from <code>output/combined_video.mp4</code>)</sub>
    </td>
  </tr>
</table>

Swap the two together (`python main.py swap`) to produce
`output/output.mp4`; `python main.py combine` then renders the original and
swapped footage side by side into `output/combined_video.mp4`, and
`python main.py gif` turns that comparison video into a lightweight,
inline-renderable GIF like the one above.

## Technologies

| Layer | Tool |
|---|---|
| Language / runtime | Python 3.9+ |
| Face-swap engine | [insightface](https://github.com/deepinsight/insightface) (`FaceAnalysis` + `inswapper_128`, called directly, in-process) |
| Video / image processing | OpenCV (`opencv-python`) |
| Notebook preview | IPython + Pillow (inline frame preview, Colab- and Jupyter-compatible) |
| CLI | `argparse` (`main.py setup` / `swap` / `combine` / `gif`) |
| Interactive use | Jupyter / Google Colab (`notebooks/deepfake_faceswap_demo.ipynb`) |

## How it works

The pipeline is three independent, composable steps:

1. **Environment setup** (`environment_setup.py`) — downloads the
   `inswapper_128` ONNX model into `models/`. Idempotent: safe to call again
   once the model is already present.
2. **Face swap** (`face_swap.py`) — runs insightface's `FaceAnalysis`
   directly, in-process, to detect a face in the source image and in every
   frame of the target video, then swaps it in with the `inswapper_128`
   model. GPU (`--execution-provider cuda`) or CPU. No external CLI or
   subprocess involved.
3. **Video compositing** (`video_compositor.py`) — reads the per-frame
   swapped images and the original video, and writes a new video with the
   two placed side by side for a visual, frame-aligned comparison. The same
   module can also render any video as a downsampled, resized animated GIF
   — handy for a README preview that plays inline on GitHub.

The CLI (`main.py`) and the notebook
(`notebooks/deepfake_faceswap_demo.ipynb`) are both thin wrappers over
these three steps — pick whichever fits how you want to run it.

## Getting started

### Requirements

- Python 3.9+
- `curl` (to download the swap model) — preinstalled on macOS and most Linux distros

### 1. Install this project's dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. One-time setup: download the swap model

```bash
python main.py setup   # downloads inswapper_128.onnx (~550MB) into models/
```

### 3. Run the face swap

Pick the execution provider for your hardware: `cuda` needs an NVIDIA GPU;
use `cpu` otherwise (e.g. on macOS, or any machine without CUDA) — it works
everywhere but is noticeably slower.

```bash
python main.py swap \
    --target assets/videos/sample_video.mp4 \
    --source assets/faces/elon_musk.jpeg \
    --output output/output.mp4 \
    --execution-provider cpu
```

### 4. Optional: render a side-by-side comparison video

```bash
python main.py combine \
    --frames-dir output/frames/sample_video \
    --source-video assets/videos/sample_video.mp4 \
    --output output/combined_video.mp4
```

### 5. Optional: render that comparison video as a GIF

```bash
python main.py gif \
    --video output/combined_video.mp4 \
    --output output/preview.gif \
    --fps 10 --width 480
```

### Or: run it as a notebook

```bash
jupyter notebook notebooks/deepfake_faceswap_demo.ipynb
```

It runs the same four steps and works both locally and on Google Colab (it
auto-detects Colab and offers a Google Drive mount step for Drive-hosted
assets). Its swap cell defaults to `execution_provider="cuda"` — change it
to `"cpu"` if you're running without a GPU.

## Notes

- **`--execution-provider cuda` requires a CUDA-capable GPU** — pass
  `cpu` instead when running without one (slower, but works anywhere).
- **`models/` (the downloaded swap model) is gitignored, not vendored** — a
  fresh clone of this repo needs `python main.py setup` (or the notebook's
  setup cell) run once before `swap`/`combine` will work.
- **Licensed under [MIT](LICENSE)**.
