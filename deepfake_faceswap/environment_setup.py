"""One-time environment setup: download the swap model.

Face detection (insightface's `buffalo_l`) and the swap model both run through the
`insightface` package, installed like any other dependency via requirements.txt — no
external repository is cloned. Only the inswapper_128 ONNX weights need fetching, since
they're too large to vendor in the repo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

FACE_SWAP_MODEL_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
FACE_SWAP_MODEL_FILENAME = "inswapper_128.onnx"


def download_face_swap_model(models_dir: Path | str = "models") -> Path:
    """Download the inswapper_128 ONNX model into `models_dir`."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_path = models_path / FACE_SWAP_MODEL_FILENAME
    if not model_path.exists():
        subprocess.run(
            ["curl", "-L", FACE_SWAP_MODEL_URL, "-o", str(model_path)],
            check=True,
        )
    return model_path


def setup_environment(models_dir: Path | str = "models") -> Path:
    """Run the one-time setup: download the swap model."""
    return download_face_swap_model(models_dir)
