"""Optional GFPGAN-based face restoration, applied after the swap.

`inswapper_128` swaps at a fixed 128x128 resolution and upscales back, which leaves the
swapped face visibly softer than the rest of the frame. GFPGAN restores detail in just
the face region afterwards.

This is opt-in (`--enhance`) and heavy: it pulls in PyTorch + GFPGAN (~1GB of
dependencies) and, on CPU, took roughly 30-60s *per frame* in testing on this project's
hardware — practical on a CUDA GPU, not on CPU for anything but a still image or a
handful of frames.

Kept in its own module, importing `torch`/`gfpgan` lazily inside `FaceEnhancer.__init__`,
so the base pipeline (`run_face_swap` without `enhance=True`) never needs these installed.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
GFPGAN_MODEL_FILENAME = "GFPGANv1.4.pth"


def download_face_enhancer_model(models_dir: Path | str = "models") -> Path:
    """Download the GFPGANv1.4 restoration model into `models_dir`."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_path = models_path / GFPGAN_MODEL_FILENAME
    if not model_path.exists():
        subprocess.run(["curl", "-L", GFPGAN_MODEL_URL, "-o", str(model_path)], check=True)
    return model_path


def _patch_torchvision_functional_tensor_shim() -> None:
    """`basicsr` (a `gfpgan` dependency, unmaintained since 2022) imports
    `torchvision.transforms.functional_tensor`, which was removed in torchvision>=0.17.
    Inject a shim module exposing the one function it needs before basicsr imports it,
    rather than pinning an old torchvision.
    """
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401

        return  # older torchvision already has it
    except ModuleNotFoundError:
        pass

    from torchvision.transforms import functional as F

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


class FaceEnhancer:
    """Restores detail in swapped faces via GFPGAN. Construct once, reuse across frames."""

    def __init__(self, model_path: Path | str = "models/GFPGANv1.4.pth"):
        _patch_torchvision_functional_tensor_shim()
        from gfpgan import GFPGANer  # local import: heavy optional dependency

        self._restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

    def enhance(self, frame):
        """Restore the face(s) in `frame` (a BGR ndarray) and return the merged frame."""
        _, _, restored = self._restorer.enhance(
            frame, has_aligned=False, only_center_face=False, paste_back=True
        )
        return restored
