"""Face-swap deepfake pipeline built on top of insightface."""

from .environment_setup import download_face_swap_model, setup_environment
from .face_swap import run_face_swap
from .video_compositor import combine_frames_with_video, save_video_as_gif

__all__ = [
    "download_face_swap_model",
    "setup_environment",
    "run_face_swap",
    "combine_frames_with_video",
    "save_video_as_gif",
]
