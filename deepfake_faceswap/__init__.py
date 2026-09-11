"""Face-swap deepfake pipeline built on top of insightface."""

from .demo_paths import DemoPaths, resolve_demo_paths
from .environment_setup import download_face_swap_model, setup_environment
from .face_enhancement import FaceEnhancer, download_face_enhancer_model
from .face_swap import run_face_swap
from .video_compositor import combine_frames_with_video, save_video_as_gif

__all__ = [
    "DemoPaths",
    "resolve_demo_paths",
    "download_face_swap_model",
    "setup_environment",
    "FaceEnhancer",
    "download_face_enhancer_model",
    "run_face_swap",
    "combine_frames_with_video",
    "save_video_as_gif",
]
