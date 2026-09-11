"""Run the face swap directly via insightface's own API (FaceAnalysis + inswapper_128).

No external CLI or subprocess is involved for the swap itself: `s0md3v/roop`, which this
module used to shell out to, has been archived and its repository content replaced, so its
`run.py` CLI no longer exists upstream. The actual swap only ever needed insightface's face
detector and the inswapper_128 ONNX model, both of which are used here in-process.

OpenCV's VideoWriter doesn't carry audio, so frames are first written to a silent temp
video, then the target video's original audio track is muxed back in via ffmpeg (bundled
through `imageio-ffmpeg`, no system install required).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import insightface
from insightface.app import FaceAnalysis

_EXECUTION_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}

# RetinaFace's detector letterboxes the input to a square before resizing to `det_size`,
# so a small source photo (e.g. a ~300x300 headshot) can end up scaled down enough that no
# face is found at the 640x640 default. Retry at progressively smaller det_size values
# before giving up, since those still comfortably fit a single headshot's face.
_SOURCE_FACE_DET_SIZES = ((640, 640), (320, 320), (160, 160))


def _set_det_size(face_analyser: FaceAnalysis, det_size: tuple[int, int]) -> None:
    if face_analyser.det_size == det_size:
        return
    # RetinaFace.prepare() silently no-ops once input_size has been set once (it only
    # warns), so it has to be cleared first to actually pick up a new det_size.
    face_analyser.models["detection"].input_size = None
    face_analyser.prepare(ctx_id=0, det_size=det_size)


def _detect_source_face(face_analyser: FaceAnalysis, source_image, source_path: Path | str):
    original_det_size = face_analyser.det_size
    try:
        for det_size in _SOURCE_FACE_DET_SIZES:
            _set_det_size(face_analyser, det_size)
            faces = face_analyser.get(source_image)
            if faces:
                return faces[0]
        raise ValueError(f"No face detected in source image: {source_path}")
    finally:
        _set_det_size(face_analyser, original_det_size)


class _LandmarkSmoother:
    """Exponential moving average over the detected face's alignment keypoints.

    `inswapper` aligns purely on `face.kps` (5-point landmarks) per frame, independently
    of neighboring frames, so tiny detection jitter between otherwise-similar frames shows
    up as flicker in the swapped output. Smoothing `kps` across frames damps that without
    touching the swap itself.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._smoothed_kps = None

    def smooth(self, face):
        if face.kps is None:
            return face
        if self._smoothed_kps is None:
            self._smoothed_kps = face.kps.astype("float64").copy()
        else:
            self._smoothed_kps = self.alpha * face.kps + (1 - self.alpha) * self._smoothed_kps
        face.kps = self._smoothed_kps
        return face


def _cap_video_length(video_path: Path, max_seconds: float | None, cache_dir: Path) -> Path:
    """Return `video_path` unchanged, or a copy trimmed to `max_seconds` if it's longer.

    Per-frame face detection + swap is expensive enough (multiple seconds/frame on CPU)
    that an untrimmed multi-minute video is impractical; capping keeps a `swap`/`demo` run
    tractable by default. Pass `max_seconds=None` (or <= 0) to disable.
    """
    if not max_seconds or max_seconds <= 0:
        return video_path

    probe = cv2.VideoCapture(str(video_path))
    try:
        fps = probe.get(cv2.CAP_PROP_FPS) or 30
        frame_count = probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    finally:
        probe.release()

    duration = frame_count / fps if fps else 0
    if duration <= max_seconds:
        return video_path

    import imageio_ffmpeg

    cache_dir.mkdir(parents=True, exist_ok=True)
    trimmed_path = cache_dir / f"_{video_path.stem}_capped_{int(max_seconds)}s{video_path.suffix}"
    if not trimmed_path.exists():
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [ffmpeg_exe, "-y", "-i", str(video_path), "-t", str(max_seconds), "-c", "copy", str(trimmed_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg trim failed: {result.stderr}")
    print(f"{video_path} is {duration:.1f}s, longer than max_seconds={max_seconds}; using trimmed copy {trimmed_path}")
    return trimmed_path


def _mux_audio(silent_video_path: Path, audio_source_path: Path, output_path: Path) -> None:
    """Copy the video stream from `silent_video_path` and the audio from `audio_source_path`.

    Uses an optional audio map (`1:a:0?`) so this still succeeds, producing a silent
    output, if `audio_source_path` has no audio track.
    """
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(silent_video_path),
        "-i", str(audio_source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed: {result.stderr}")


def run_face_swap(
    target_video_path: Path | str,
    source_face_image_path: Path | str,
    output_video_path: Path | str,
    execution_provider: str = "cpu",
    model_path: Path | str = "models/inswapper_128.onnx",
    frames_dir: Path | str | None = None,
    stabilize: bool = True,
    stabilize_alpha: float = 0.5,
    enhance: bool = False,
    enhancer_model_path: Path | str = "models/GFPGANv1.4.pth",
    max_seconds: float | None = 15.0,
) -> Path:
    """Swap the face from `source_face_image_path` onto every frame of `target_video_path`.

    Also writes each swapped frame as a PNG into `frames_dir` (defaults to a directory
    named after the target video, alongside the output video) so `combine_frames_with_video`
    can build a side-by-side comparison afterwards.

    `stabilize` smooths the per-frame alignment landmarks (cheap, on by default) to reduce
    flicker; lower `stabilize_alpha` smooths more but adds lag. `enhance` runs each swapped
    frame through GFPGAN afterwards to restore detail inswapper_128 loses (off by default —
    heavy, and ~30-60s/frame on CPU; see `face_enhancement.py`). `max_seconds` (default 15)
    trims a longer `target_video_path` down first, since per-frame swapping is too slow on
    CPU to be practical on a multi-minute video; pass `None` to disable.
    """
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_target_path = Path(target_video_path)
    target_video_path = _cap_video_length(original_target_path, max_seconds, output_path.parent)

    providers = [_EXECUTION_PROVIDERS.get(execution_provider, "CPUExecutionProvider")]

    face_analyser = FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)

    face_enhancer = None
    if enhance:
        from .face_enhancement import FaceEnhancer  # local import: heavy optional dependency

        face_enhancer = FaceEnhancer(enhancer_model_path)

    source_image = cv2.imread(str(source_face_image_path))
    source_face = _detect_source_face(face_analyser, source_image, source_face_image_path)

    frames_path = Path(frames_dir) if frames_dir else output_path.parent / "frames" / original_target_path.stem
    frames_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(target_video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    silent_output_path = output_path.with_name(f"{output_path.stem}_silent{output_path.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(silent_output_path), fourcc, fps, (frame_width, frame_height))

    landmark_smoother = _LandmarkSmoother(stabilize_alpha) if stabilize else None

    frame_index = 0
    try:
        while True:
            frame_read, frame = capture.read()
            if not frame_read:
                break

            target_faces = face_analyser.get(frame)
            if target_faces:
                target_face = target_faces[0]
                if landmark_smoother is not None:
                    target_face = landmark_smoother.smooth(target_face)
                frame = swapper.get(frame, target_face, source_face, paste_back=True)
                if face_enhancer is not None:
                    frame = face_enhancer.enhance(frame)

            cv2.imwrite(str(frames_path / f"{frame_index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            video_writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        video_writer.release()

    try:
        _mux_audio(silent_output_path, target_video_path, output_path)
    finally:
        silent_output_path.unlink(missing_ok=True)

    print(f"Swapped video saved to {output_path}")
    print(f"Swapped frames saved to {frames_path}")
    return output_path
