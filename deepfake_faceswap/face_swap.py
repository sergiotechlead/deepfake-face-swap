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
) -> Path:
    """Swap the face from `source_face_image_path` onto every frame of `target_video_path`.

    Also writes each swapped frame as a PNG into `frames_dir` (defaults to a directory
    named after the target video, alongside the output video) so `combine_frames_with_video`
    can build a side-by-side comparison afterwards.
    """
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    providers = [_EXECUTION_PROVIDERS.get(execution_provider, "CPUExecutionProvider")]

    face_analyser = FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)

    source_image = cv2.imread(str(source_face_image_path))
    source_face = _detect_source_face(face_analyser, source_image, source_face_image_path)

    frames_path = Path(frames_dir) if frames_dir else output_path.parent / "frames" / Path(target_video_path).stem
    frames_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(target_video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    silent_output_path = output_path.with_name(f"{output_path.stem}_silent{output_path.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(silent_output_path), fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    try:
        while True:
            frame_read, frame = capture.read()
            if not frame_read:
                break

            target_faces = face_analyser.get(frame)
            if target_faces:
                frame = swapper.get(frame, target_faces[0], source_face, paste_back=True)

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
