"""Run the face swap directly via insightface's own API (FaceAnalysis + inswapper_128).

No external CLI or subprocess is involved: `s0md3v/roop`, which this module used to shell
out to, has been archived and its repository content replaced, so its `run.py` CLI no
longer exists upstream. The actual swap only ever needed insightface's face detector and
the inswapper_128 ONNX model, both of which are used here in-process.

Note: the output video has no audio track — OpenCV's VideoWriter doesn't carry one.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import insightface
from insightface.app import FaceAnalysis

_EXECUTION_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


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
    source_faces = face_analyser.get(source_image)
    if not source_faces:
        raise ValueError(f"No face detected in source image: {source_face_image_path}")
    source_face = source_faces[0]

    frames_path = Path(frames_dir) if frames_dir else output_path.parent / "frames" / Path(target_video_path).stem
    frames_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(target_video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

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

    print(f"Swapped video saved to {output_path}")
    print(f"Swapped frames saved to {frames_path}")
    return output_path
