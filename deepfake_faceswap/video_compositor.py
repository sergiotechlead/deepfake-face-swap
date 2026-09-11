"""Combine the swapped per-frame images side-by-side with the source video."""

from __future__ import annotations

from pathlib import Path

import cv2

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _preview_frame(frame) -> None:
    """Best-effort inline preview: works in Jupyter/Colab, silently does nothing elsewhere."""
    try:
        from IPython.display import display
        from PIL import Image

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display(Image.fromarray(rgb_frame))
    except ImportError:
        pass


def combine_frames_with_video(
    frames_dir: Path | str,
    source_video_path: Path | str,
    output_video_path: Path | str,
    fps: int = 30,
    preview_every_n_frames: int | None = 50,
) -> Path:
    """Write a video that places each swapped frame next to the matching source-video frame.

    `frames_dir` holds the swapped frames (e.g. `run_face_swap`'s output directory),
    `source_video_path` is the original input video shown side-by-side for comparison.
    """
    frames_path = Path(frames_dir)
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_filenames = sorted(
        name for name in frames_path.iterdir() if name.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frame_filenames:
        raise FileNotFoundError(f"No frame images found in {frames_path}")

    first_frame = cv2.imread(str(frame_filenames[0]))
    frame_height, frame_width, _ = first_frame.shape

    source_capture = cv2.VideoCapture(str(source_video_path))
    source_height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    common_height = min(frame_height, source_height)
    common_width = frame_width + source_width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (common_width, common_height))

    try:
        for index, frame_filename in enumerate(frame_filenames):
            swapped_frame = cv2.imread(str(frame_filename))
            swapped_frame = cv2.resize(swapped_frame, (frame_width, common_height))

            frame_read, source_frame = source_capture.read()
            if not frame_read:
                break

            resized_source_width = int(source_width * common_height / source_height)
            source_frame = cv2.resize(source_frame, (resized_source_width, common_height))

            combined_frame = cv2.hconcat([source_frame, swapped_frame])

            if preview_every_n_frames and index % preview_every_n_frames == 0:
                preview = cv2.resize(combined_frame, (500, 444))
                _preview_frame(preview)

            video_writer.write(combined_frame)
    finally:
        source_capture.release()
        video_writer.release()

    print(f"Combined video saved to {output_path}")
    return output_path


def save_video_as_gif(
    video_path: Path | str,
    gif_path: Path | str,
    fps: int = 10,
    width: int = 480,
    max_seconds: float | None = None,
) -> Path:
    """Render a video (e.g. the combined comparison video) as an animated GIF.

    Useful for embedding a lightweight, inline-renderable preview in the README,
    where the source video itself is a binary GitHub can't play inline.
    """
    from PIL import Image

    video_path = Path(video_path)
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
    frame_interval = max(round(source_fps / fps), 1)
    max_frames = int(max_seconds * source_fps) if max_seconds else None

    frames = []
    frame_index = 0
    try:
        while True:
            frame_read, frame = capture.read()
            if not frame_read or (max_frames is not None and frame_index >= max_frames):
                break
            if frame_index % frame_interval == 0:
                frame_height, frame_width = frame.shape[:2]
                if frame_width != width:
                    scale = width / frame_width
                    frame = cv2.resize(frame, (width, int(frame_height * scale)))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )

    print(f"GIF preview saved to {gif_path}")
    return gif_path
