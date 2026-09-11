"""Resolve the standard asset/output layout for a named demo (e.g. "demo_1", "demo_2").

Each demo is a matched pair of folders: `assets/videos/<demo>/` holding exactly one
target video, and `assets/faces/<demo>/` holding exactly one source face image. Outputs
for that demo are written under `output/<demo>/`. This is the single path-resolution
function shared by the CLI's `demo` subcommand and the notebook, so both demo_1 and
demo_2 run through the same code path instead of each hardcoding its own filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class DemoPaths:
    target_video: Path
    source_face: Path
    output_video: Path
    frames_dir: Path
    combined_video: Path
    gif: Path


def _find_one(directory: Path, extensions: tuple[str, ...], kind: str) -> Path:
    matches = sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)
    if not matches:
        raise FileNotFoundError(f"No {kind} file found in {directory}")
    if len(matches) > 1:
        names = ", ".join(match.name for match in matches)
        raise ValueError(f"Expected exactly one {kind} file in {directory}, found {len(matches)}: {names}")
    return matches[0]


def resolve_demo_paths(project_root: Path | str, demo_name: str) -> DemoPaths:
    """Resolve target video, source face, and output paths for `demo_name`.

    Expects `assets/videos/<demo_name>/` and `assets/faces/<demo_name>/` to each hold
    exactly one video/image file respectively.
    """
    project_root = Path(project_root)
    video_dir = project_root / "assets" / "videos" / demo_name
    face_dir = project_root / "assets" / "faces" / demo_name
    output_dir = project_root / "output" / demo_name

    if not video_dir.is_dir():
        raise FileNotFoundError(f"No such demo video directory: {video_dir}")
    if not face_dir.is_dir():
        raise FileNotFoundError(f"No such demo face directory: {face_dir}")

    target_video = _find_one(video_dir, VIDEO_EXTENSIONS, "video")
    source_face = _find_one(face_dir, IMAGE_EXTENSIONS, "face image")

    return DemoPaths(
        target_video=target_video,
        source_face=source_face,
        output_video=output_dir / "output.mp4",
        frames_dir=output_dir / "frames" / target_video.stem,
        combined_video=output_dir / "combined_video.mp4",
        gif=output_dir / "preview.gif",
    )
