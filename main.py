"""Command-line entry point for the deepfake face-swap pipeline.

Examples:
    python main.py setup
    python main.py demo --name demo_1
    python main.py demo --name demo_2
    python main.py swap --target assets/videos/demo_1/sample_video.mp4 \\
        --source assets/faces/demo_1/elon_musk.jpeg --output output/demo_1/output.mp4
    python main.py combine --frames-dir output/demo_1/frames/sample_video \\
        --source-video assets/videos/demo_1/sample_video.mp4 --output output/demo_1/combined_video.mp4
    python main.py gif --video output/demo_1/combined_video.mp4 --output output/demo_1/preview.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deepfake_faceswap import (
    combine_frames_with_video,
    download_face_enhancer_model,
    resolve_demo_paths,
    run_face_swap,
    save_video_as_gif,
    setup_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _add_swap_quality_args(parser: argparse.ArgumentParser) -> None:
    """Shared between `swap` and `demo`: length-capping, temporal-stabilization, and
    face-restoration knobs."""
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="Trim the target video to this length before swapping if it's longer "
        "(per-frame swap is too slow on CPU otherwise); pass 0 to disable (default: 15).",
    )
    parser.add_argument(
        "--stabilize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Smooth per-frame alignment landmarks to reduce flicker (cheap, on by default).",
    )
    parser.add_argument(
        "--stabilize-alpha",
        type=float,
        default=0.5,
        help="Landmark smoothing factor in (0, 1]; lower = smoother but more lag (default: 0.5).",
    )
    parser.add_argument(
        "--enhance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore detail in swapped faces via GFPGAN. Needs `pip install -r "
        "requirements-enhance.txt` and `python main.py setup --enhancer` first. Heavy and "
        "slow on CPU (~30-60s/frame observed) — off by default.",
    )
    parser.add_argument(
        "--enhancer-model", default="models/GFPGANv1.4.pth", help="Path to the GFPGAN model (see --enhance)."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face-swap deepfake pipeline (built on insightface).")
    subcommands = parser.add_subparsers(dest="command", required=True)

    setup_parser = subcommands.add_parser("setup", help="Download the inswapper_128 swap model.")
    setup_parser.add_argument(
        "--enhancer",
        action="store_true",
        help="Also download the GFPGAN model needed by --enhance. Requires "
        "`pip install -r requirements-enhance.txt` first.",
    )

    swap_parser = subcommands.add_parser("swap", help="Run the face swap on a target video.")
    swap_parser.add_argument("--target", required=True, help="Path to the target video.")
    swap_parser.add_argument("--source", required=True, help="Path to the source face image.")
    swap_parser.add_argument("--output", required=True, help="Path for the output video.")
    swap_parser.add_argument("--execution-provider", default="cpu", choices=["cpu", "cuda"])
    swap_parser.add_argument("--model", default="models/inswapper_128.onnx", help="Path to the inswapper_128 model.")
    swap_parser.add_argument("--frames-dir", default=None, help="Directory to save swapped frames (for `combine`).")
    _add_swap_quality_args(swap_parser)

    combine_parser = subcommands.add_parser(
        "combine", help="Build a side-by-side comparison video from swapped frames."
    )
    combine_parser.add_argument("--frames-dir", required=True, help="Directory of swapped frame images.")
    combine_parser.add_argument("--source-video", required=True, help="Original source video.")
    combine_parser.add_argument("--output", required=True, help="Path for the combined video.")
    combine_parser.add_argument("--fps", type=int, default=30)

    gif_parser = subcommands.add_parser("gif", help="Render a video as an animated GIF preview.")
    gif_parser.add_argument("--video", required=True, help="Path to the source video (e.g. the combined comparison video).")
    gif_parser.add_argument("--output", required=True, help="Path for the output GIF.")
    gif_parser.add_argument("--fps", type=int, default=10)
    gif_parser.add_argument("--width", type=int, default=480)
    gif_parser.add_argument("--max-seconds", type=float, default=None)

    demo_parser = subcommands.add_parser(
        "demo", help="Run the full pipeline (swap + combine + gif) for a named demo (e.g. demo_1)."
    )
    demo_parser.add_argument("--name", required=True, help="Demo folder name under assets/, e.g. demo_1 or demo_2.")
    demo_parser.add_argument("--execution-provider", default="cpu", choices=["cpu", "cuda"])
    demo_parser.add_argument("--model", default="models/inswapper_128.onnx", help="Path to the inswapper_128 model.")
    demo_parser.add_argument("--fps", type=int, default=30, help="FPS for the side-by-side comparison video.")
    demo_parser.add_argument("--gif-fps", type=int, default=10)
    demo_parser.add_argument("--gif-width", type=int, default=480)
    _add_swap_quality_args(demo_parser)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "setup":
        setup_environment()
        if args.enhancer:
            download_face_enhancer_model()
    elif args.command == "swap":
        run_face_swap(
            target_video_path=args.target,
            source_face_image_path=args.source,
            output_video_path=args.output,
            execution_provider=args.execution_provider,
            model_path=args.model,
            frames_dir=args.frames_dir,
            stabilize=args.stabilize,
            stabilize_alpha=args.stabilize_alpha,
            enhance=args.enhance,
            enhancer_model_path=args.enhancer_model,
            max_seconds=args.max_seconds,
        )
    elif args.command == "combine":
        combine_frames_with_video(
            frames_dir=args.frames_dir,
            source_video_path=args.source_video,
            output_video_path=args.output,
            fps=args.fps,
        )
    elif args.command == "gif":
        save_video_as_gif(
            video_path=args.video,
            gif_path=args.output,
            fps=args.fps,
            width=args.width,
            max_seconds=args.max_seconds,
        )
    elif args.command == "demo":
        paths = resolve_demo_paths(PROJECT_ROOT, args.name)
        run_face_swap(
            target_video_path=paths.target_video,
            source_face_image_path=paths.source_face,
            output_video_path=paths.output_video,
            execution_provider=args.execution_provider,
            model_path=args.model,
            frames_dir=paths.frames_dir,
            stabilize=args.stabilize,
            stabilize_alpha=args.stabilize_alpha,
            enhance=args.enhance,
            enhancer_model_path=args.enhancer_model,
            max_seconds=args.max_seconds,
        )
        combine_frames_with_video(
            frames_dir=paths.frames_dir,
            source_video_path=paths.target_video,
            output_video_path=paths.combined_video,
            fps=args.fps,
        )
        save_video_as_gif(
            video_path=paths.combined_video,
            gif_path=paths.gif,
            fps=args.gif_fps,
            width=args.gif_width,
        )


if __name__ == "__main__":
    main()
