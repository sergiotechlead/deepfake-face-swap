"""Command-line entry point for the deepfake face-swap pipeline.

Examples:
    python main.py setup
    python main.py swap --target assets/videos/sample_video.mp4 \\
        --source assets/faces/elon_musk.jpeg --output output/output.mp4
    python main.py combine --frames-dir output/frames/sample_video \\
        --source-video assets/videos/sample_video.mp4 --output output/combined_video.mp4
    python main.py gif --video output/combined_video.mp4 --output output/preview.gif
"""

from __future__ import annotations

import argparse

from deepfake_faceswap import (
    combine_frames_with_video,
    run_face_swap,
    save_video_as_gif,
    setup_environment,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face-swap deepfake pipeline (built on insightface).")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("setup", help="Download the inswapper_128 swap model.")

    swap_parser = subcommands.add_parser("swap", help="Run the face swap on a target video.")
    swap_parser.add_argument("--target", required=True, help="Path to the target video.")
    swap_parser.add_argument("--source", required=True, help="Path to the source face image.")
    swap_parser.add_argument("--output", required=True, help="Path for the output video.")
    swap_parser.add_argument("--execution-provider", default="cpu", choices=["cpu", "cuda"])
    swap_parser.add_argument("--model", default="models/inswapper_128.onnx", help="Path to the inswapper_128 model.")
    swap_parser.add_argument("--frames-dir", default=None, help="Directory to save swapped frames (for `combine`).")

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

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "setup":
        setup_environment()
    elif args.command == "swap":
        run_face_swap(
            target_video_path=args.target,
            source_face_image_path=args.source,
            output_video_path=args.output,
            execution_provider=args.execution_provider,
            model_path=args.model,
            frames_dir=args.frames_dir,
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


if __name__ == "__main__":
    main()
