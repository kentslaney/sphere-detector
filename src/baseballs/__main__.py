import argparse
import pathlib
import cv2
import numpy as np

from src.baseballs.pipeline import (
    process_video,
    save_cache,
    load_cache,
    DEFAULT_VIDEO_PATH,
    DEFAULT_CACHE_DIR,
)
from src.baseballs.detector import SphereDetectorWrapper
from src.baseballs.heuristic import HeuristicConfig


def main():
    parser = argparse.ArgumentParser(
        description="Detect baseballs in video using heuristic localization and full-resolution tight-crop sphere detector refinement."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help=f"Path to input video file (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--heuristic-cache",
        type=str,
        default=None,
        help="Path for initial heuristic bounding box cache file (.npy/.npz/.json)",
    )
    parser.add_argument(
        "--sphere-cache",
        type=str,
        default=None,
        help="Path for sphere detector bounding box cache file (.npy/.npz/.json)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl"],
        help="DepthAnything encoder model variant (default: vits)",
    )
    parser.add_argument(
        "--visualize-output",
        type=str,
        default=None,
        help="Directory to save separated and zoomed visualization frames",
    )

    args = parser.parse_args()

    print(f"Processing baseballs from video: {args.video}")
    initial_cache, sphere_cache = process_video(
        video_path=args.video,
        heuristic_cache_path=args.heuristic_cache,
        sphere_cache_path=args.sphere_cache,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        detector_encoder=args.encoder,
        show_progress=True,
    )

    print(f"Completed detection for {len(initial_cache)} frames.")

    if args.visualize_output:
        out_dir = pathlib.Path(args.visualize_output)
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(args.video)
        detector = SphereDetectorWrapper(encoder=args.encoder)
        
        sample_indices = list(initial_cache.keys())[:10]
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            init_boxes = initial_cache.get(f_idx, [])
            sphere_boxes = sphere_cache.get(f_idx, [])
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_full = detector.compute_depth(frame_rgb) if len(init_boxes) > 0 else None

            # 1. Panel: Initial Heuristic Only (Green)
            vis_h = frame.copy()
            cv2.putText(vis_h, "Initial Heuristic Guess", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            for i, (y1, x1, y2, x2) in enumerate(init_boxes):
                cv2.rectangle(vis_h, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_h, f"H{i}", (int(x1) - 24, int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 2. Panel: Sphere Detector Refined Only (Yellow/Cyan)
            vis_s = frame.copy()
            cv2.putText(vis_s, "Sphere Detector Refined", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            for i, (y1, x1, y2, x2) in enumerate(sphere_boxes):
                cv2.rectangle(vis_s, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(vis_s, f"S{i}", (int(x1) + int(x2 - x1) + 6, int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 3. Side-by-Side Comparison
            comparison = np.hstack([vis_h, vis_s])
            cv2.imwrite(str(out_dir / f"comparison_frame_{f_idx:05d}.jpg"), comparison)
            cv2.imwrite(str(out_dir / f"heuristic_frame_{f_idx:05d}.jpg"), vis_h)
            cv2.imwrite(str(out_dir / f"sphere_frame_{f_idx:05d}.jpg"), vis_s)

            # 4. Zoomed debug panels for each detected ball
            for i in range(len(init_boxes)):
                debug_strip = detector.generate_debug_panel(
                    frame_rgb, init_boxes[i], depth_full=depth_full, zoom=5
                )
                cv2.imwrite(str(out_dir / f"debug_frame_{f_idx:05d}_ball_{i}.jpg"), debug_strip)

        cap.release()
        print(f"Saved separated, zoomed, and debug visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
