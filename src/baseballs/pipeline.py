import os
import cv2
import json
import pathlib
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from tqdm import tqdm

from src.baseballs.heuristic import HeuristicConfig, detect_candidates_frame
from src.baseballs.detector import SphereDetectorWrapper
from src.sphere_detector.utils import local


DEFAULT_VIDEO_PATH = local / "assets" / "examples" / "baseballs.mp4"
DEFAULT_CACHE_DIR = local / "cache"


def save_cache(
    file_path: Union[str, pathlib.Path],
    data: Union[Dict[int, np.ndarray], List[np.ndarray], np.ndarray]
) -> None:
    """Save bounding box data to a cache file (.npy, .npz, or .json)."""
    path = pathlib.Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        if isinstance(data, dict):
            serializable = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            serializable = [(v.tolist() if isinstance(v, np.ndarray) else v) for v in data]
        else:
            serializable = data.tolist()
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
    elif path.suffix == ".npz":
        if isinstance(data, dict):
            np.savez_compressed(path, **{f"frame_{int(k):06d}": np.asarray(v) for k, v in data.items()})
        else:
            np.savez_compressed(path, data=np.asarray(data, dtype=object))
    else:  # Default .npy
        np.save(path, data, allow_pickle=True)


def load_cache(file_path: Union[str, pathlib.Path]) -> Any:
    """Load bounding box data from a cache file (.npy, .npz, or .json)."""
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {int(k): np.array(v, dtype=np.int32) for k, v in data.items()}
        return [np.array(v, dtype=np.int32) for v in data]
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            if "data" in npz:
                return npz["data"]
            return {int(k.split("_")[1]): npz[k] for k in npz.files}
    else:  # .npy
        loaded = np.load(path, allow_pickle=True)
        if loaded.shape == ():
            return loaded.item()
        return loaded


def process_frame(
    frame: np.ndarray,
    heuristic_config: Optional[HeuristicConfig] = None,
    detector: Optional[SphereDetectorWrapper] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single video frame at full resolution:
      1. Detect initial baseball candidate bounding boxes using heuristic.
      2. Refine bounding boxes with sphere detector on tight full-resolution crops.
      
    Args:
        frame: BGR frame (H, W, 3)
        heuristic_config: Configuration for heuristic detector
        detector: SphereDetectorWrapper instance
        
    Returns:
        Tuple of (initial_boxes, sphere_boxes, confidences)
        where initial_boxes and sphere_boxes are np.ndarray of shape (N, 4) in [y1, x1, y2, x2].
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Heuristic initial candidate boxes
    initial_boxes = detect_candidates_frame(frame, config=heuristic_config)

    if detector is None or len(initial_boxes) == 0:
        return initial_boxes, initial_boxes.copy(), np.zeros(len(initial_boxes), dtype=np.float32)

    # 2. Sphere detector refinement on tight full-resolution crops
    sphere_boxes, confidences = detector.refine_candidate_crops(
        frame_rgb, initial_boxes
    )

    return initial_boxes, sphere_boxes, confidences


def process_video(
    video_path: Union[str, pathlib.Path] = DEFAULT_VIDEO_PATH,
    heuristic_cache_path: Optional[Union[str, pathlib.Path]] = None,
    sphere_cache_path: Optional[Union[str, pathlib.Path]] = None,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
    detector_encoder: str = "vits",
    heuristic_config: Optional[HeuristicConfig] = None,
    show_progress: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Process a video to detect baseballs across frames:
      - Uses heuristic to narrow down candidate regions.
      - Uses sphere detector on tight full-resolution crops to refine bounding boxes.
      - Writes both initial and sphere bounding box caches.
      
    Args:
        video_path: Path to video file
        heuristic_cache_path: Output path for initial bounding box guess cache (.npy/.npz/.json)
        sphere_cache_path: Output path for sphere detector bounding box cache (.npy/.npz/.json)
        max_frames: Maximum number of frames to process
        frame_step: Process every N-th frame
        detector_encoder: Encoder for DepthAnythingV2 ('vits', 'vitb', etc.)
        heuristic_config: Config for heuristic detector
        show_progress: Show progress bar
        
    Returns:
        Tuple of (initial_cache_dict, sphere_cache_dict)
    """
    video_path = pathlib.Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if heuristic_cache_path is None:
        heuristic_cache_path = DEFAULT_CACHE_DIR / f"{video_path.stem}_heuristic_bboxes.npy"
    if sphere_cache_path is None:
        sphere_cache_path = DEFAULT_CACHE_DIR / f"{video_path.stem}_sphere_bboxes.npy"

    heuristic_cache_path = pathlib.Path(heuristic_cache_path)
    sphere_cache_path = pathlib.Path(sphere_cache_path)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames if max_frames is None else min(total_frames, max_frames)

    detector = SphereDetectorWrapper(encoder=detector_encoder)
    if heuristic_config is None:
        heuristic_config = HeuristicConfig()

    initial_cache: Dict[int, np.ndarray] = {}
    sphere_cache: Dict[int, np.ndarray] = {}

    frame_indices = range(0, frames_to_process, frame_step)
    progress = tqdm(frame_indices, desc="Processing baseball frames") if show_progress else frame_indices

    for f_idx in progress:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        init_boxes, sphere_boxes, conf = process_frame(
            frame,
            heuristic_config=heuristic_config,
            detector=detector
        )

        initial_cache[f_idx] = init_boxes
        sphere_cache[f_idx] = sphere_boxes

    cap.release()

    # Write cache files
    save_cache(heuristic_cache_path, initial_cache)
    save_cache(sphere_cache_path, sphere_cache)

    if show_progress:
        print(f"Saved initial heuristic bounding box cache to: {heuristic_cache_path}")
        print(f"Saved sphere detector bounding box cache to: {sphere_cache_path}")

    return initial_cache, sphere_cache
