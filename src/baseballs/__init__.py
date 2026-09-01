"""Baseball detection and sphere refinement package."""

from .heuristic import HeuristicConfig, detect_candidates_frame
from .detector import SphereDetectorWrapper
from .pipeline import process_video, process_frame, save_cache, load_cache

__all__ = [
    "HeuristicConfig",
    "detect_candidates_frame",
    "SphereDetectorWrapper",
    "process_video",
    "process_frame",
    "save_cache",
    "load_cache",
]
