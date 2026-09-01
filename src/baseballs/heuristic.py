import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class HeuristicConfig:
    """Configuration for baseball candidate heuristic detector."""
    # Sky region boundaries (excludes top caption text overlay)
    sky_y_min: int = 250
    sky_y_max: int = 620
    sky_min_intensity: int = 45
    sky_max_border_mean: float = 65.0
    sky_min_contrast: float = 18.0
    sky_max_ball_size: int = 30
    sky_max_area: float = 400.0

    # Ground / Tee region boundaries
    field_y_min: int = 520
    field_y_max: int = 1150
    # Yellow / green softball HSV bounds
    yellow_hsv_min: Tuple[int, int, int] = (18, 85, 85)
    yellow_hsv_max: Tuple[int, int, int] = (42, 255, 255)
    field_min_area: float = 35.0
    field_max_area: float = 2000.0
    field_min_circularity: float = 0.45

    # Shape bounds
    border_margin: int = 6


def detect_candidates_frame(
    frame: np.ndarray,
    config: Optional[HeuristicConfig] = None
) -> np.ndarray:
    """
    Detect candidate baseball bounding boxes in a single frame using specialized CV heuristics.
    
    Args:
        frame: BGR image (H, W, 3)
        config: HeuristicConfig parameters
        
    Returns:
        np.ndarray of shape (N, 4) with bounding boxes in [y_min, x_min, y_max, x_max] format.
    """
    if config is None:
        config = HeuristicConfig()

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 and frame.shape[2] == 3 else frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if frame.ndim == 3 and frame.shape[2] == 3 else None

    candidates = []

    # -------------------------------------------------------------
    # 1. Baseballs in the Night Sky (bright objects against dark sky)
    # -------------------------------------------------------------
    sky_y1 = max(0, config.sky_y_min)
    sky_y2 = min(h, config.sky_y_max)
    if sky_y2 > sky_y1:
        sky_gray = gray[sky_y1:sky_y2, :]
        _, thresh_sky = cv2.threshold(sky_gray, config.sky_min_intensity, 255, cv2.THRESH_BINARY)
        contours_sky, _ = cv2.findContours(thresh_sky, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours_sky:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)

            if (
                bw > config.sky_max_ball_size or
                bh > config.sky_max_ball_size or
                area > config.sky_max_area or
                bx < config.border_margin or
                bx + bw > w - config.border_margin
            ):
                continue

            # 4-quadrant surrounding night sky contrast check:
            # A true baseball against night sky must be surrounded by dark sky on all sides.
            pad = max(6, int(max(bw, bh) * 0.8))
            t_slice = sky_gray[max(0, by - pad):by, max(0, bx - pad):min(sky_gray.shape[1], bx + bw + pad)]
            b_slice = sky_gray[by + bh:min(sky_gray.shape[0], by + bh + pad), max(0, bx - pad):min(sky_gray.shape[1], bx + bw + pad)]
            l_slice = sky_gray[by:by + bh, max(0, bx - pad):bx]
            r_slice = sky_gray[by:by + bh, bx + bw:min(sky_gray.shape[1], bx + bw + pad)]

            borders = [t_slice, b_slice, l_slice, r_slice]
            border_means = [float(s.mean()) for s in borders if s.size > 0]

            if border_means and max(border_means) < config.sky_max_border_mean:
                inner = sky_gray[by:by + bh, bx:bx + bw]
                if inner.mean() - min(border_means) >= config.sky_min_contrast:
                    candidates.append([by + sky_y1, bx, by + bh + sky_y1, bx + bw])

    # -------------------------------------------------------------
    # 2. Softballs / Baseballs on Tee and Ground (color & circularity)
    # -------------------------------------------------------------
    if hsv is not None:
        field_y1 = max(0, config.field_y_min)
        field_y2 = min(h, config.field_y_max)
        if field_y2 > field_y1:
            field_hsv = hsv[field_y1:field_y2, :]
            lower_yg = np.array(config.yellow_hsv_min, dtype=np.uint8)
            upper_yg = np.array(config.yellow_hsv_max, dtype=np.uint8)
            mask_yg = cv2.inRange(field_hsv, lower_yg, upper_yg)

            kernel_yg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned_yg = cv2.morphologyEx(mask_yg, cv2.MORPH_OPEN, kernel_yg)
            contours_yg, _ = cv2.findContours(cleaned_yg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours_yg:
                area = cv2.contourArea(c)
                if not (config.field_min_area <= area <= config.field_max_area):
                    continue

                perimeter = cv2.arcLength(c, True)
                if perimeter <= 0:
                    continue
                circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
                if circularity < config.field_min_circularity:
                    continue

                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 0
                if not (0.6 <= aspect <= 1.6):
                    continue

                candidates.append([by + field_y1, bx, by + bh + field_y1, bx + bw])

    if not candidates:
        return np.empty((0, 4), dtype=np.int32)

    # Sort candidates spatially (top-to-bottom, left-to-right)
    candidates.sort(key=lambda b: (b[0] // 30, b[1]))
    return np.array(candidates, dtype=np.int32)
