import numpy as np
import cv2
from PIL import Image
from typing import Optional, List, Tuple, Union, Dict, Any
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.sphere_detector.detect import Raster, Config
from src.sphere_detector.depth import Da2


class SphereDetectorWrapper:
    """Wrapper around the sphere_detector for full-resolution tight-crop baseball detection."""

    def __init__(
        self,
        encoder: str = "vits",
        min_crop_radius: int = 32,
        crop_scale: float = 1.5,
        subdivisions: int = 4
    ):
        """
        Initialize the sphere detector wrapper.
        
        Args:
            encoder: DepthAnythingV2 encoder ('vits', 'vitb', etc.)
            min_crop_radius: Minimum half-width/height for tight candidate crops
            crop_scale: Multiplier on candidate box size for crop window
            subdivisions: Number of subdivisions in Seives feature pyramid
        """
        self.encoder = encoder
        self.min_crop_radius = min_crop_radius
        self.crop_scale = crop_scale
        self.subdivisions = subdivisions
        self.da2 = Da2(encoder)

    def compute_depth(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Compute the full-resolution inverse depth map using DepthAnythingV2 directly (zero rescaling)."""
        np_im = np.array(image) if isinstance(image, Image.Image) else image
        return np.array(self.da2.infer_direct(np_im))

    def refine_candidate_crops(
        self,
        frame_rgb: np.ndarray,
        candidate_boxes: np.ndarray,
        depth_full: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine heuristic candidate bounding boxes using the sphere detector on tight full-resolution crops.
        
        Args:
            frame_rgb: Full resolution RGB frame (H, W, 3)
            candidate_boxes: Initial bounding boxes (N, 4) in [y_min, x_min, y_max, x_max]
            depth_full: Optional precomputed full-resolution depth map (H, W)
            
        Returns:
            Tuple of (refined_bounding_boxes, confidences)
            where refined_bounding_boxes has shape (N, 4) in [y1, x1, y2, x2] coordinates.
        """
        if len(candidate_boxes) == 0:
            return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.float32)

        if depth_full is None:
            depth_full = self.compute_depth(frame_rgb)

        h, w = frame_rgb.shape[:2]
        refined_boxes = []
        confidences = []

        for y1, x1, y2, x2 in candidate_boxes:
            bw = x2 - x1
            bh = y2 - y1
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2

            radius = max(self.min_crop_radius, int(max(bw, bh) * self.crop_scale))
            cy1 = max(0, cy - radius)
            cx1 = max(0, cx - radius)
            cy2 = min(h, cy + radius)
            cx2 = min(w, cx + radius)

            crop_rgb = frame_rgb[cy1:cy2, cx1:cx2]
            crop_depth = depth_full[cy1:cy2, cx1:cx2]
            crop_h, crop_w = crop_rgb.shape[:2]

            try:
                pil_crop = Image.fromarray(crop_rgb)
                r = Raster(
                    pil_crop,
                    cache=jnp.array(crop_depth),
                    resolution=(crop_h, crop_w),
                    candidates=1,
                    subdivisions=self.subdivisions,
                    extent=2
                )
                conf, bounds = r.opt(1).predict()
                conf_val = float(conf[0]) if not np.isnan(conf[0]) else 0.0

                if not np.any(np.isnan(bounds)):
                    b = bounds[0]
                    gy1 = int(np.clip(b[0] + cy1, 0, h))
                    gx1 = int(np.clip(b[1] + cx1, 0, w))
                    gy2 = int(np.clip(b[2] + cy1, 0, h))
                    gx2 = int(np.clip(b[3] + cx1, 0, w))

                    # Ensure refined box is valid and non-collapsed
                    if gy2 > gy1 and gx2 > gx1:
                        refined_boxes.append([gy1, gx1, gy2, gx2])
                        confidences.append(conf_val)
                        continue
            except Exception:
                pass

            # Fallback to initial heuristic box if depth fit is unavailable
            refined_boxes.append([int(y1), int(x1), int(y2), int(x2)])
            confidences.append(0.0)

        return np.array(refined_boxes, dtype=np.int32), np.array(confidences, dtype=np.float32)

    def generate_debug_panel(
        self,
        frame_rgb: np.ndarray,
        candidate_box: np.ndarray,
        depth_full: Optional[np.ndarray] = None,
        zoom: int = 5
    ) -> np.ndarray:
        """
        Generate a multi-stage debug visualization panel:
          - Top row: 4 image stages (RGB Crop, Depth DA2, Binned Counts, Circle/Ray Fit).
          - Bottom row: Graph of aliased ray depth slices (Green: resolved, Red: unresolved).
          
        Args:
            frame_rgb: Full resolution RGB frame (H, W, 3)
            candidate_box: Initial bounding box [y1, x1, y2, x2]
            depth_full: Full-resolution depth map
            zoom: Magnification factor for top image strip
            
        Returns:
            BGR combined debug image
        """
        if depth_full is None:
            depth_full = self.compute_depth(frame_rgb)

        h, w = frame_rgb.shape[:2]
        y1, x1, y2, x2 = candidate_box
        bw, bh = x2 - x1, y2 - y1
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        radius = max(self.min_crop_radius, int(max(bw, bh) * self.crop_scale))
        cy1 = max(0, cy - radius)
        cx1 = max(0, cx - radius)
        cy2 = min(h, cy + radius)
        cx2 = min(w, cx + radius)

        crop_rgb = frame_rgb[cy1:cy2, cx1:cx2]
        crop_depth = depth_full[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop_rgb.shape[:2]

        pil_crop = Image.fromarray(crop_rgb)
        r = Raster(
            pil_crop,
            cache=jnp.array(crop_depth),
            resolution=(crop_h, crop_w),
            candidates=1,
            subdivisions=self.subdivisions,
            extent=2
        )

        # 1. Top Image Strip Panels
        counts = np.array(r.depth.binned().counts)
        counts_norm = np.uint8(counts / (np.max(counts) + 1e-8) * 255)
        counts_color = cv2.applyColorMap(counts_norm, cv2.COLORMAP_JET)

        d_norm = np.uint8((crop_depth - crop_depth.min()) / (crop_depth.max() - crop_depth.min() + 1e-8) * 255)
        depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)

        opt = r.opt(1)
        fit = opt.fit
        trace = opt.points

        pts_y = np.array(trace.points[0, 0])
        pts_x = np.array(trace.points[1, 0])
        pts_valid = np.array(trace.valid[0])

        z_rgb = cv2.resize(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), (crop_w * zoom, crop_h * zoom), interpolation=cv2.INTER_NEAREST)
        z_depth = cv2.resize(depth_color, (crop_w * zoom, crop_h * zoom), interpolation=cv2.INTER_NEAREST)
        z_counts = cv2.resize(counts_color, (crop_w * zoom, crop_h * zoom), interpolation=cv2.INTER_NEAREST)
        z_overlay = z_rgb.copy()

        c_y, c_x, c_r = float(fit.center_0th[0]), float(fit.center_1st[0]), float(fit.radius[0])

        for py, px, v in zip(pts_y, pts_x, pts_valid):
            if v and not np.isnan(py) and not np.isnan(px):
                pt_screen = (int(round(px * zoom)), int(round(py * zoom)))
                center_screen = (int(round(c_x * zoom)), int(round(c_y * zoom)))
                cv2.line(z_overlay, center_screen, pt_screen, (100, 200, 255), 1)
                cv2.circle(z_overlay, pt_screen, 4, (255, 0, 255), -1)
                cv2.circle(z_overlay, pt_screen, 5, (0, 0, 0), 1)

        if not np.isnan(c_y) and not np.isnan(c_x) and c_r > 0:
            cv2.circle(z_overlay, (int(round(c_x * zoom)), int(round(c_y * zoom))), int(round(c_r * zoom)), (0, 255, 255), 2)
            cv2.drawMarker(z_overlay, (int(round(c_x * zoom)), int(round(c_y * zoom))), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)

        hx1 = int((x1 - cx1) * zoom)
        hy1 = int((y1 - cy1) * zoom)
        hx2 = int((x2 - cx1) * zoom)
        hy2 = int((y2 - cy1) * zoom)
        cv2.rectangle(z_overlay, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

        cv2.putText(z_rgb, "1. RGB Crop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(z_depth, "2. Depth (DA2)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(z_counts, "3. Binned Counts", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        samples_count = int(fit.samples[0]) if not np.isnan(fit.samples[0]) else 0
        r_str = f"r={c_r:.1f}px" if not np.isnan(c_r) and c_r > 0 else "no fit"
        cv2.putText(z_overlay, f"4. Fit ({samples_count} pts, {r_str})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        top_strip = np.hstack([z_rgb, z_depth, z_counts, z_overlay])
        strip_w = top_strip.shape[1]

        # 2. Bottom Row: Depth Slice Ray Graph
        lo_depths = np.array(opt.adjacent[0][0])
        hi_depths = np.array(opt.adjacent[1][0])
        ray_depths = (lo_depths + hi_depths) / 2.0
        occludes = np.array(opt.occludes[0][0])

        fig, ax = plt.subplots(figsize=(strip_w / 100.0, 3.2), dpi=100)
        fig.patch.set_facecolor("#0e0e14")
        ax.set_facecolor("#161622")

        distances = np.arange(ray_depths.shape[1])
        # Plot unresolved rays in solid blue
        for i in range(len(pts_valid)):
            if not pts_valid[i]:
                ax.plot(distances, ray_depths[i], color="#3388ff", alpha=0.4, linewidth=1.1, linestyle="-")

        # Plot resolved rays in green
        for i in range(len(pts_valid)):
            if pts_valid[i]:
                ax.plot(distances, ray_depths[i], color="#00ff88", alpha=0.85, linewidth=1.6)
                step_idx = int(np.clip(occludes[i], 0, len(distances) - 1))
                ax.scatter(step_idx, ray_depths[i, step_idx], color="#ff00ff", s=35, zorder=5, edgecolors="#ffffff", linewidth=0.8)

        if not np.isnan(c_r) and c_r > 0:
            ax.axvline(x=c_r, color="#ffff00", linestyle=":", linewidth=2, label=f"Fitted Radius ({c_r:.1f}px)")

        ax.tick_params(colors="#cccccc", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#3a3a4c")
        ax.grid(True, color="#2a2a3c", linestyle="--", alpha=0.6)
        ax.set_xlabel("Ray Step Distance (pixels from center)", color="#ffffff", fontsize=10)
        ax.set_ylabel("Depth Disparity", color="#ffffff", fontsize=10)
        resolved_count = int(np.sum(pts_valid))
        unresolved_count = len(pts_valid) - resolved_count
        ax.set_title(f"Ray Depth Slices (Green: {resolved_count} Resolved Dropoffs, Blue: {unresolved_count} Unresolved Rays, Magenta: Occlusion Edge)", color="#ffffff", fontsize=11, pad=8)

        legend_elements = [
            Line2D([0], [0], color="#00ff88", lw=2, label=f"Resolved ({resolved_count})"),
            Line2D([0], [0], color="#3388ff", lw=1.5, label=f"Unresolved ({unresolved_count})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff00ff", markersize=6, label="Dropoff Edge"),
        ]
        if not np.isnan(c_r) and c_r > 0:
            legend_elements.append(Line2D([0], [0], color="#ffff00", linestyle=":", lw=2, label=f"Fitted r={c_r:.1f}px"))
        ax.legend(handles=legend_elements, loc="upper right", facecolor="#1e1e2c", edgecolor="#3a3a4c", labelcolor="#ffffff", fontsize=8)

        plt.tight_layout()
        fig.canvas.draw()
        plot_rgba = np.asarray(fig.canvas.buffer_rgba())
        plot_bgr = cv2.cvtColor(plot_rgba, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        if plot_bgr.shape[1] != strip_w:
            plot_bgr = cv2.resize(plot_bgr, (strip_w, plot_bgr.shape[0]), interpolation=cv2.INTER_AREA)

        return np.vstack([top_strip, plot_bgr])
