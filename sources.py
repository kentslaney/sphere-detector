import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from static import im4, im5, im7, im8
sys.path.pop(0)

import matplotlib.pyplot as plt
import numpy as np

def show_sources(im):
    sources = im.depth.sources(topk=24) # (h, w, topk, 2)
    background = im.depth.density() # (h, w)

    fig, ax = plt.subplots()
    ax.imshow(background, cmap='gray', interpolation='nearest')
    ax.set_title('Hover over a pixel to see its sources')

    # Create an empty image for the overlay
    h, w = background.shape
    # RGBA overlay, initially all transparent
    overlay_data = np.zeros((h, w, 4), dtype=float)
    overlay = ax.imshow(overlay_data)

    def on_move(event):
        if event.inaxes is not ax:
            return

        # Get mouse coordinates as integer indices
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)

        # Check bounds
        if not (0 <= y < h and 0 <= x < w):
            return

        # Create a boolean mask for the sources
        mask = np.zeros_like(background, dtype=bool)
        pixel_sources = sources[y, x] # (topk, 2)
        for sy, sx in pixel_sources:
            if sx != -1 and sy != -1: # Check for invalid coordinates
                mask[sy, sx] = True

        # Update overlay: show mask in red with 50% opacity
        overlay.set_data(np.stack([mask, np.zeros_like(mask), np.zeros_like(mask), mask * 0.5], axis=-1))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()

if __name__ == "__main__":
    show_sources(im5)