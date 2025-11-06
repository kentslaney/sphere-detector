import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from static import im4, im5, im7, im8
sys.path.pop(0)

import jax.numpy as jnp
import matplotlib.pyplot as plt

def sifting(im):
    out = [im.depth.binned()] + [None] * 5
    for i in range(1, len(out)):
        out[i] = out[i - 1].sifted()

    fig, rows = plt.subplots(2, len(out))
    for ax0, ax1, bins in zip(*rows, out):
        ax0.imshow(bins.counts)
        metric = bins.metric()
        mean = bins.combine(metric) / bins.combined()
        ax1.imshow(metric / mean / jnp.log(bins.counts.size))
    plt.show()

if __name__ == "__main__":
    sifting(im4)
    sifting(im5)
    sifting(im7)
    sifting(im8)
