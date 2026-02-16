import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from display import *
sys.path.pop(0)

examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"

im4 = Example.file(examples_dir / "IMG_0004.HEIC", cache_dir / "out4.npy")
im5 = Example.file(examples_dir / "IMG_0005.HEIC", cache_dir / "out5.npy")
im7 = Example.file(examples_dir / "IMG_0007.HEIC", cache_dir / "out7.npy")
im8 = Example.file(examples_dir / "IMG_0008.HEIC", cache_dir / "out8.npy")

import matplotlib.pyplot as plt
for im in [im4, im5, im7, im8]:
    im.readable()

# im4.plot_depths(0); plt.show()
# im7.plot_depths(7); plt.show()
# im7.plot_depths(2); plt.show()
# im8.plot_depths(2); plt.show()
