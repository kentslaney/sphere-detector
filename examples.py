import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from detect import *
sys.path.pop(0)

examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"

im4 = Raster.file(examples_dir / "IMG_0004.HEIC", cache_dir / "out4.npy")
im5 = Raster.file(examples_dir / "IMG_0005.HEIC", cache_dir / "out5.npy")
im7 = Raster.file(examples_dir / "IMG_0007.HEIC", cache_dir / "out7.npy")
im8 = Raster.file(examples_dir / "IMG_0008.HEIC", cache_dir / "out8.npy")

import matplotlib.pyplot as plt
im4.opt().plot_depths(0, "im4")
plt.show()
im7.opt().plot_depths(7, "im7")
plt.show()
im7.opt().plot_depths(2, "im7")
plt.show()
im8.opt().plot_depths(2, "im8")
plt.show()
exit(0)

# import matplotlib.pyplot as plt
for im in [im4, im5, im7, im8]:
    im.refit().readable()
    # im.draw_refit()
    # plt.show()
