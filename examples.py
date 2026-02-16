import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from display import *
sys.path.pop(0)

examples = local / "assets" / "examples"
cache = local / "cache"

im4 = Example.file(examples / "IMG_0004.HEIC", cache / "da2_4.npy", "im4")
im5 = Example.file(examples / "IMG_0005.HEIC", cache / "da2_5.npy", "im5")
im7 = Example.file(examples / "IMG_0007.HEIC", cache / "da2_7.npy", "im7")
im8 = Example.file(examples / "IMG_0008.HEIC", cache / "da2_8.npy", "im8")

import matplotlib.pyplot as plt
print(Readable(im.debug() for im in [im4, im5, im7, im8]))

_, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
im4.plot_depths(0, ax0)
im7.plot_depths(7, ax1)
im7.plot_depths(2, ax2)
im8.plot_depths(2, ax3)
plt.show()
