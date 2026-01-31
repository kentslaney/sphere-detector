import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

im4 = M2.file(examples_dir / "IMG_0004.HEIC", cache_dir / "m2_out4.npy")
# print(im4.depth.depth)
import matplotlib.pyplot as plt
plt.imshow(im4.depth.depth)
plt.show()
exit(0)
im5 = M2.file(examples_dir / "IMG_0005.HEIC", cache_dir / "m2_out5.npy")
im7 = M2.file(examples_dir / "IMG_0007.HEIC", cache_dir / "m2_out7.npy")
im8 = M2.file(examples_dir / "IMG_0008.HEIC", cache_dir / "m2_out8.npy")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # for im in [im4, im5, im7, im8]:
    for im in [im4]:
        im.draw_candidates(plt.subplots()[1])
        plt.show()
