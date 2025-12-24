import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

im4 = M2.file(examples_dir / "IMG_0004.HEIC", cache_dir / "m2_out4.npy")
im5 = M2.file(examples_dir / "IMG_0005.HEIC", cache_dir / "m2_out5.npy")
im7 = M2.file(examples_dir / "IMG_0007.HEIC", cache_dir / "m2_out7.npy")
im8 = M2.file(examples_dir / "IMG_0008.HEIC", cache_dir / "m2_out8.npy")

if __name__ == "__main__":
    im = im4
    bins = im.depth.binned()
    levels = Seives.create(bins)
    mask = [levels.nms(i) for i in range(1, len(levels.stack) - 1)]
    print([i.shape for i in levels.stack])
    print([i.primaries.shape for i in levels.stack[1:]])
    print([i.shape for i in mask])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    i = 2
    ax[0].imshow(levels.stack[i + 2].primaries)
    ax[1].imshow(mask[i])
    plt.show()

exit(0)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for im in [im4, im5, im7, im8]:
    # for im in [im4]:
        im.draw_candidates(plt.subplots()[1])
        plt.show()
