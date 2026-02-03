import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

im4 = M2.file(examples_dir / "IMG_0004.HEIC", cache_dir / "m2_out4.npy")
crop = (slice(160, 210), slice(240, 290))
sphere = im4.depth.depth[crop]
stats = Seives.create(im4.depth.binned()).stat(1)[1]

import matplotlib.pyplot as plt

a = jnp.array([
    stats.mean.center_0th - crop[0].start,
    stats.mean.center_1st - crop[1].start])

for b in range(7):
    c = AliasedRay(sphere, a, b, 20)
    # fig1, fig2 = plt.figure(), plt.figure()
    # ax1, ax2 = fig1.add_subplot(111), fig2.add_subplot(111)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.scatter(*a[::-1])
    d = jnp.hstack(c.steps)
    ax1.imshow(sphere)
    ax1.scatter(*d[::-1])

    # lo, hi = c.adjacent()
    mean, std = stats.mean.depth[0], stats.std.depth[0]
    # ax2.plot(lo)
    # ax2.plot(hi)
    # ax2.axhline(mean - std)
    # ax2.axhline(mean)
    # ax2.axhline(mean + std)

    ax1.scatter(*c.poi(mean - std, mean + std)[::-1])

    plt.show()
