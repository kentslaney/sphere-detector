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

a = jnp.array([[
    stats.mean.center_0th[0] - crop[0].start,
    stats.mean.center_1st[0] - crop[1].start]])
b = jnp.arange(63) / 10

mean, std = stats.mean.depth[0], stats.std.depth[0]
lo, hi = mean - std, mean + std
lo, hi = jnp.array([lo]), jnp.array([hi])

c = AliasedRay(sphere, a, b, 20)
d = c.poi(lo, hi)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.imshow(sphere)
ax1.scatter(*d[::-1])
plt.show()
