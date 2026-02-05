import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

im4 = M2.file(examples_dir / "IMG_0004.HEIC", cache_dir / "m2_out4.npy")
crop = (slice(160, 210), slice(240, 290))
sphere = im4.depth.depth[crop]
stats = Seives.create(im4.depth.binned()).stat(1)[1]
cropped = stats.offset(jnp.array([crop[0].start, crop[1].start]))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

opt = AliasedRay.from_binstats(sphere, cropped, 63, 20)
fit = opt.fit()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.imshow(sphere)
ax1.scatter(*opt.poi[jnp.where(opt.poi >= 0)].reshape(2, -1)[::-1])

kw = { 'linewidth': 1, 'edgecolor': 'r', 'facecolor': 'none' }
for y, x, r in jnp.stack(fit).T:
    overlay = patches.Circle((x, y), r, **kw)
    ax1.add_patch(overlay)

plt.show()
