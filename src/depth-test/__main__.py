import matplotlib.pyplot as plt

from .examples import im4, im5, im7, im8
from .display import Readable

print(Readable(im.opt().surface.debug() for im in [im4, im5, im7, im8]))

_, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
im4.opt().plot_depths(0, ax0)
im7.opt().plot_depths(7, ax1)
im7.opt().plot_depths(2, ax2)
im8.opt().plot_depths(2, ax3)
plt.show()
