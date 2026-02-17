from .examples import *

import matplotlib.pyplot as plt

print(Readable(im.debug() for im in [im4, im5, im7, im8]))

_, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
im4.plot_depths(0, ax0)
im7.plot_depths(7, ax1)
im7.plot_depths(2, ax2)
im8.plot_depths(2, ax3)
plt.show()
