import matplotlib.pyplot as plt

from .examples import im4, im5, im7, im8
from .display import Readable

print(Readable(im.opt().surface.debug() for im in [im4, im5, im7, im8]))

for im in [im4, im5, im7, im8]:
    im.draw_refit()
    plt.show()
