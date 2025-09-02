import pathlib
import numpy as np
import matplotlib.pyplot as plt

local = pathlib.Path(__file__).parents[0]
mlmodel = None

def mlpackage():
    global Image
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()

    import coremltools as ct
    return ct.models.MLModel('DepthAnythingV2SmallF16.mlpackage')

def fs(pth, npy = None):
    global mlmodel
    if type(npy) is str:
        npy = pathlib.Path(npy)
    if npy is not None and npy.exists():
        return np.load(npy)
    if mlmodel is None:
        mlmodel = mlpackage()
    im = Image.open(pth)
    target = np.array([518, 392])
    size = np.array(im.size)
    scaled = np.int32(np.max(target / size) * size)
    im_s = im.resize(scaled)
    origin = (scaled - target) // 2
    im_t = im_s.crop(np.concat((origin, origin + target)))
    out = np.array(mlmodel.predict({"image": im_t})['depth'])
    if npy is not None:
        np.save(npy, out)
    return out

a = fs(local / "IMG_0004.HEIC", local / "out4.npy")
b = fs(local / "IMG_0005.HEIC", local / "out5.npy")

def partials(arr):
    from scipy.ndimage import sobel
    return sobel(arr, 0), sobel(arr, 1)

def normals(d0, d1, t):
    out = np.zeros_like(d0)
    t *= np.sqrt(np.sum(np.array(out.shape) ** 2))
    c1, c0 = np.meshgrid(*map(np.arange, out.shape[::-1]))
    e0, e1 = (c0 + d0 * t).flatten(), (c1 + d1 * t).flatten()
    f0, f1 = np.int32(np.floor(e0)), np.int32(np.floor(e1))
    g0, g1 = e0 - f0, e1 - f1
    for i0, i1 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        z0, z1 = f0 + i0, f1 + i1
        y0, y1 = (1 - i0) + (2 * i0 - 1) * g0, (1 - i1) + (2 * i1 - 1) * g1
        x0 = np.logical_and(z0 >= 0, z0 < out.shape[0])
        x1 = np.logical_and(z1 >= 0, z1 < out.shape[1])
        w = np.logical_and(x0, x1)
        out[z0[w], z1[w]] += y0[w] * y1[w]
    return out

def slide(arr):
    from matplotlib.widgets import Slider
    fig, ax = plt.subplots()
    out = ax.imshow(np.ones_like(arr), vmin=0, vmax=4)
    fig.subplots_adjust(bottom=0.25)

    axt = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(
        ax=axt,
        label='t',
        valmin=0,
        valmax=8,
        valinit=0,
    )

    d0, d1 = partials(arr)
    def update(val):
        out.set_data(normals(d0, d1, slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    slide(a)
