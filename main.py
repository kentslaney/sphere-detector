import pathlib
import numpy as np
import matplotlib.pyplot as plt

local = pathlib.Path(__file__).parents[0]
mlmodel = None
target = np.array([392, 518])
diag = np.sqrt(np.sum(target ** 2))

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
    size = np.array(im.size)
    scaled = np.int32(np.max(target / size) * size)
    im_s = im.resize(scaled)
    origin = (scaled - target[::-1]) // 2
    im_t = im_s.crop(np.concat((origin, origin + target[::-1])))
    out = np.array(mlmodel.predict({"image": im_t})['depth'])
    if npy is not None:
        np.save(npy, out)
    return out

im4 = fs(local / "IMG_0004.HEIC", local / "out4.npy")
im5 = fs(local / "IMG_0005.HEIC", local / "out5.npy")

def partials(arr):
    from scipy.ndimage import sobel
    return sobel(arr, 0), sobel(arr, 1)

def hessian(arr):
    from scipy.ndimage import sobel
    return list(map(partials, partials(arr)))

# https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
# L = T / 2 \pm (T ** 2 / 4 - D) ** 1/2
def eigenvalues2x2(sq):
    tr = sq[0][0] + sq[1][1]
    det = tr - sq[0][1] * sq[1][0]
    center = tr / 2
    radius = np.sqrt(tr ** 2 / 4 - det)
    return [center - radius, center + radius]

def normals(d0, d1, t):
    out = np.zeros(target)
    t *= diag
    c1, c0 = np.meshgrid(*map(np.arange, target[::-1]))
    e0, e1 = (c0 + d0 * t).flatten(), (c1 + d1 * t).flatten()
    f0, f1 = np.int32(np.floor(e0)), np.int32(np.floor(e1))
    g0, g1 = e0 - f0, e1 - f1
    for i0, i1 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        z0, z1 = f0 + i0, f1 + i1
        y0, y1 = (1 - i0) + (2 * i0 - 1) * g0, (1 - i1) + (2 * i1 - 1) * g1
        x0 = np.logical_and(z0 >= 0, z0 < target[0])
        x1 = np.logical_and(z1 >= 0, z1 < target[1])
        w = np.logical_and(x0, x1)
        out[z0[w], z1[w]] += y0[w] * y1[w]
    return out

def sources(d0, d1, t, s1, s0):
    t *= diag
    c1, c0 = np.meshgrid(*map(np.arange, target[::-1]))
    e0, e1 = c0 + d0 * t, c1 + d1 * t
    return np.log(np.sqrt((e0 - s0) ** 2 + (e1 - s1) ** 2))

def slide(arr):
    frame = np.ones_like(arr)
    d0, d1 = partials(arr)

    from matplotlib.widgets import Slider, Button
    fig, ax = plt.subplots()
    out = ax.imshow(frame, vmin=0, vmax=4)
    fig.subplots_adjust(bottom=0.25)

    def onclick(event):
        if event.inaxes != ax:
            return
        out.set_data(sources(d0, d1, slider.val, event.xdata, event.ydata))
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', onclick)

    axt = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(
        ax=axt,
        label='t',
        valmin=0,
        valmax=8,
        valinit=0,
    )
    def update(val):
        nonlocal frame
        frame = normals(d0, d1, slider.val)
        out.set_data(frame)
        fig.canvas.draw_idle()
    slider.on_changed(update)

    bax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(bax, 'reset', hovercolor='0.975')
    def bev(event):
        out.set_data(frame)
        fig.canvas.draw_idle()
    button.on_clicked(bev)

    plt.show()

def density(arr, samples=100, scale=4):
    d0, d1 = partials(arr)
    t = np.linspace(0, scale, samples)
    out = np.zeros((samples,) + arr.shape)
    for i in range(samples):
        out[i] = normals(d0, d1, t[i])
    return out

def blur(arr, sigma=3):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)

def ndmax(arr):
    return np.unravel_index(arr.argmax(), arr.shape)

def fake_sphere(r=np.min(target) // 5):
    out = np.zeros(target)
    c1, c0 = np.meshgrid(*map(np.arange, target[::-1]))
    t0, t1 = target // 2
    d2 = (t0 - c0) ** 2 + (t1 - c1) ** 2
    out[d2 < r ** 2] = np.sqrt(r ** 2 - d2[d2 < r ** 2])
    out *= 0.8 / np.max(out)
    return out

def tmp0():
    slide(fake_sphere())

# https://www.desmos.com/c/h0pkuzhfzh
# https://www.desmos.com/3d/9ijh5b7ok9
# m = ∂∥∂∥ / (∂⟂∂⟂)^3 * ∂∥ wrt gradient

def tmp1(im):
    for i in im:
        plt.plot(i)
    plt.show()

if __name__ == "__main__":
    slide(im4)
