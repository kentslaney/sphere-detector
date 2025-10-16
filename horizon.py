import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from functools import partial, cached_property
import sympy as sy
from typing import Literal

local = pathlib.Path(__file__).parents[0]
mlmodel = None
target = np.array([392, 518])
coord = np.stack(np.meshgrid(*map(np.arange, target[::-1]))[::-1], -1)
diag = np.sqrt(np.sum(target ** 2))

f_px = 18 / 35 * np.linalg.norm(target)
fov_csc = np.sqrt(1 + np.sum(((coord - target[None] / 2) / f_px) ** 2, axis=-1))

flat2d = np.ndarray[tuple[Literal[2], int], np.dtype[np.float64]]

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
im7 = fs(local / "IMG_0007.HEIC", local / "out7.npy")
im8 = fs(local / "IMG_0008.HEIC", local / "out8.npy")

scharr = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

def grad(arr):
    return np.stack((
        correlate2d(arr, scharr, boundary='symm', mode='same'),
        correlate2d(arr, scharr.T, boundary='symm', mode='same')), -1) / np.sum(
                np.abs(scharr))

def hessian(partials):
    return np.stack((grad(partials[..., 0]), grad(partials[..., 1])), -1)

def interpolate(continuous, *a, **kw):
    out = np.zeros(target)
    floored = np.int32(np.floor(continuous))
    remainder = continuous - floored
    for offset in np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]):
        filling = floored + offset
        overlap = 1 - offset + (2 * offset - 1) * remainder
        valid = np.all(np.logical_and(filling >= 0, filling < target[None]), axis=1)
        np.add.at(
                out,
                (filling[valid][..., 0], filling[valid][..., 1]),
                overlap[valid][..., 0] * overlap[valid][..., 1])
    return out

def raster_supports(continuous, partials, topk=24):
    sources = np.vstack((
        np.zeros([3, 4, continuous.shape[0]]),
        -np.ones([1, 4, continuous.shape[0]])))
    floored = np.int32(np.floor(continuous))
    remainder = continuous - floored
    normed = np.divide(
            partials, np.linalg.norm(partials, axis=1, keepdims=True),
            where=~np.all(partials == 0, axis=1, keepdims=True),
            out=np.zeros_like(partials))
    for offset in np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]):
        filling = floored + offset
        overlap = 1 - offset + (2 * offset - 1) * remainder
        valid = np.all(
                np.logical_and(filling >= 0, filling < target[None]), axis=1)
        overlap, filling = overlap[valid], filling[valid]
        filling_flat = filling[..., 0] * target[1] + filling[..., 1]
        values = overlap[..., 0] * overlap[..., 1]
        rays = normed[valid] * values[:, None]
        # lexsort is last-key-first but unique requires first-key-first
        # so the flattened index has to be used instead of multiple fields
        filling_flat = filling[..., 0] * target[1] + filling[..., 1]
        sources[:, offset[0, 0] * 2 + offset[0, 1], :filling_flat.shape[0]] = \
                np.vstack((rays[:, ::-1].T, -values[None], filling_flat[None]))
    sources = sources.reshape(4, -1)
    # ensure consistent behavior even when all valid
    sources = np.hstack((np.array([0, 0, 0, -1])[:, None], sources))
    sources = sources[:, np.lexsort(sources)]
    coord_flat, offset, counts = np.unique(
            sources[-1], return_index=True, return_counts=True, sorted=True)
    keeping = np.arange(topk)[None] < counts[:, None]
    ref = (offset[:, None] + np.arange(topk)[None]) * keeping
    deref = sources[:2, np.ravel(ref)].reshape(2, -1, topk)
    deref = np.transpose(deref, axes=(1, 0, 2))
    deref, coord_flat = deref[1:], np.int32(coord_flat[1:])
    # set coord_flat to deref
    out = np.zeros(target.tolist() + [2, topk])
    out[coord_flat // target[1], coord_flat % target[1]] = deref
    return out

def support_casts(im, slopes, depth):
    delta = im - depth
    intersection = coord + delta[..., None] * slopes
    return raster_supports(
            intersection[delta > 0].reshape([-1, 2]), slopes[delta > 0])

def casts(im, slopes, depth):
    return horizon_metric(support_casts(im, slopes, depth))
    delta = im - depth
    intersection = coord + delta[..., None] * slopes
    return interpolate(intersection[delta > 0].reshape([-1, 2]))

def slide(f, **kw):
    frame = f(kw["valinit"] if "valinit" in kw else 0)

    from matplotlib.widgets import Slider
    fig, ax = plt.subplots()
    out = ax.imshow(frame, vmin=0, vmax=4)
    fig.subplots_adjust(bottom=0.25)

    axt = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(ax=axt, label='t', **kw)
    def update(val):
        nonlocal frame
        frame = f(slider.val)
        out.set_data(frame)
        fig.canvas.draw_idle()
    slider.on_changed(update)

    plt.show()

def slide_partials(arr, slopes, **kw):
    defaults = { "valmin": 0, "valmax": 1, "valinit": 0.155 }
    return slide(partial(casts, arr, slopes), **{**defaults, **kw})

def rotated(partials, second):
    norm = np.sqrt(np.sum(partials ** 2, -1, keepdims=True))
    basis0 = np.divide(
            partials, norm, out=np.zeros_like(partials), where=norm != 0)
    basis1 = basis0[..., ::-1] * np.array([[[-1, 1]]])
    basis = np.stack((basis0, basis1), -1)
    inv = basis * np.array([[[[1, -1], [-1, 1]]]])
    return inv @ second @ basis

def relative_slopes(arr):
    partials = grad(arr)
    second = hessian(partials)
    out = rotated(partials, second)
    concave_down = np.logical_and(np.linalg.det(out) > 0, out[..., 0, 0] < 0)
    sec2 = np.divide(
            out[..., 0, 0], out[..., 1, 1],
            out=2 * np.ones_like(arr), where=concave_down)
    norm = np.sum(partials ** 2, -1)
    coef = np.divide(sec2 - 1, norm, out=np.ones_like(norm), where=norm != 0)
    slopes = partials * coef[..., None]
    return slopes

def depth_slices(arr=im5, **kw):
    slide_partials(arr, relative_slopes(arr), **kw)

def cis_h(half_turns):
    half_turns = np.array(half_turns) * np.pi
    return np.stack((np.cos(half_turns), np.sin(half_turns)))

def diagonal_stretch(ndim, k):
    stretch = np.eye(ndim) * k ** (-1 / (ndim - 1))
    stretch[0, 0] = k
    initial_basis = np.eye(ndim)
    initial_basis[:, 0] = np.ones(ndim)
    # Q is orthonormal with a matching direction for the first vector
    rotation, _ = np.linalg.qr(initial_basis)
    symmetric_matrix = rotation @ stretch @ rotation.T
    return symmetric_matrix

# non-parametric (unknown shape) but zero-pad agnostic
def horizon_metric(rays, stretch=1):
    assert rays.shape[-2] == 2
    n = rays.shape[-1]
    # make anti-parallel an independent axis
    halved = np.atan2(rays[..., 0, :], rays[..., 1, :]) / 2
    transformed = np.stack((np.cos(halved), np.sin(halved)), -2) * \
            np.linalg.norm(rays, axis=-2, keepdims=True)
    # consider each of the "support rays" along an extra dimension
    spread = np.eye(n) if stretch == 1 else diagonal_stretch(n, stretch)
    spread = np.broadcast_to(
            spread[(None,) * (rays.ndim - 2)], rays.shape[:-2] + (n, n))
    expanded = np.concatenate((transformed, spread), -2)
    expanded_T = np.moveaxis(expanded, -2, -1)
    # multiplicative vs additive metric; monotonic mapping
    return np.linalg.slogdet(expanded_T @ expanded).logabsdet
    # return np.sqrt(np.abs(np.linalg.det(expanded_T @ expanded))) - 1

def fov_fix(arr):
    return 1 - fov_csc * (1 - arr)

def density(arr, samples=100, cache=None, lo=0, hi=1):
    if cache is not None:
        cache = pathlib.Path(cache)
        if cache.exists():
            return np.load(cache)
    slopes = relative_slopes(arr)
    t = np.linspace(lo, hi, samples)
    out = np.zeros((samples,) + arr.shape)
    for i in range(samples):
        out[i] = casts(arr, slopes, t[i])
    if cache is not None:
        np.save(cache, out)
    return out

def slide_voxels(arr, **kw):
    defaults = { "valmin": 0, "valmax": 1, "valinit": 0.155 }
    samples = arr.shape[0]
    def update(val):
        return arr[np.minimum(samples - 1, np.int32(np.round(val * samples)))]
    return slide(update, **{**defaults, **kw})

def ndmax(arr):
    return np.unravel_index(arr.argmax(), arr.shape)

def sample(arr=im4, depth=0.155):
    frame = support_casts(arr, relative_slopes(arr), depth)

    from matplotlib.widgets import Slider
    fig, ax = plt.subplots()
    out = ax.imshow(horizon_metric(frame))
    fig.subplots_adjust(bottom=0.25)

    domain = { "valmin": -4, "valmax": 4, "valinit": 0 }
    axs = fig.add_axes([0.2, 0.15, 0.7, 0.03])
    axt = fig.add_axes([0.2, 0.05, 0.7, 0.03])
    ln_stretch = Slider(ax=axs, label='ln stretch', **domain)
    ln_scale = Slider(ax=axt, label='ln scale', **domain)
    def update(val):
        out.set_data(horizon_metric(
            np.e ** ln_scale.val * frame, stretch=np.e ** ln_stretch.val))
        fig.canvas.draw_idle()
    ln_stretch.on_changed(update)
    ln_scale.on_changed(update)

    plt.show()

def tmp(arr=im4, depth=0.155, pos=(186, 264), name=None):
    frame = support_casts(arr, relative_slopes(arr), depth)
    value = horizon_metric(frame[pos])
    print(frame[pos])
    print("horizon metric", *(() if name is None else ("for", name)), value)

def tmp2(arr=im4, depth=0.155, cmp=2):
    frame = support_casts(arr, relative_slopes(arr), depth)
    nominal = horizon_metric(frame)
    updated = horizon_metric(frame, stretch=cmp)
    return np.sum(np.argsort(nominal) == np.argsort(updated)) / nominal.size

if __name__ == "__main__":
    print(tmp2())
    # sample()
    # tmp(name="signal")
    # tmp(pos=(101, 130), name="noise")
    # slide_voxels(density(im4, cache="voxels4.npy", lo=0.1, hi=0.2))
    # slide_voxels(density(im5, cache="voxels5.npy", lo=0, hi=1))
    # depth_slices()
