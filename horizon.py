import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
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

def grad(arr):
    return np.stack((sobel(arr, 0), sobel(arr, 1)), -1) / 8

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

def supports(continuous, partials, topk=8):
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
    return horizon_vectorized(out)

def casts(im, slopes, depth):
    delta = im - depth
    intersection = coord + delta[..., None] * slopes
    return supports(
            intersection[delta > 0].reshape([-1, 2]), slopes[delta > 0])

def slide(f, **kw):
    frame = f(kw["valinit"] if "valinit" in kw else 0)

    from matplotlib.widgets import Slider, Button
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

def depth_slices(arr=im5):
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
    slide_partials(arr, slopes)

def cis_h(half_turns):
    half_turns = np.array(half_turns) * np.pi
    return np.stack((np.cos(half_turns), np.sin(half_turns)))

# non-parametric (unknown shape) but zero-pad agnostic
def horizon_metric(rays: flat2d):
    assert rays.shape[0] == 2
    # make anti-parallel an independent axis
    halved = np.atan2(*rays[::-1]) / 2
    transformed = np.stack((np.cos(halved), np.sin(halved))) * \
            np.linalg.norm(rays, axis=0, keepdims=True)
    # consider each of the "support rays" along an extra dimension
    expanded = np.vstack((transformed, np.eye(rays.shape[1])))
    # svd.U is an orthonormal basis for the column space of expanded
    svd = np.linalg.svd(expanded, full_matrices=False)
    # sub is the unstretched embedding of expanded into one dimension per point
    sub = svd.U.T @ expanded
    # and the absoulte value of the determinant gives the volume formed by the
    #     new vectors, plus an extra for the unit cube added by the identity
    return np.abs(np.linalg.det(sub)) - 1

# TODO
horizon_vectorized = np.vectorize(horizon_metric, signature='(m, n)->()')

def fov_fix(arr):
    return 1 - fov_csc * (1 - arr)

if __name__ == "__main__":
    depth_slices(im4)
