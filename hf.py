import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from functools import partial, cached_property
import sympy as sy
from typing import Literal

local = pathlib.Path(__file__).parents[0]
examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"
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

    import torch, sys
    sys.path.insert(0, str(local / "assets" / "depth_anything_v2"))
    from depth_anything_v2.dpt import DepthAnythingV2

    DEVICE = 'cuda' if torch.cuda.is_available() else \
            'mps' if torch.backends.mps.is_available() else 'cpu'

    model_repo = lambda size: \
        f'https://huggingface.co/depth-anything/Depth-Anything-V2-{size}'
    model_path = lambda encoder: f'resolve/main/depth_anything_v2_{encoder}.pth'
    model_urls = {
        'vits': f'{model_repo('Small')}/{model_path('vits')}?download=true',
        'vitb': f'{model_repo('Base')}/{model_path('vitb')}?download=true',
        'vitl': f'{model_repo('Large')}/{model_path('vitl')}?download=true',
    }

    model_configs = {
        'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    model_configs = {k: {**v, 'encoder': k} for k, v in model_configs.items()}

    encoder = 'vits' # 'vits', 'vitb', 'vitl'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        model_urls[encoder], map_location='cpu'))
    model = model.to(DEVICE).eval()

    return lambda im: model.infer_image(np.array(im))

def fs(pth, npy=None):
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
    out = np.array(mlmodel(im_t))
    if npy is not None:
        npy.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(npy, out)
    return out

im4 = fs(examples_dir / "IMG_0004.HEIC", cache_dir / "out4.npy")
im5 = fs(examples_dir / "IMG_0005.HEIC", cache_dir / "out5.npy")
im7 = fs(examples_dir / "IMG_0007.HEIC", cache_dir / "out7.npy")
im8 = fs(examples_dir / "IMG_0008.HEIC", cache_dir / "out8.npy")

scharr = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

def grad(arr):
    return np.stack((
        correlate2d(arr, scharr, boundary='symm', mode='same'),
        correlate2d(arr, scharr.T, boundary='symm', mode='same')), -1) / np.sum(
                np.abs(scharr))

def hessian(partials):
    return np.stack((grad(partials[..., 0]), grad(partials[..., 1])), -1)

def rotated(partials, second):
    norm = np.sqrt(np.sum(partials ** 2, -1, keepdims=True))
    basis0 = np.divide(
            partials, norm, out=np.zeros_like(partials), where=norm != 0)
    basis1 = basis0[..., ::-1] * np.array([[[-1, 1]]])
    basis = np.stack((basis0, basis1), -1)
    inv = basis * np.array([[[[1, -1], [-1, 1]]]])
    return inv @ second @ basis

def interpolate(continuous, *a, **kw):
    depths = None
    if continuous.shape[-1] == 3:
        continuous, depths = continuous[..., :2], continuous[..., 2]
        total, total_z = np.zeros(target), np.zeros(target)
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
        np.add.at(
                total,
                (filling[valid][..., 0], filling[valid][..., 1]),
                1)
        np.add.at(
                total_z,
                (filling[valid][..., 0], filling[valid][..., 1]),
                depths[valid])
    if depths is None:
        return out
    average = np.divide(
            total_z, total, where=total != 0, out=np.full(target, np.nan))
    return out, average

def orthogonal_centers(arr):
    partials = grad(arr)
    second = hessian(partials)
    aligned = rotated(partials, second)
    concave_away = np.logical_and(
            np.linalg.det(aligned) > 0, aligned[..., 0, 0] < 0)
    inwards = np.logical_and(
            aligned[..., 0, 0] <= aligned[..., 1, 1], concave_away)
    flat_radius_over_norm = np.divide(
            1, aligned[..., 1, 1],
            out=np.zeros_like(arr), where=inwards)
    yx = coord - flat_radius_over_norm[..., None] * partials

    sec2 = np.divide(
            aligned[..., 0, 0], aligned[..., 1, 1],
            out=np.ones_like(arr), where=inwards)
    norm2 = np.sum(partials ** 2, -1)
    coef = np.divide(sec2 - 1, norm2, out=np.ones_like(norm2), where=norm2 != 0)
    z = arr + np.divide(
            flat_radius_over_norm, coef,
            out=np.zeros_like(arr), where=inwards)

    return np.concatenate((yx, z[..., None]), axis=-1)

def tmp0(arr=im4, crop=(slice(171, 204), slice(250, 281))):
    out = orthogonal_centers(arr)
    density, depth = interpolate(out.reshape(-1, 3))
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(density[crop])
    axarr[1].imshow(depth[crop], vmin=0)
    axarr[2].imshow(arr[crop], vmin=0)
    plt.show()

def tmp1(arr=im4):
    out = orthogonal_centers(arr)
    density, depth = interpolate(out.reshape(-1, 3))
    plt.imshow(density)
    plt.show()

if __name__ == "__main__":
    tmp0()
