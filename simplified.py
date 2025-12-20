import sys, pathlib, math
from functools import cached_property, partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.signal import correlate2d
from transformers import pipeline

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

local = pathlib.Path(__file__).parents[0]

# TODO: switch to Depth Anything 3
# TODO: why is this casting to u8 at the end? it's quantized to f16.
#       is the other version the same granularity but normalized?
class Da2:
    size_mapping = { 'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large' }

    def __init__(self, encoder):
        self.encoder = encoder

    @property
    def model_repo(self):
        size = self.size_mapping[self.encoder]
        return f'depth-anything/Depth-Anything-V2-{size}-hf'

    @cached_property
    def model(self):
        pipe = pipeline(task="depth-estimation", model=self.model_repo)
        return lambda x: pipe(x)["depth"]

    def __call__(self, im):
        return jnp.array(self.model(im))

class Da2:
    model_configs = {
        'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    size_mapping = { 'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large' }

    def __init__(self, encoder):
        self.encoder = encoder

    @property
    def model_config(self):
        return { "encoder": self.encoder, **self.model_configs[self.encoder] }

    @property
    def model_repo(self):
        size = self.size_mapping[self.encoder]
        return f'https://huggingface.co/depth-anything/Depth-Anything-V2-{size}'

    @property
    def model_path(self):
        return f'main/depth_anything_v2_{self.encoder}.pth'

    @property
    def model_url(self):
        return f'{self.model_repo}/resolve/{self.model_path}?download=true'

    @cached_property
    def model(self):
        import torch
        sys.path.insert(0, str(local / "assets" / "depth_anything_v2"))
        from depth_anything_v2.dpt import DepthAnythingV2
        sys.path.pop(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else \
                'mps' if torch.backends.mps.is_available() else 'cpu'

        model = DepthAnythingV2(**self.model_config)
        model.load_state_dict(torch.hub.load_state_dict_from_url(
                self.model_url, map_location='cpu'))
        return model.to(DEVICE).eval()

    def __call__(self, im):
        import numpy as np
        return jnp.array(self.model.infer_image(np.array(im)))

class Raster:
    rng = [jax.random.key(0)]
    key = None
    model = Da2('vits')
    target = None
    f_35mm = None

    def data(self, *a, **kw):
        return Depth(*a, rng=self.key, **kw)

    @classmethod
    def file(cls, path, npy=None):
        im, cache = Image.open(path), None
        if npy is not None:
            npy = pathlib.Path(npy)
            if npy.exists():
                cache = jnp.load(npy)
                if jnp.any(cache.shape != im.size[::-1]) if cls.target is None \
                        else jnp.any(cache.shape != cls.target):
                    cache = None
        obj = cls(im, cache)
        if npy is not None:
            npy.parents[0].mkdir(parents=True, exist_ok=True)
            jnp.save(npy, obj.cache)
        return obj

    @cached_property
    def cache(self):
        return self.model(self.cropped())

    @cached_property
    def depth(self):
        return self.data(self.cache)

    @property
    def spec(self):
        return self.full.size if self.target is None else self.target

    @property
    def shape(self):
        return jnp.array(self.spec)

    @property
    def coord(self):
        x, y = jnp.meshgrid(*map(jnp.arange, self.spec[::-1]))
        return jnp.stack((y, x), -1)

    def cropped(self):
        if self.target is None:
            return self.full
        size = jnp.array(self.full.size)
        scaled = jnp.int32(jnp.max(self.shape[::-1] / size) * size)
        resample = self.full.resize(scaled)
        origin = (scaled - self.shape[::-1]) // 2
        return resample.crop(jnp.concat((origin, origin + self.shape[::-1])))

    def __init__(self, im, cache=None):
        self.rng[0], self.key = jax.random.split(self.rng[0])
        self.full = im
        if cache is not None:
            self.cache = cache

    def draw_candidates(self, ax):
        import matplotlib.patches as patches
        ax.imshow(self.cropped())
        _, bboxes = self.depth.binned().nominate()
        kw = { 'linewidth': 1, 'edgecolor': 'r', 'facecolor': 'none' }
        for i in jnp.unstack(bboxes):
            rect = patches.Rectangle(i[1::-1], *(i[:1:-1] - i[1::-1]), **kw)
            ax.add_patch(rect)
        return ax

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["depth", "rng"], meta_fields=[])
@dataclass
class Depth(object):
    depth: any
    rng: any

    @property
    def shape(self):
        return jnp.array(self.depth.shape)

    @cached_property
    def coord(self):
        x, y = jnp.meshgrid(*map(jnp.arange, self.depth.shape[::-1]))
        return jnp.stack((y, x), -1)

    @staticmethod
    def scharr(arr):
        kernel = [[-47, -162, -47], [0, 0, 0], [47, 162, 47]]
        l1 = sum(map(abs, sum(kernel, [])))
        kernel = jnp.array(kernel) / l1
        kw = { 'boundary': 'fill', 'fillvalue': 0, 'mode': 'same' }
        return jnp.stack((
                correlate2d(arr, kernel, **kw),
                correlate2d(arr, kernel.T, **kw)), -1)

    @cached_property
    def norm2(self):
        return jnp.sum(self.grad ** 2, -1)

    @cached_property
    def norm(self):
        return jnp.sqrt(self.norm2)

    @cached_property
    def grad(self):
        return self.scharr(self.depth)

    @cached_property
    def hessian(self):
        return jnp.stack((
                self.scharr(self.grad[..., 0]),
                self.scharr(self.grad[..., 1])), -1)

    @cached_property
    def rotated(self):
        norm = self.norm[..., None]
        basis0 = jnp.where(norm != 0, self.grad / norm, 0)
        basis1 = basis0[..., ::-1] * jnp.array([[[-1, 1]]])
        basis = jnp.stack((basis0, basis1), -1)
        inv = basis * jnp.array([[[[1, -1], [-1, 1]]]])
        return inv @ self.hessian @ basis

    @cached_property
    def inwards(self):
        convex = jnp.logical_and(
                jnp.linalg.det(self.rotated) > 0, self.rotated[..., 0, 0] < 0)
        return jnp.logical_and(
                convex, self.rotated[..., 0, 0] <= self.rotated[..., 1, 1])

    @cached_property
    def flat_radius_over_norm(self):
        return jnp.where(self.inwards, 1 / self.rotated[..., 1, 1], 0)

    @cached_property
    def centers(self):
        return self.coord - self.flat_radius_over_norm[..., None] * self.grad

    @cached_property
    def radii(self):
        da2, db2 = self.rotated[..., 0, 0], self.rotated[..., 1, 1]
        sec2 = jnp.where(self.inwards, da2 / (db2 ** 2 * (da2 - db2)), 1)
        p = self.norm * jnp.sqrt(sec2)
        bound = jnp.linalg.norm(jnp.array(self.depth.shape))
        return jax.lax.min(bound, p)

    @cached_property
    def masked(self):
        return jnp.where(self.inwards[..., None], self.centers, -1)

    @jax.jit
    def binned(self):
        centers = self.masked.reshape(-1, 2)
        indices = Casts2d(self.depth.shape, self.masked)
        counts = indices.scatter('add', 1, 0)
        flat_0th = jnp.ravel(self.coord[..., 0])
        flat_1st = jnp.ravel(self.coord[..., 1])
        bounds = jnp.stack((
            indices.scatter('min', flat_0th, self.depth.shape[0]),
            indices.scatter('min', flat_1st, self.depth.shape[1]),
            indices.scatter('max', flat_0th, -1),
            indices.scatter('max', flat_1st, -1)), -1)
        bounds = Bounds(
                self.depth.shape, bounds, counts,
                jnp.array([0, 0]), jnp.array([1, 1]), jnp.array([0, 0]))
        return Bins(
                bounds,
                indices.stat(centers[:, 0]),
                indices.stat(centers[:, 1]),
                indices.stat(jnp.ravel(self.radii)))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["indices"], meta_fields=["shape"])
@dataclass
class Casts2d(object):
    shape: any
    indices: any

    def __init__(self, shape, continuous):
        self.shape, shape = shape, jnp.array(shape)
        assert continuous.shape[-1] == 2
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
        floored = jnp.int32(jnp.floor(continuous))
        valid = jnp.logical_and(floored >= 0, floored < shape[None])
        valid = jnp.logical_and(valid[:, 0], valid[:, 1])
        self.indices = jnp.where(valid[:, None], floored, shape[None])

    def scatter(self, mode, value, fill):
        out = jnp.full(self.shape, fill)
        fn = getattr(out.at[self.indices[..., 0], self.indices[..., 1]], mode)
        return fn(value, mode="drop")

    def stat(self, values):
        return BinStat(
                BinSum(self.scatter('add', values, 0.)),
                BinSum(self.scatter('add', values ** 2, 0.)))

bin_win = ((2, 2), (2, 2))  # dimensions, strides

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["bounds", "counts", "origin", "scale", "offset"],
        meta_fields=["shape"])
@dataclass
class Bounds(object):
    shape: any
    bounds: any
    counts: any
    origin: any
    scale: any
    offset: any

    alpha = 0.1  # density stabilization coefficient
    beta = 1.5  # count exponent; no real justification
    off = (
            (slice(0, -1), slice(0, -1)),
            (slice(1, None), slice(0, -1)),
            (slice(0, -1), slice(1, None)),
            (slice(1, None), slice(1, None)))

    def merge(self):
        f = jax.lax.reduce_window
        inits = tuple(map(self.bounds.dtype.type, self.shape + (-1, -1)))
        reduced = jnp.stack((
                f(self.bounds[..., 0], inits[0], jax.lax.min, *bin_win),
                f(self.bounds[..., 1], inits[1], jax.lax.min, *bin_win),
                f(self.bounds[..., 2], inits[2], jax.lax.max, *bin_win),
                f(self.bounds[..., 3], inits[3], jax.lax.max, *bin_win)), -1)
        counts = f(self.counts, 0, jax.lax.add, *bin_win)
        return __class__(
                self.shape, reduced, counts,
                self.origin, self.scale * jnp.array(bin_win[1]))

    def area(self):
        hi = self.bounds[..., 2:] + (self.bounds[..., 2:] < 0)
        y, x = jnp.unstack(hi - self.bounds[..., :2], axis=-1)
        return jnp.int32(y) * x

    @cached_property
    def metric(self):
        areas, total = self.area(), self.counts.size
        return (self.counts ** self.beta) / (
                areas + self.alpha * total * self.scale[0] * self.scale[1]) / (
                self.scale[0] * self.scale[1])

    def __getitem__(self, key):
        offset = jnp.array([i.start for i in key])
        origin = offset * self.scale + self.origin
        return __class__(
                self.shape, self.bounds[key], self.counts[key],
                origin, self.scale, offset)

    def sifted(self):
        hi, val = None, None
        for i in self.off:
            shift = self[i].merge()
            total = jnp.sum(shift.metric)
            if val is None:
                hi, val = total, shift
            else:
                hi, val = jax.lax.cond(
                        total > hi, lambda: (total, shift), lambda: (hi, val))
        return val

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["bounds", "center_0th", "center_1st", "radius"],
        meta_fields=[])
@dataclass
class Bins(object):
    bounds: any
    center_0th: any
    center_1st: any
    radius: any

    def sifted(self):
        bounds = self.bounds.sifted()
        return Bins(
                bounds,
                self.center_0th.merge(bounds.offset),
                self.center_1st.merge(bounds.offset),
                self.radius.merge(bounds.offset))

# TODO(?): https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
@partial(
        jax.tree_util.register_dataclass,
        data_fields=["sum", "sum_sq"], meta_fields=[])
@dataclass
class BinStat(object):
    sum: any
    sum_sq: any

    def merge(self, offset):
        return BinStat(self.sum.merge(offset), self.sum_sq.merge(offset))

    def mean(self, counts):
        return self.sum.sum / counts

    def var(self, counts):
        return (self.sum_sq.sum - self.sum.sum ** 2 / counts) / (counts - 1)

    def stat(self, counts):
        mean = self.mean(counts)
        return mean, (self.sum_sq.sum - self.sum.sum * mean) / (counts - 1)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["sum"], meta_fields=[])
@dataclass
class BinSum(object):
    sum: any

    def merge(self, offset):
        assert bin_win[0] == bin_win[1] == (2, 2)
        shifted = jax.lax.dynamic_slice(offset, [i - 1 for i in self.sum.shape])
        return BinSum(jax.lax.reduce_window(shifted, 0., jax.lax.add, *bin_win))

class Perspective(Raster):
    f_35mm = None

    @property
    def f_px(self):
        return self.f_35mm / 35 * jnp.linalg.norm(self.shape)

    @property
    def fov_sec(self):
        f_center = (self.coord - self.shape[None] / 2) / self.f_px
        return jnp.sqrt(1 + jnp.sum(f_center ** 2, axis=-1))

    @cached_property
    def depth(self):
        if self.f_35mm is None:
            return self.data(self.cache)
        # trends the right direction since depths are flipped
        return self.data(self.cache / self.fov_sec)

class M2(Perspective):
    target = (392, 518)  # coremltools benchmark resolution
    f_35mm = 18

class Demo(Perspective):  # Logitech Webcam C925e
    target = (1080, 1920)
    f_35mm = 15.17  # 3.67 mm focal length / (1/3 inch sensor size) * (35mm)

class Demo(Demo):  # testing
    target = (392, 518)

examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"
