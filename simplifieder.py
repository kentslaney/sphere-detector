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

    def draw_candidates(self, ax, max_outliers=3):
        import matplotlib.patches as patches
        ax.imshow(self.cropped())
        _, bboxes = Seives.create(self.depth.binned()).nominate()
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
    def sobel(arr):
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
        return self.sobel(self.depth)

    @cached_property
    def hessian(self):
        return jnp.stack((
                self.sobel(self.grad[..., 0]),
                self.sobel(self.grad[..., 1])), -1)

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

    def binned(self):
        centers = self.masked.reshape(-1, 2)
        indices = Casts2d.create(self.depth.shape, self.masked)
        counts = indices.scatter('add', 1, 0)
        flat_0th = jnp.ravel(self.coord[..., 0])
        flat_1st = jnp.ravel(self.coord[..., 1])
        bounds = jnp.stack((
            indices.scatter('min', flat_0th, self.depth.shape[0]),
            indices.scatter('min', flat_1st, self.depth.shape[1]),
            indices.scatter('max', flat_0th, -1),
            indices.scatter('max', flat_1st, -1)), -1)
        bounds = Bounds(
                self.depth.shape, 1, bounds, counts,
                jnp.array([0, 0]), jnp.array([0, 0]), None)
        return Bins(0, bounds)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["indices"], meta_fields=["shape"])
@dataclass
class Casts2d(object):
    shape: any
    indices: any

    @classmethod
    def create(cls, shape, continuous):
        _shape, shape = shape, jnp.array(shape)
        assert continuous.shape[-1] == 2
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
        floored = jnp.int32(jnp.floor(continuous))
        valid = jnp.logical_and(floored >= 0, floored < shape[None])
        valid = jnp.logical_and(valid[:, 0], valid[:, 1])
        indices = jnp.where(valid[:, None], floored, shape[None])
        return cls(_shape, indices)

    def scatter(self, mode, value, fill):
        out = jnp.full(self.shape, fill)
        fn = getattr(out.at[self.indices[..., 0], self.indices[..., 1]], mode)
        return fn(value, mode="drop")

bin_win = ((2, 2), (2, 2))  # dimensions, strides

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["bounds", "counts", "origin", "offset"],
        meta_fields=["shape", "scale", "upscale"])
@dataclass
class Bounds(object):
    shape: any
    scale: any
    bounds: any
    counts: any
    origin: any
    offset: any
    upscale: any

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
        assert bin_win[0] == bin_win[1] == (2, 2)
        return __class__(
                self.shape, self.scale * bin_win[1][0], reduced, counts,
                self.origin, self.offset, self.upscale)

    def area(self):
        hi = jax.lax.max(self.bounds[..., :2], self.bounds[..., 2:]) + (
                self.bounds[..., 2:] >= 0)
        y, x = jnp.unstack(hi - self.bounds[..., :2], axis=-1)
        return jnp.int32(y) * x

    @cached_property
    def metric(self):
        # TODO
        areas, total = self.area(), self.counts.size
        alpha, beta = 1.5, 0.1
        return (self.counts ** beta) / (
                areas + alpha * total * self.scale ** 2) / (
                self.scale ** 2)
        # counts ~ area
        # counts ** 1.5 / area / sqrt(scale) ~ sqrt(area / scale)
        # which is resolution invariant
        return (self.counts ** 1.5) / \
                jax.lax.max(self.area(), self.scale ** 2) / self.scale

    def __getitem__(self, key):
        offset = jnp.array([i.start for i in key])
        origin = offset * self.scale + self.origin
        return __class__(
                self.shape, self.scale, self.bounds[key], self.counts[key],
                origin, offset, self.counts.shape)

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

    @cached_property
    def coord(self):
        x, y = jnp.meshgrid(*map(jnp.arange, self.counts.shape[::-1]))
        return jnp.stack((y, x), -1)

    @property
    def centers(self):
        assert self.scale > 1
        return self.origin[None, None] + \
                self.scale * self.coord + self.scale // 2

    @property
    def valid_conv_pad(self):
        assert self.scale > 1
        return tuple((i - 1) % 2 for i in self.upscale)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["bounds"],
        meta_fields=["level"])
@dataclass
class Bins(object):
    level: any
    bounds: any

    @property
    def counts(self):
        return self.bounds.counts

    @property
    def shape(self):
        return self.counts.shape

    def sifted(self):
        bounds = self.bounds.sifted()
        return __class__(self.level + 1, bounds)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["stack"], meta_fields=[])
@dataclass
class Seives(object):
    stack: any

    @classmethod
    def create(cls, base, layers=None):
        # TODO
        layers = int(math.log2(base.counts.size) // 3) if layers is None \
                else layers - 1
        out = [base]
        for _ in range(layers):
            out.append(out[-1].sifted())
        return cls(tuple(out))

    @jax.jit(static_argnames=['candidates'])
    def nominate(self, candidates=16):
        values = jnp.ravel(self.stack[-1].bounds.metric)
        boundaries = self.stack[-1].bounds.bounds.reshape(-1, 4)
        for layer in self.stack[-2::-1]:
            nominees = layer.bounds.metric
            values = jnp.concatenate((values, jnp.ravel(nominees)))
            boundaries = jnp.concatenate(
                    (boundaries, layer.bounds.bounds.reshape(-1, 4)))
        values, indices = jax.lax.top_k(values, candidates)
        return values, boundaries[indices]

class M2(Raster):
    target = (392, 518)  # coremltools benchmark resolution

examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"
