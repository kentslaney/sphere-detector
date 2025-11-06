import sys, pathlib
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
        cache = None
        if npy is not None:
            npy = pathlib.Path(npy)
            if npy.exists():
                cache = jnp.load(npy)
                if cls.target is not None and jnp.all(
                        cache.shape != cls.target):
                    cache = None
        obj = cls(Image.open(path), cache)
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
    def shape(self):
        return jnp.array(self.full.size if self.target is None else self.target)

    @property
    def coord(self):
        x, y = jnp.meshgrid(*map(jnp.arange, self.shape[::-1]))
        return jnp.stack((y, x), -1)

    def cropped(self):
        if self.target is None:
            return self.full
        size = jnp.array(self.full.size)
        scaled = jnp.int32(jnp.max(self.shape / size) * size)
        resample = self.full.resize(scaled)
        origin = (scaled - self.shape[::-1]) // 2
        return resample.crop(jnp.concat((origin, origin + self.shape[::-1])))

    def __init__(self, im, cache=None):
        self.rng[0], self.key = jax.random.split(self.rng[0])
        self.full = im
        if cache is not None:
            self.cache = cache

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["depth", "rng"], meta_fields=[])
@dataclass
class Depth(object):
    depth: any
    rng: any

    delta = 0.1 # boundary outlier proportion

    @property
    def shape(self):
        return jnp.array(self.depth.shape)

    @property
    def coord(self):
        x, y = jnp.meshgrid(*map(jnp.arange, self.depth.shape[::-1]))
        return jnp.stack((y, x), -1)

    @staticmethod
    def scharr(arr):
        kernel = jnp.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        l1 = jnp.sum(jnp.abs(kernel))
        kw = { 'boundary': 'fill', 'fillvalue': 0, 'mode': 'same' }
        return jnp.stack((
                correlate2d(arr, kernel, **kw),
                correlate2d(arr, kernel.T, **kw)), -1) / l1

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
    def w2(self):
        sec2 = jnp.where(
                self.inwards, self.rotated[..., 0, 0] / self.rotated[..., 1, 1],
                1)
        return jnp.where(self.norm2 != 0, (sec2 - 1) / self.norm2, 1)

    @cached_property
    def w(self):
        return jnp.sqrt(self.w2)

    def rasterize(self, continuous, interpolate=False):
        assert continuous.shape[-1] == 2
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
        out = jnp.zeros(self.depth.shape,
                dtype=self.depth.dtype if interpolate else jnp.int32)
        floored = jnp.int32(jnp.floor(continuous))
        remainder = continuous - floored
        offsets = jnp.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        for offset in offsets[slice(None) if interpolate else slice(0, 1)]:
            filling = floored + offset
            overlap = 1 - offset + (2 * offset - 1) * remainder if interpolate \
                    else jnp.array([1, 1])
            valid = jnp.all(jnp.logical_and(
                    filling >= 0, filling < self.shape[None]), axis=1)
            filling = jnp.where(valid[:, None], filling, self.shape[None])
            out = out.at[filling[..., 0], filling[..., 1]].add(
                    overlap[..., 0] * overlap[..., 1], mode="drop")
        return out

    def density(self, interpolate=False):
        arr = jnp.where(self.inwards[..., None], self.centers, -2)
        return self.rasterize(arr, interpolate)

    def bin(self, continuous, data, priority=None, topk=8, default=0):
        slots = data.shape[-1]
        if priority is None:
            self.rng, subkey = jax.random.split(self.rng)
            priority = jax.random.uniform(subkey, shape=continuous.shape[:-1])
        assert continuous.ndim == priority.ndim + 1 == data.ndim
        assert continuous.shape[-1] == 2
        assert continuous.size // 2 == priority.size == data.size // slots
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
            priority = priority.reshape(-1)
            data = data.reshape(-1, slots)

        mapping = { 'V': 'i', 'u': 'i', 'i': 'i', 'f': 'f', 'c': 'c' }
        padded = jnp.dtype(
                mapping[data.dtype.kind] + str(max(2, data.dtype.itemsize)))
        sources = jnp.vstack((
            jnp.full([slots, continuous.shape[0] + 1], default, dtype=padded),
            jnp.zeros([1, continuous.shape[0] + 1], dtype=padded),
            -jnp.ones([1, continuous.shape[0] + 1], dtype=padded)))
        floored = jnp.int32(jnp.floor(continuous))

        valid = jnp.all(jnp.logical_and(
                floored >= 0, floored < self.shape[None]), axis=1)
        floored, priority, data = floored[valid], priority[valid], data[valid]
        flat_index = floored[..., 0] * self.shape[1] + floored[..., 1]
        sources = sources.at[:, :flat_index.shape[0]].set(
                jnp.vstack((data.T, -priority[None], flat_index[None])))

        sources = sources[:, jnp.lexsort(sources)]
        coord_flat, offset, counts = jnp.unique(
                sources[-1], return_index=True, return_counts=True)
        keeping = jnp.arange(topk)[None] < counts[:, None]
        ref = (offset[:, None] + jnp.arange(topk)[None]) * keeping
        deref = sources[:slots, jnp.ravel(ref)].reshape(slots, -1, topk)
        deref = jnp.transpose(deref, axes=(1, 2, 0))
        deref, coord_flat = deref[1:], jnp.int32(coord_flat[1:])
        out = jnp.full(
                tuple(self.shape.tolist()) + (topk, slots), default,
                dtype=data.dtype)
        r, c = coord_flat // self.shape[1], coord_flat % self.shape[1]
        out = out.at[r, c].set(deref)
        return out

    def binned(self, max_outliers=3):
        counts = self.density()
        binner = lambda x, y: self.bin(
                self.centers[self.inwards], self.coord[self.inwards, x:x + 1],
                y * self.coord[self.inwards, x], topk=max_outliers + 1,
                default=-1)
        bounds = jnp.squeeze(jnp.stack((
            binner(0, -1),
            binner(1, -1),
            binner(0,  1),
            binner(1,  1)), -3), -1)
        idx = jnp.minimum(max_outliers, jnp.int32(counts * self.delta))
        bounds = jnp.take_along_axis(
                bounds, idx[..., None, None], axis=-1).squeeze(axis=-1)
        return Bins(bounds, counts, jnp.array([0, 0]), jnp.array([1, 1]))

    def sources(self, topk=8):
        return self.bin(
            self.centers[self.inwards], self.coord[self.inwards],
            topk=topk, default=-1)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["bounds", "counts", "origin", "scale"], meta_fields=[])
@dataclass
class Bins(object):
    bounds: any
    counts: any
    origin: any
    scale: any

    alpha = 0.1 # area stabilization coefficient
    beta = 0.1 # proportion of pixels' metrics combined
    gamma = 0.1 # negligible merge threshold
    win = ((2, 2), (2, 2))
    off = (
            (slice(0, -1), slice(0, -1)),
            (slice(1, None), slice(0, -1)),
            (slice(0, -1), slice(1, None)),
            (slice(1, None), slice(1, None)))

    @staticmethod
    def upper(a, b):
        merged = jax.lax.max(a['bound'], b['bound'])
        small = jax.lax.min(a['count'], b['count'])
        large = jax.lax.max(a['count'], b['count'])
        overrun = jnp.where(a['count'] > b['count'], a['bound'], b['bound'])
        bound = jnp.where(small <= __class__.gamma * large, overrun, merged)
        return { 'bound': bound, 'count': a['count'] + b['count'] }

    @staticmethod
    def lower(a, b):
        lo = jax.lax.min(a['bound'], b['bound'])
        hi = jax.lax.max(a['bound'], b['bound'])
        merged = jnp.where(lo == -1, hi, lo)

        small = jax.lax.min(a['count'], b['count'])
        large = jax.lax.max(a['count'], b['count'])
        overrun = jnp.where(a['count'] < b['count'], a['bound'], b['bound'])
        bound = jnp.where(small <= __class__.gamma * large, overrun, merged)
        return { 'bound': bound, 'count': a['count'] + b['count'] }

    def merge(self):
        f = jax.lax.reduce_window
        init = { 'bound': -1, 'count': 0 }
        operand = lambda i: {
                'bound': self.bounds[..., i], 'count': self.counts }
        reduced = jnp.stack((
                f(operand(0), init, self.lower, *self.win)['bound'],
                f(operand(1), init, self.lower, *self.win)['bound'],
                f(operand(2), init, self.upper, *self.win)['bound'],
                f(operand(3), init, self.upper, *self.win)['bound']), -1)
        counts = f(self.counts, 0, jax.lax.add, *self.win)
        return __class__(
                reduced, counts, self.origin,
                self.scale * jnp.array(self.win[1]))

    def area(self):
        hi = self.bounds[..., 2:] + (self.bounds[..., 2:] < -1)
        y, x = jnp.unstack(hi - self.bounds[..., :2], axis=-1)
        return y * x

    def metric(self):
        areas, total = self.area(), self.counts.size
        return self.counts / (areas + self.alpha / total * jnp.sum(areas))

    def combine(self, metric):
        cutoff = jnp.percentile(metric, 100 * (
            1 - self.beta * metric.size ** -0.5))
        return jnp.sum(jnp.where(metric > cutoff, metric, 0))

    def combined(self):
        return self.beta * jnp.sqrt(self.counts.size)

    def __getitem__(self, key):
        origin = jnp.array([i.start * self.scale for i in key]) + self.origin
        return __class__(self.bounds[key], self.counts[key], origin, self.scale)

    @jax.jit
    def sifted(self):
        hi, val = None, None
        for i in self.off:
            shift = self[i].merge()
            total = self.combine(shift.metric())
            if val is None:
                hi, val = total, shift
            else:
                hi, val = jax.lax.cond(
                        total > hi, lambda: (total, shift), lambda: (hi, val))
        return val

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
    target = (392, 518)
    f_35mm = 18

examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"

im4 = M2.file(examples_dir / "IMG_0004.HEIC", cache_dir / "out4.npy")
im5 = M2.file(examples_dir / "IMG_0005.HEIC", cache_dir / "out5.npy")
im7 = M2.file(examples_dir / "IMG_0007.HEIC", cache_dir / "out7.npy")
im8 = M2.file(examples_dir / "IMG_0008.HEIC", cache_dir / "out8.npy")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(im4.depth.density())
    plt.show()
