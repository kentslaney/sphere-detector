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
        scaled = jnp.int32(jnp.max(self.shape / size) * size)
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
    def w2(self):
        sec2 = jnp.where(
                self.inwards, self.rotated[..., 0, 0] / self.rotated[..., 1, 1],
                1)
        return jnp.where(self.norm2 != 0, (sec2 - 1) / self.norm2, 1)

    @cached_property
    def w(self):
        return jnp.sqrt(self.w2)

    @cached_property
    def masked(self):
        return jnp.where(self.inwards[..., None], self.centers, -1)

    @jax.jit
    def binned(self):
        indices = Casts2d(self.depth.shape, self.masked)
        counts = indices.scatter('add', 1)
        bounds = jnp.stack((
            indices.scatter('min', jnp.ravel(self.coord[..., 0]), 0),
            indices.scatter('min', jnp.ravel(self.coord[..., 1]), 1),
            indices.scatter('max', jnp.ravel(self.coord[..., 0]), 0),
            indices.scatter('max', jnp.ravel(self.coord[..., 1]), 1)), -1)
        return Bins(
                self.depth.shape, bounds, counts,
                jnp.array([0, 0]), jnp.array([1, 1]))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["indices"], meta_fields=["shape"])
@dataclass
class Casts2d(object):
    shape: any
    indices: any

    dtype = jnp.int32

    def __init__(self, shape, continuous):
        self.shape, shape = shape, jnp.array(shape)
        assert continuous.shape[-1] == 2
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
        floored = self.dtype(jnp.floor(continuous))
        valid = jnp.logical_and(floored >= 0, floored < shape[None])
        valid = jnp.logical_and(valid[:, 0], valid[:, 1])
        self.indices = jnp.where(valid[:, None], floored, shape[None])

    def scatter(self, mode, value, axis=None):
        fill = self.shape[axis] if mode == 'min' else -1 if mode == 'max' else 0
        out = jnp.full(self.shape, fill, dtype=self.dtype)
        fn = getattr(out.at[self.indices[..., 0], self.indices[..., 1]], mode)
        return fn(value, mode="drop")

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["bounds", "counts", "origin", "scale"],
        meta_fields=["shape"])
@dataclass
class Bins(object):
    shape: any
    bounds: any
    counts: any
    origin: any
    scale: any

    alpha = 0.1 # density stabilization coefficient
    beta = 0.1 # proportion of pixels' metrics combined
    win = ((2, 2), (2, 2))
    off = (
            (slice(0, -1), slice(0, -1)),
            (slice(1, None), slice(0, -1)),
            (slice(0, -1), slice(1, None)),
            (slice(1, None), slice(1, None)))

    def merge(self):
        f = jax.lax.reduce_window
        operand = lambda i: {
                'bound': self.bounds[..., i], 'count': self.counts }
        inits = tuple(map(self.bounds.dtype.type, self.shape + (-1, -1)))
        reduced = jnp.stack((
                f(self.bounds[..., 0], inits[0], jax.lax.min, *self.win),
                f(self.bounds[..., 1], inits[1], jax.lax.min, *self.win),
                f(self.bounds[..., 2], inits[2], jax.lax.max, *self.win),
                f(self.bounds[..., 3], inits[3], jax.lax.max, *self.win)), -1)
        counts = f(self.counts, 0, jax.lax.add, *self.win)
        return __class__(
                self.shape, reduced, counts,
                self.origin, self.scale * jnp.array(self.win[1]))

    def area(self):
        hi = self.bounds[..., 2:] + (self.bounds[..., 2:] < 0)
        y, x = jnp.unstack(hi - self.bounds[..., :2], axis=-1)
        return jnp.int32(y) * x

    def metric(self):
        areas, total = self.area(), self.counts.size
        # TODO: total * scale is almost constant across scales, and this
        #       just incentivizes sourced area vs counts, not center density
        return (self.counts + self.alpha * jnp.sum(self.counts) / total) / (
                areas + self.alpha * total * self.scale[0] * self.scale[1])

    def combine(self, metric):
        cutoff = jnp.quantile(
                metric, 1 - self.beta * metric.size ** -0.5, method="higher")
        return jnp.sum(jnp.where(metric >= cutoff, metric, 0))

    def combine(self, metric):
        freq = -(-metric.size // 2 ** 14)
        cutoff = jnp.quantile(
                metric[::freq], 1 - self.beta * metric.size ** -0.5,
                method="higher")
        return jnp.sum(jnp.where(metric >= cutoff, metric, 0))

    def combined(self):
        return jnp.ceil(self.beta * jnp.sqrt(self.counts.size))

    def normalized(self):
        metric = self.metric()
        mean = self.combine(metric) / self.combined()
        return metric / mean

    def __getitem__(self, key):
        origin = jnp.array([i.start for i in key]) * self.scale + self.origin
        return __class__(
                self.shape, self.bounds[key], self.counts[key],
                origin, self.scale)

    @jax.jit
    def sifted(self):
        hi, val = None, None
        for i in self.off:
            shift = self[i].merge()
            total = jnp.sum(shift.metric())
            if val is None:
                hi, val = total, shift
            else:
                hi, val = jax.lax.cond(
                        total > hi, lambda: (total, shift), lambda: (hi, val))
        return val

    @jax.jit(static_argnames=['candidates', 'seives'])
    def nominate(self, candidates=16, seives=None):
        scaled, seives = self, seives or int(math.log2(self.counts.size) // 3)
        values = jnp.ravel(scaled.normalized())
        boundaries = scaled.bounds.reshape(-1, 4)
        for _ in range(seives):
            scaled = scaled.sifted()
            values = jnp.concatenate((values, jnp.ravel(scaled.normalized())))
            boundaries = jnp.concatenate(
                    (boundaries, scaled.bounds.reshape(-1, 4)))
        values, indices = jax.lax.top_k(values, candidates)
        return values, boundaries[indices]

    @property
    def coord(self):
        x, y = jnp.meshgrid(*map(jnp.arange, self.counts.shape[::-1]))
        return jnp.stack((y, x), -1)

    def sources(self): # inclusive ranges
        broadcastable = self.scale[None, None]
        top_left = self.origin[None, None] + broadcastable * self.coord
        bottom_right = top_left + broadcastable - 1
        return jnp.concatenate((top_left, bottom_right), axis=-1)

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
