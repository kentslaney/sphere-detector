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
                self.depth.shape, 1, bounds, counts,
                jnp.array([0, 0]), jnp.array([0, 0]), None)
        return Bins(
                0, bounds,
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
        hi = self.bounds[..., 2:] + (self.bounds[..., 2:] < 0)
        y, x = jnp.unstack(hi - self.bounds[..., :2], axis=-1)
        return jnp.int32(y) * x

    @cached_property
    def metric(self):
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
        data_fields=["bounds", "center_0th", "center_1st", "radius"],
        meta_fields=["level"])
@dataclass
class Bins(object):
    level: any
    bounds: any
    center_0th: any
    center_1st: any
    radius: any

    @property
    def counts(self):
        return self.bounds.counts

    @property
    def shape(self):
        return self.counts.shape

    def sifted(self):
        bounds = self.bounds.sifted()
        return __class__(
                self.level + 1, bounds,
                self.center_0th.merge(bounds.offset),
                self.center_1st.merge(bounds.offset),
                self.radius.merge(bounds.offset))

    @cached_property
    def primaries(self):
        assert bin_win[0] == bin_win[1] == (2, 2)
        centers = self.bounds.centers
        primary_0th = self.center_0th.mean(self.counts) > centers[..., 0]
        primary_1st = self.center_1st.mean(self.counts) > centers[..., 1]
        alternating = jnp.array([True, False])
        alternating_0th = jnp.tile(alternating[:, None], self.shape)
        alternating_1st = jnp.tile(alternating[None, :], self.shape)
        _0th = jnp.logical_xor(jnp.repeat(primary_0th, 2, 0), alternating_0th)
        _1st = jnp.logical_xor(jnp.repeat(primary_1st, 2, 1), alternating_1st)
        return jnp.logical_and(jnp.repeat(_0th, 2, 1), jnp.repeat(_1st, 2, 0))

    def unshift(self, arr, fill=None):
        fill = jnp.zeros((), dtype=arr.dtype) if fill is None else fill
        def full(i, n):
            assert 0 <= n <= 2, f"padded {n}; more than expected"
            return jnp.full((arr.shape[:i] + (n,) + arr.shape[i + 1:]), fill)
        for i, prefix in enumerate(self.bounds.offset):
            arr = jax.lax.cond(
                    prefix == 1,
                    lambda: jnp.concat((full(i, 1), arr), axis=i),
                    lambda: jnp.concat((arr, full(i, 1)), axis=i))
            arr = jnp.concat(
                    (arr, full(i, self.bounds.upscale[i] - arr.shape[i])),
                    axis=i)
        assert arr.shape[:2] == self.bounds.upscale[:2]
        return arr

    def pyramids(self, init=None):
        inc = jnp.ones_like(self.counts, dtype=jnp.int32) if init is None \
                else init + 1
        upscaled = jnp.repeat(jnp.repeat(inc, 2, 0), 2, 1)
        return upscaled * self.primaries

def kron_bool(a, b):
    return jnp.bool(jnp.kron(jnp.uint8(a), jnp.uint8(b)))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["stack"], meta_fields=[])
@dataclass
class Seives(object):
    stack: any

    def __init__(self, base, layers=None):
        layers = int(math.log2(base.counts.size) // 3) if layers is None \
                else layers - 1
        out = [base]
        for _ in range(layers):
            out.append(out[-1].sifted())
        self.stack = tuple(out)

    def ruler(self, axis): # OEIS A001511
        up = jnp.ones(self.stack[-1].shape[axis] + 3, dtype=jnp.int32)
        out, pre, post = ([], []), 1, -2
        for cur in self.stack[:0:-1]:
            out[0].append(up[post - 1:pre - 1:-1])
            out[1].append(up[pre:post])
            assert cur.shape[axis] == out[0][-1].size == out[1][-1].size

            up = jnp.ravel(jnp.column_stack((jnp.ones_like(up), up + 1)))

            pad_left = cur.bounds.offset[axis]
            pre = 2 * pre - pad_left
            post = 2 * post + 1 + cur.bounds.valid_conv_pad[axis] - pad_left
            assert cur.bounds.upscale[axis] == up.size - pre + post
        return tuple(tuple(i[::-1]) for i in out)

    @cached_property
    def ruler_0th(self):
        return self.ruler(0)

    @cached_property
    def ruler_1st(self):
        return self.ruler(1)

    @cached_property
    def pyramids(self):
        cur, out = None, []
        fn = lambda x: x
        for layer in self.stack[:0:-1]:
            cur = layer.pyramids(fn(cur))
            out.append(cur)
            fn = layer.unshift
        return tuple(out[::-1])

    def nms(self, level):
        assert 0 < level < len(self.stack) - 1
        # Only primaries can be candidates, subject to the filter:
        #     Primaries block out 1 level down surroundings
        #         whose common ancestor
        #             is a primary or is the first non-primary in the path
        empty = jnp.zeros((4, 4), dtype=jnp.bool)
        kernels = \
            [  [empty.at[0, 0].set(True)
            ,   empty.at[0, 3].set(True)
            ,   empty.at[3, 0].set(True)
            ,   empty.at[3, 3].set(True)
            ], [empty.at[0, 1:4].set(True)
            ,   empty.at[0, 0:3].set(True)
            ,   empty.at[3, 1:4].set(True)
            ,   empty.at[3, 0:3].set(True)
            ], [empty.at[1:4, 0].set(True)
            ,   empty.at[1:4, 3].set(True)
            ,   empty.at[0:3, 0].set(True)
            ,   empty.at[0:3, 3].set(True)
            ], [empty.at[1:4, 3].set(True).at[3, 1:3].set(True)
            ,   empty.at[1:4, 0].set(True).at[3, 1:3].set(True)
            ,   empty.at[0:3, 3].set(True).at[0, 1:3].set(True)
            ,   empty.at[0:3, 0].set(True).at[0, 1:3].set(True)
            ]  ]
        strides = [
                (slice(0, None, 2), slice(0, None, 2)),
                (slice(0, None, 2), slice(1, None, 2)),
                (slice(1, None, 2), slice(0, None, 2)),
                (slice(1, None, 2), slice(1, None, 2))]
        masks = [None] * 3
        out = ([], [], [], [])
        for a, b in zip(strides, kernels[3]):
            out[3].append(kron_bool(self.stack[level + 1].primaries[a], b))
        mask1lo   = self.ruler_0th[0][level][:, None]
        mask1hi   = self.ruler_0th[1][level][:, None]
        masks[1] = [mask1lo, mask1lo, mask1hi, mask1hi]
        mask2even = self.ruler_1st[0][level][None, :]
        mask2odd  = self.ruler_1st[1][level][None, :]
        masks[2] = [mask2even, mask2odd, mask2even, mask2odd]
        mask0prod = [
                (self.ruler_0th[0][level], self.ruler_1st[0][level]),
                (self.ruler_0th[0][level], self.ruler_1st[1][level]),
                (self.ruler_0th[1][level], self.ruler_1st[0][level]),
                (self.ruler_0th[1][level], self.ruler_1st[1][level])]
        masks[0] = [jax.lax.max(*jnp.meshgrid(x, y)) for y, x in mask0prod]
        for i, (mask, kernel) in enumerate(zip(masks, kernels[:3])):
            for bound, a, b in zip(mask, strides, kernel):
                mask = self.pyramids[level][a] >= bound
                mask = jnp.logical_and(mask, self.stack[level + 1].primaries[a])
                out[i].append(kron_bool(mask, b))
        reduced_1st = []
        for ll, lh, hl, hh in out:
            ll = jnp.concatenate((ll[1:, :], jnp.zeros((1, ll.shape[1]))), 0)
            ll = jnp.concatenate((ll[:, 1:], jnp.zeros((ll.shape[0], 1))), 1)

            lh = jnp.concatenate((lh[1:, :], jnp.zeros((1, lh.shape[1]))), 0)
            lh = jnp.concatenate((jnp.zeros((lh.shape[0], 1)), lh[:, :-1]), 1)

            hl = jnp.concatenate((jnp.zeros((1, hl.shape[1])), hl[:-1, :]), 0)
            hl = jnp.concatenate((hl[:, 1:], jnp.zeros((hl.shape[0], 1))), 1)

            hh = jnp.concatenate((jnp.zeros((1, hh.shape[1])), hh[:-1, :]), 0)
            hh = jnp.concatenate((jnp.zeros((hh.shape[0], 1)), hh[:, :-1]), 1)

            reduced_1st.append(jnp.logical_or(
                jnp.logical_or(ll, hh),
                jnp.logical_or(lh, hl)))
        return jnp.logical_or(
                jnp.logical_or(reduced_1st[0], reduced_1st[3]),
                jnp.logical_or(reduced_1st[1], reduced_1st[2]))

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
        shape = [i - 1 for i in self.sum.shape]
        shifted = jax.lax.dynamic_slice(self.sum, offset, shape)
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
