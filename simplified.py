import sys, pathlib, math, inspect
from functools import cached_property, partial, wraps
from dataclasses import dataclass
from collections import namedtuple, OrderedDict

import jax
import jax.numpy as jnp
from jax.scipy.signal import correlate2d
from jax.scipy.optimize import minimize

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

local = pathlib.Path(__file__).parents[0]

# TODO: switch to Depth Anything 3 (after RealityKit and before Godot)
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

def poplt(x=None, init=None):
    import matplotlib.pyplot as plt
    if x is not None:
        return x, x.gca()
    fig = plt.figure()
    ax = fig.subplots()
    if init is not None:
        init(ax)
    return fig, ax

def jax_limit_cache(arg, *excluded, axis=0, maxsize=None):
    cache = OrderedDict()
    def decorator(f):
        sig = inspect.signature(f)
        @wraps(f)
        def wrapper(*a, **kw):
            bound = sig.bind(*a, **kw)
            bound.apply_defaults()
            limit = bound.arguments[arg]
            key = frozenset(
                    (k, v) for k, v in bound.arguments.items() if k != arg)
            res = None
            if key in cache:
                cache.move_to_end(key)
                size, res = cache[key]
                if size < limit:
                    res = None
                elif size == limit:
                    return res
            if res is None:
                res = f(*a, **kw)
                cache[key] = (limit, res)
                if maxsize is not None and len(cache) > maxsize:
                    cache.popitem(last=False)
                return res
            def mapping(path, x):
                if jax.tree_util.keystr(path) in excluded:
                    return x
                assert x.shape[axis] == size
                return jax.lax.slice_in_dim(x, 0, limit, axis=axis)
            return jax.tree.map_with_path(mapping, res)
        return wrapper
    return decorator

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["alpha", "beta", "delta", "eps", "eta", "chi"],
        meta_fields=["resolution", "candidates", "rays"])
@dataclass(kw_only=True)
class Config:
    # Raster
    resolution: any = (392, 518)
    candidates: any = 16
    rays: any = 64

    # Bounds
    eps: any = 0.1  # density stabilization coefficient

    # AliasedRay
    alpha: any = 0.0  # standard deviations above mean for ray start depth
    beta: any = 2.0  # standard deviations below mean for ray start depth
    delta: any = 0.5  # threshold in standard deviations for ray depth jump
    eta: any = 0.1  # ridge regression coefficient
    chi: any = 0.5  # standard deviations above initial mean radius to look

@jax.tree_util.register_pytree_node_class
class Raster:
    config: any
    model = Da2('vits')
    f_35mm = None

    def tree_flatten(self):
        return (jnp.array(self.full), self.cache, self.config), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        import numpy as np
        im, depths, config = children
        return cls(Image.fromarray(np.array(im)), depths, **config)

    def data(self, *a, **kw):
        return Depth(self.config, *a, **kw)

    @classmethod
    def file(cls, path, npy=None, **kw):
        im, cache = Image.open(path), None
        if npy is not None:
            npy = pathlib.Path(npy)
            if npy.exists():
                cache = jnp.load(npy)
                target = kw.get("resolution", None)
                if jnp.any(cache.shape != im.size[::-1]) if target is None \
                        else jnp.any(cache.shape != target):
                    cache = None
        obj = cls(im, cache, **kw)
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
        return self.full.size if self.config.resolution is None else \
                self.config.resolution

    @property
    def shape(self):
        return jnp.array(self.spec)

    @property
    def coord(self):
        x, y = jnp.meshgrid(*map(jnp.arange, self.spec[::-1]))
        return jnp.stack((y, x), -1)

    def cropped(self):
        if self.config.resolution is None:
            return self.full
        size = jnp.array(self.full.size)
        scaled = jnp.int32(jnp.max(self.shape[::-1] / size) * size)
        resample = self.full.resize(scaled)
        origin = (scaled - self.shape[::-1]) // 2
        return resample.crop(jnp.concat((origin, origin + self.shape[::-1])))

    def __init__(self, im, cache=None, **kw):
        self.full = im
        self.config = Config(**kw)
        if cache is not None:
            self.cache = cache

    @property
    def diag(self):
        return int(sum(i ** 2 for i in self.spec) ** 0.5)

    @cached_property
    def seives(self):
        return Seives.create(self.depth.binned())

    def cropped_background(self, fig=None):
        return poplt(fig, lambda ax: ax.imshow(self.cropped()))

    def draw_sifted(self, fig=None, candidates=16, color='r'):
        fig, ax = self.cropped_background(fig)
        import matplotlib.patches as patches
        _, bboxes = self.seives.bound(candidates)
        kw = { 'linewidth': 1, 'edgecolor': color, 'facecolor': 'none' }
        for i, x in enumerate(jnp.unstack(bboxes)):
            rect = patches.Rectangle(x[1::-1], *(x[:1:-1] - x[1::-1]), **kw)
            ax.add_patch(rect)
            ax.annotate(
                    str(i), x[1::-1], xytext=(1, -1), textcoords="offset points",
                    va='top', ha='left', color=color)
        return fig

    @jax_limit_cache('candidates')
    def stat(self, candidates=16):
        return self.seives.stat(candidates)

    def draw_centers(self, fig=None, candidates=16):
        fig, ax = self.cropped_background(fig)
        pred = self.stat(candidates)
        ax.scatter(*pred.mean.centers.T[::-1], color='b')
        for i, (y, x) in enumerate(pred.mean.centers):
            ax.annotate(str(i), (x, y), color='b')
        return fig

    @jax_limit_cache('candidates', '.depth', '.theta')
    def opt(self, candidates=16):
        pred = self.stat(candidates)
        return AliasedRay.from_binstats(
                self.depth, pred, self.config.rays, self.diag // 4)

    @jax_limit_cache('candidates')
    def refit(self, candidates=16):
        return self.opt(candidates).split().fit()

    def draw_refit(self, fig=None, candidates=16, index=None, label=True):
        fig, ax = self.cropped_background(fig)
        import matplotlib.patches as patches
        if index is None:
            stats = self.refit(candidates)
        else:
            stats = self.opt(index + 1).split().candidates[-1].fit()
        color = 'r'
        kw = { 'linewidth': 1, 'edgecolor': color, 'facecolor': 'none' }
        if label is not None:
            thetas = jax.random.uniform(
                    jax.random.key(label),
                    stats.radius.shape, maxval=2 * jnp.pi)
        it = jnp.unstack(
                jnp.stack((stats.center_0th, stats.center_1st, stats.radius)),
                axis=1)
        for y, x, r in it:
            overlay = patches.Circle((x, y), r, **kw)
            ax.add_patch(overlay)
        if label is not None and index is None:
            ax.autoscale(False)
            for i, (y, x, r) in enumerate(it):
                ax.plot(
                        [x, x + jnp.cos(thetas[i]) * r],
                        [y, y + jnp.sin(thetas[i]) * r], color=color)
                ax.annotate(str(i), (x, y), color=color)
        return fig

    def draw_candidate(self, index=0, fig=None):
        fig, ax = self.cropped_background(fig)
        opt = self.opt(index + 1).split().candidates[-1]
        ax.scatter(*opt.origin.T[::-1], color='r')
        ax.scatter(*opt.unsized[::-1], color='b')
        return fig

    def plot_rays(self, index=0):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax0, ax1 = fig.subplots(2, 1)

        opt = self.opt(index + 1).split().candidates[-1]
        for side in opt.adjacent():
            for cast in side[0]:
                ax0.plot(cast)
        ax0.axhline(opt.depth_mean[0])
        ax0.axhline(opt.depth_mean[0] + self.config.alpha * opt.depth_std[0])
        ax0.axhline(opt.depth_mean[0] - self.config.beta * opt.depth_std[0])
        ax0.axvline(opt.radius_mean[0] + self.config.chi * opt.radius_std[0])

        for side in opt.adjacent():
            for cast in side[0]:
                ax1.plot(cast[1:] - cast[:-1])
        ax1.axhline(-self.config.delta * opt.depth_std[0])
        ax1.axvline(opt.radius_mean[0] + self.config.chi * opt.radius_std[0])
        return fig

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["config", "depth"], meta_fields=[])
@dataclass
class Depth:
    config: any
    depth: any

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
                self.config, self.depth.shape, 1, bounds, counts,
                jnp.array([0, 0]), jnp.array([0, 0]), None)
        return Bins(
                0, bounds,
                indices.stat(centers[:, 0]),
                indices.stat(centers[:, 1]),
                indices.stat(jnp.ravel(self.radii)),
                indices.stat(jnp.ravel(self.depth)))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["indices"], meta_fields=["shape"])
@dataclass
class Casts2d:
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

    def stat(self, values):
        return BinStat(
                BinSum(self.scatter('add', values, 0.)),
                BinSum(self.scatter('add', values ** 2, 0.)))

bin_win = ((2, 2), (2, 2))  # dimensions, strides

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["config", "bounds", "counts", "origin", "offset"],
        meta_fields=["shape", "scale", "upscale"])
@dataclass
class Bounds:
    config: any
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
        return self.__class__(
                self.config, self.shape, self.scale * bin_win[1][0], reduced,
                counts, self.origin, self.offset, self.upscale)

    def area(self):
        hi = jax.lax.max(self.bounds[..., :2], self.bounds[..., 2:]) + (
                self.bounds[..., 2:] >= 0)
        y, x = jnp.unstack(hi - self.bounds[..., :2], axis=-1)
        return jnp.int32(y) * x

    # TODO: float16
    @cached_property
    def metric(self):
        # counts ~ area
        # counts ** 1.5 / area / sqrt(scale) ~ sqrt(area / scale)
        # which is resolution invariant
        areas, total = self.area(), self.counts.size
        # TODO: justify the epsilon term
        return (self.counts ** 1.5) / (
                areas + self.config.eps * total * self.scale ** 2) / self.scale

    def __getitem__(self, key):
        offset = jnp.array([i.start for i in key])
        origin = offset * self.scale + self.origin
        return self.__class__(
                self.config, self.shape, self.scale, self.bounds[key],
                self.counts[key], origin, offset, self.counts.shape)

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
        data_fields=["bounds", "center_0th", "center_1st", "radius", "depth"],
        meta_fields=["level"])
@dataclass
class Bins:
    level: any
    bounds: any
    center_0th: any
    center_1st: any
    radius: any
    depth: any

    @property
    def counts(self):
        return self.bounds.counts

    @property
    def shape(self):
        return self.counts.shape

    def sifted(self):
        bounds = self.bounds.sifted()
        return self.__class__(
                self.level + 1, bounds,
                self.center_0th.merge(bounds.offset),
                self.center_1st.merge(bounds.offset),
                self.radius.merge(bounds.offset),
                self.depth.merge(bounds.offset))

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

    def unshift(self, x, n=1, fill=None):
        fill = jnp.zeros((), dtype=x.dtype) if fill is None else fill
        def full(i, m):
            assert 0 <= m <= 2, f"padded {m}; more than expected"
            return jnp.full((x.shape[:i] + (n * m,) + x.shape[i + 1:]), fill)
        for i, prefix in enumerate(self.bounds.offset):
            x = jax.lax.cond(
                    prefix == 1,
                    lambda: jnp.concat((full(i, 1), x), axis=i),
                    lambda: jnp.concat((x, full(i, 1)), axis=i))
            x = jnp.concat(
                    (x, full(i, self.bounds.upscale[i] - x.shape[i] // n)),
                    axis=i)
        assert x.shape[:2] == tuple(i * n for i in self.bounds.upscale[:2])
        return x

    def pyramids(self, init=None):
        inc = jnp.ones_like(self.counts, dtype=jnp.int32) if init is None \
                else init + 1
        upscaled = jnp.repeat(jnp.repeat(inc, 2, 0), 2, 1)
        return upscaled * self.primaries

    stats = ('center_0th', 'center_1st', 'radius', 'depth')
    # means then variances
    def stat(self):
        out = []
        for i in self.stats:
            out.append(getattr(self, i).stat(self.counts))
        return jnp.stack(sum(zip(*out), ()), axis=-1)

def kron_bool(a, b):
    assert a.ndim == 2 and b.ndim == 2
    return jnp.logical_and(a[:, None, :, None], b[None, :, None, :]).reshape(
            a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["stack"], meta_fields=[])
@dataclass
class Seives:
    stack: any

    @classmethod
    def create(cls, base, layers=None):
        layers = int(math.log2(base.counts.size) // 3) if layers is None \
                else layers - 1
        out = [base]
        for _ in range(layers):
            out.append(out[-1].sifted())
        return cls(tuple(out))

    def ruler(self, axis): # OEIS A001511
        up = jnp.ones(self.stack[-1].shape[axis] + 4, dtype=jnp.int32)
        out, pre = ([], []), 2
        for cur in self.stack[:0:-1]:
            shape = (cur.shape[axis],)
            shifted = jax.lax.dynamic_slice(up, (pre,), shape)
            flipped = jax.lax.dynamic_slice(up[::-1], (pre,), shape)
            out[0].append(flipped)
            out[1].append(shifted)

            up = jnp.ravel(jnp.column_stack((jnp.ones_like(up), up + 1)))
            pre = 2 * pre - cur.bounds.offset[axis]
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

    @jax.jit(static_argnames=["level"])
    def nms(self, level):
        assert 0 < level < len(self.stack), "layer must store primaries"
        # Only primaries can be candidates, subject to the filter:
        #     Primaries block out 1 level down surroundings
        #         if the primary's upwards path to the common ancestor
        #             is all primaries
        #             or the the common ancestor is the first non-primary
        if level == len(self.stack) - 1:
            return self.stack[level].unshift(self.stack[level].primaries)
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
        reduced = jnp.logical_or(
                jnp.logical_or(reduced_1st[0], reduced_1st[3]),
                jnp.logical_or(reduced_1st[1], reduced_1st[2]))
        suppressions = self.stack[level + 1].unshift(reduced, 2)
        allowed = jnp.logical_not(suppressions)
        candidates = jnp.logical_and(self.stack[level].primaries, allowed)
        return self.stack[level].unshift(candidates)

    @jax.jit(static_argnames=['candidates'])
    def nominate(self, candidates=16):
        values = jnp.ravel(self.stack[-1].bounds.metric)
        suppressors = range(len(self.stack) - 1, 0, -1)
        for i, layer in zip(suppressors, self.stack[-2::-1]):
            nominees = jnp.where(self.nms(i), layer.bounds.metric, 0)
            values = jnp.concatenate((values, jnp.ravel(nominees)))
        values, indices = jax.lax.top_k(values, candidates)
        return values, indices

    def flattened(self, f):
        res = None
        for layer in self.stack[-1::-1]:
            adding = f(layer).reshape(layer.counts.size, -1)
            if res is None:
                res = adding
            else:
                res = jnp.concatenate((res, adding))
        return res

    def bound(self, candidates):
        values, indices = self.nominate(candidates)
        return values, self.flattened(lambda x: x.bounds.bounds)[indices]

    def stat(self, candidates):
        # TODO: avoid flattening the stats to enable dead op removal
        values, indices = self.nominate(candidates)
        return FlatStat(values, self.flattened(lambda x: x.stat())[indices])

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["confidences", "stats"], meta_fields=[])
@dataclass
class FlatStat:
    confidences: any
    stats: any

    order = Bins.stats

    def __post_init__(self, *a, **kw):
        assert self.stats.shape[-1] == len(self.order) * 2
        assert self.stats.ndim == 2

    @cached_property
    def mean(self):
        return SiftedMeans(*jnp.unstack(
            self.stats[:, :len(self.order)], axis=-1))

    @cached_property
    def var(self):
        return namedtuple('Variances', self.order)(*jnp.unstack(
            self.stats[:, len(self.order):], axis=-1))

    @cached_property
    def std(self):
        return namedtuple('StandardDeviations', self.order)(*map(
            jnp.sqrt, self.var))

    def offset(self, origin):
        return self.__class__(self.stats.at[0, :2].subtract(origin))

class SiftedMeans(namedtuple('Means', FlatStat.order)):
    @property
    def centers(self):
        return jnp.stack((self.center_0th, self.center_1st), -1)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["sum", "sum_sq"], meta_fields=[])
@dataclass
class BinStat:
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
class BinSum:
    sum: any

    def merge(self, offset):
        assert bin_win[0] == bin_win[1] == (2, 2)
        shape = [i - 1 for i in self.sum.shape]
        shifted = jax.lax.dynamic_slice(self.sum, offset, shape)
        return BinSum(jax.lax.reduce_window(shifted, 0., jax.lax.add, *bin_win))

# TODO: adjust according to confidences?
@partial(
        jax.tree_util.register_dataclass,
        data_fields=[
            "config", "depth", "origin", "theta", "depth_mean", "depth_std",
            "radius_mean", "radius_std", "confidences"],
        meta_fields=["distance"])
@dataclass
class AliasedRay:
    config: any
    depth: any
    origin: any
    theta: any
    distance: any
    depth_mean: any
    depth_std: any
    radius_mean: any
    radius_std: any
    confidences: any

    def __post_init__(self, *a, **kw):
        assert self.origin.ndim == 2 and self.origin.shape[-1] == 2

    @classmethod
    def from_binstats(cls, depth, stats, rays, distance):
        return cls(
                depth.config, depth.depth, stats.mean.centers,
                jnp.linspace(0, 2 * jnp.pi, rays, endpoint=False), distance,
                stats.mean.depth, stats.std.depth,
                stats.mean.radius, stats.std.radius, stats.confidences)

    @property
    def candidates(self):
        return self.origin.shape[0]

    @cached_property
    def steps(self):
        offset = self.theta + jnp.pi / 4
        quad = jnp.astype(offset // (jnp.pi / 2), jnp.int8) % 4
        flip = jnp.where(quad % 3, -1, 1)
        slope = jnp.tan(offset % (jnp.pi / 2) - jnp.pi / 4) * flip
        sign, axis = jnp.where(quad // 2, 1, -1)[None, :], quad % 2
        bias = sign * self.origin[:, axis] % 1
        fp = self.origin[:, 1 - axis] + bias * slope
        counting = jnp.int16(self.origin[:, axis] - bias * sign)
        steps = jnp.arange(self.distance, dtype=jnp.int16)[None, None, :]
        indices = counting[..., None] - sign[..., None] * steps
        frac = fp[..., None] + slope[None, :, None] * steps
        lo = jnp.int16(frac)
        hi = lo + 1
        axis = axis[None, None, :, None]
        lo = jnp.where(axis, jnp.stack((lo, indices)), jnp.stack((indices, lo)))
        hi = jnp.where(axis, jnp.stack((hi, indices)), jnp.stack((indices, hi)))
        return (lo, hi)

    def adjacent(self):
        lo, hi = self.steps
        return (
            self.depth.at[*lo].get(wrap_negative_indices=False, mode="fill"),
            self.depth.at[*hi].get(wrap_negative_indices=False, mode="fill"))

    def occludes(self, series):
        hi = self.depth_mean + self.config.alpha * self.depth_std
        lo = self.depth_mean - self.config.beta * self.depth_std
        lim = self.radius_mean + self.config.chi * self.radius_std

        lo, hi, lim = lo[:, None, None], hi[:, None, None], lim[:, None, None]
        near, far = series[..., :-1], series[..., 1:]
        valid = jnp.logical_and(near >= lo, near < hi)
        lowers = far - near < -self.config.delta * self.depth_std
        expected = jnp.arange(self.distance - 1)[None, None] < lim
        edge = jnp.logical_and(expected, jnp.logical_and(valid, lowers))
        edge = jnp.concatenate(
                (jnp.zeros(edge.shape[:2] + (1,), dtype=jnp.bool), edge), -1)
        return jnp.argmax(edge, -1) - 1

    # (2, self.candidates, self.theta.size)
    @cached_property
    def poi(self):
        # TODO: remove duplicates?
        x0, x1 = self.steps
        y0, y1 = self.adjacent()
        z0, z1 = self.occludes(y0), self.occludes(y1)
        dims = jax.lax.GatherDimensionNumbers((2,), (), (2,), (0, 1), (0, 1))
        w0 = jnp.stack((
            jax.lax.gather(x0[0], z0[..., None], dims, (1, 1, 2)),
            jax.lax.gather(x0[1], z0[..., None], dims, (1, 1, 2))))
        w1 = jnp.stack((
            jax.lax.gather(x1[0], z1[..., None], dims, (1, 1, 2)),
            jax.lax.gather(x1[1], z1[..., None], dims, (1, 1, 2))))
        w = jnp.sum(w0 + w1, -1) / 4
        z = jnp.logical_or(z0 < 0, z1 < 0)[None]
        return jnp.where(z, jnp.array([-1, -1])[:, None, None], w)

    @property
    def unsized(self):
        return self.poi[jnp.where(self.poi >= 0)].reshape(2, -1)

    @cached_property
    def oob(self):
        return self.poi[:1] < 0

    @cached_property
    def count(self):
        return self.theta.size - jnp.sum(self.oob[0], -1)

    # (3, self.candidates)
    def loss(self, x):
        x = x.reshape(3, -1)
        d = jnp.sqrt(jnp.sum((x[:2, :, None] - self.poi) ** 2, 0))
        # TODO: L2 for optimizer?
        shrinkage = (x[2] - self.radius_mean) ** 2 / self.radius_std
        # sqrt(count) in the regularization term is a guess
        return (
                jnp.sum(jnp.sqrt(jnp.sum(jnp.where(
                    self.oob, 0, (x[2, :, None] - d) ** 2), -1)) / self.count) +
                self.config.eta * jnp.sum(shrinkage / jnp.sqrt(self.count))) / \
                        self.candidates

    # TODO: apparently refit for im8 doesn't converge
    @jax.jit
    def fit(self):
        init = jnp.concatenate((self.origin, self.radius_mean[:, None]), -1).T
        res = minimize(self.loss, jnp.ravel(init), method="BFGS")
        return Circles(
                self.config, *jnp.unstack(res.x.reshape(3, -1)),
                self.confidences, self.count, res.fun[None], res.success[None])

    batched = (
            "origin", "depth_mean", "depth_std", "radius_mean", "radius_std",
            "confidences")
    def split(self):
        res = [None] * self.candidates
        for i in range(self.candidates):
            res[i] = self.__class__(
                    self.config, self.depth,
                    theta=self.theta, distance=self.distance,
                    **{k: getattr(self, k)[i:i + 1] for k in self.batched})
        return OptBatch(self.config, res)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["config", "candidates"], meta_fields=[])
@dataclass
class OptBatch:
    config: any
    candidates: any

    def fit(self):
        res = [candidate.fit() for candidate in self.candidates]
        return Circles(self.config, *[
            jnp.concat([getattr(i, field) for i in res])
            for field in Circles._fields[1:]])

class Circles(namedtuple("Circles", (
        "config", "center_0th", "center_1st", "radius",
        "granularity", "samples", "loss", "converged"))):
    @property
    def bounds(self):
        return jnp.array([
                self.center_0th - self.radius, self.center_1st - self.radius,
                self.center_0th + self.radius, self.center_1st + self.radius])

    def readable(self):
        success = jnp.any(self.converged).item()
        it = jnp.nonzero(self.converged)[0] if success else \
                jnp.nonzero(~jnp.isnan(self.loss))[0]
        for i in it:
            y, x = self.center_0th[i].item(), self.center_1st[i].item()
            r, granularity = self.radius[i].item(), self.granularity[i].item()
            loss, samples = self.loss[i].item(), self.samples[i].item()
            print(
                    f"[{i:2d}] ({x:6.1f}, {y:6.1f}) radius: {r:6.2f} "
                    f"score: {granularity:.3e} "
                    f"-{{n: {samples:2d}}}-> loss: {loss:.3e}")
        print("---- converged" if success else "^^^^ failed")

    @cached_property
    def confidences(self):
        # TODO: adjust with loss & samples
        return self.granularity

examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"

im4 = Raster.file(examples_dir / "IMG_0004.HEIC", cache_dir / "out4.npy")
im5 = Raster.file(examples_dir / "IMG_0005.HEIC", cache_dir / "out5.npy")
im7 = Raster.file(examples_dir / "IMG_0007.HEIC", cache_dir / "out7.npy")
im8 = Raster.file(examples_dir / "IMG_0008.HEIC", cache_dir / "out8.npy")

# import matplotlib.pyplot as plt
for im in [im4, im5, im7, im8]:
    im.refit().readable()
    # im.draw_refit()
    # plt.show()
