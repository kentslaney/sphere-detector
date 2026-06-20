import sys, pathlib, math, inspect
from functools import cached_property, partial
from dataclasses import dataclass
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax.scipy.signal import correlate2d
from jax.scipy.optimize import minimize

from .utils import lazy_default, jax_limit_cache, kron_bool, patch_tag, Image
from .depth import Da2

@partial(
        jax.tree_util.register_dataclass,
        data_fields=[
            "alpha", "beta", "gamma", "delta", "eps", "chi", "phi", "mu", "nu"],
        meta_fields=[
            "resolution", "candidates", "rays", "extent", "subdivisions",
            "early_nms"])
@dataclass(kw_only=True)
class Config:  # hyperparameters
    # Raster
    resolution: any = (392, 518)  # downsampling resolution
    subdivisions: any = 8  # minimum number of cells per dimension
    candidates: any = 16  # number of curves to trace
    rays: any = 64  # number of 2d points to fit
    extent: any = 8  # minimum number of radii per diagonal
    # TODO: MIL integration test for early NMS (helps resolution invariance)
    early_nms: any = True  # Seives.nms bypass switch

    # Bounds
    eps: any = 0.1  # density stabilization coefficient
    # patern-matched, not derived
    phi: any = (1 + math.sqrt(5)) / 2  # metric dimensionality

    # AliasedRay
    alpha: any = 0.0  # standard deviations above mean for ray start depth
    # mean height vs center: 2 / 3 * r and standard deviation: sqrt(2) / 6 * r
    beta: any = 3.0  # standard deviations below mean for ray start depth
    gamma: any = 0.2  # interpolation value between median and mean for w
    delta: any = 0.5  # threshold in standard deviations for ray depth jump
    chi: any = 0.5  # standard deviations above initial mean radius to look

    # Surface
    # TODO: tune
    mu: any = 1.0  # edge RMSE coefficient (remember via lower case shape)
    nu: any = 1.0  # depth slice RMSE coefficient

    depth_checkpoint = "vits"

    @property
    def diag(self):
        assert self.resolution is not None
        return int(sum(i ** 2 for i in self.resolution) ** 0.5)

    @property
    def distance(self):
        return self.diag // self.extent

class Raster:  # image wrapper non-serializable for JAX
    config = Config()
    model = Da2

    def data(self, *a, **kw):
        return Depth(self.config, *a, **kw)

    @classmethod
    def file(cls, path, npy=None, **kw):
        im, cache = Image.open(path), None
        if npy is not None:
            npy = pathlib.Path(npy)
            if npy.exists():
                cache = jnp.load(npy)
                target = kw.get(
                        "resolution", getattr(cls.config, "resolution", None))
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
        return self.model(self.config.depth_checkpoint)(self.cropped())

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
        self.config = self.config.__class__(**kw)
        if cache is not None:
            self.cache = cache

    @cached_property
    def seives(self):
        return Seives.create(self.depth.binned())

    @lazy_default(candidates=lambda self, *a, **kw: self.config.candidates)
    @jax_limit_cache('candidates')
    def stat(self, candidates):
        return self.seives.stat(candidates)

    @lazy_default(candidates=lambda self, *a, **kw: self.config.candidates)
    @jax_limit_cache('candidates', '.depth', '.theta')
    def opt(self, candidates):
        pred = self.stat(candidates)
        return AliasedRay.from_binstats(
                self.depth, pred, self.config.rays, self.config.distance)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["config", "depth"], meta_fields=[])
@dataclass
class Depth:  # JAX depth data entry point
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
    def w(self):
        da2, db2 = self.rotated[..., 0, 0], self.rotated[..., 1, 1]
        ge1 = jnp.where(self.inwards, da2 / db2, 1)
        return jnp.sqrt(ge1 - 1) / self.norm

    @cached_property
    def masked(self):
        return jnp.where(self.inwards[..., None], self.centers, -1)

    def stat(self, indices):
        return BinStat(
                indices.stat(self.masked[..., 0]),
                indices.stat(self.masked[..., 1]),
                indices.stat(self.radii),
                indices.stat(self.depth),
                indices.stat(self.w))

    @jax.jit
    def binned(self):
        indices = Scatter2d.create(self.depth.shape, self.masked)
        counts = indices.scatter('add', 1, 0)
        bounds = Boundary.create(
            -indices.scatter('min', self.coord[..., 0], self.depth.shape[0]),
            -indices.scatter('min', self.coord[..., 1], self.depth.shape[1]),
            indices.scatter('max', self.coord[..., 0], -1),
            indices.scatter('max', self.coord[..., 1], -1))
        bounds = Bounds(
                self.config, self.depth.shape, 1, bounds, win_prep(counts),
                jnp.array([0, 0]), jnp.array([0, 0]), None)
        return Bins(0, bounds, self.stat(indices))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["indices"], meta_fields=["shape"])
@dataclass
class Scatter2d:  # memory shuffler
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
        return fn(jnp.ravel(value), mode="drop")

    def stat(self, values):
        values = jnp.ravel(values)
        return BinAcc(
                BinSum(self.scatter('add', values, 0.)),
                BinSum(self.scatter('add', values ** 2, 0.)))

bin_win = ((2, 2), (2, 2))  # dimensions, strides
# MIL max_pool only supports float
win_prep = jnp.float32 # lambda x: x

# tracks the extrema per image dimension
class Boundary(namedtuple("Bound", ("lo_0th", "lo_1st", "hi_0th", "hi_1st"))):
    def area(self):
        # initial state has difference > 1 for dimension size > 0
        d0 = jnp.maximum(0, self.hi_0th + self.lo_0th + 1)
        d1 = jnp.maximum(0, self.hi_1st + self.lo_1st + 1)
        return d0 * d1

    @classmethod
    def create(cls, *a):
        return cls(*map(win_prep, a))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["config", "bounds", "counts", "origin", "offset"],
        meta_fields=["shape", "scale", "upscale"])
@dataclass
class Bounds:  # Tracks the scatter density and boundaries for the sources
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
        inits = [-x for x in self.shape] + [-1, -1]
        inits = tuple(map(self.bounds.lo_0th.dtype.type, inits))
        reduced = Boundary(
                f(self.bounds[0], inits[0], jax.lax.max, *bin_win),
                f(self.bounds[1], inits[1], jax.lax.max, *bin_win),
                f(self.bounds[2], inits[2], jax.lax.max, *bin_win),
                f(self.bounds[3], inits[3], jax.lax.max, *bin_win))
        counts = f(self.counts, 0, jax.lax.add, *bin_win)
        assert bin_win[0] == bin_win[1] == (2, 2)
        return self.__class__(
                self.config, self.shape, self.scale * bin_win[1][0], reduced,
                counts, self.origin, self.offset, self.upscale)

    @cached_property
    def metric(self):
        # counts ~ area implies as proportional:
        #     counts ** phi / area / scale ** (2 * (phi - 1))
        #     (area / scale ** 2) ** (phi - 1)
        # which is resolution invariant
        areas, total = self.bounds.area(), self.counts.size
        stabilization = self.config.eps * total * self.scale ** 2
        return (self.counts ** self.config.phi) / (areas + stabilization) / \
                self.scale ** (2 * (self.config.phi - 1))

    def __getitem__(self, key):
        offset = jnp.array([i.start for i in key])
        origin = offset * self.scale + self.origin
        bound = Boundary(*(x[key] for x in self.bounds))
        return self.__class__(
                self.config, self.shape, self.scale, bound,
                self.counts[key], origin, offset, self.counts.shape)

    @jax.jit
    def sifted(self):
        hi, val = None, None
        for i in self.off:
            shift = self[i].merge()
            total = jnp.sum(shift.metric)
            if val is None:
                hi, val = total, shift
            else:
                pred = total > hi
                hi = jax.tree.map(lambda *a: jnp.where(pred, *a), total, hi)
                val = jax.tree.map(lambda *a: jnp.where(pred, *a), shift, val)
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

# stat fields for scatter dataflow
StatContainer = namedtuple("Stat", (
        'center_0th', 'center_1st', 'radius', 'depth', 'w'))
class BinStat(StatContainer):  # BinAcc for each stat field
    def merge(self, bounds):
        return self.__class__(*(x.merge(bounds.offset) for x in self))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["bounds", "stats"],
        meta_fields=["level"])
@dataclass
class Bins:  # wraps boundary tracking with stat tracking
    level: any
    bounds: any
    stats: any

    @property
    def counts(self):
        return self.bounds.counts

    @property
    def shape(self):
        return self.counts.shape

    def sifted(self):
        bounds = self.bounds.sifted()
        return self.__class__(self.level + 1, bounds, self.stats.merge(bounds))

    @cached_property
    def primaries(self):
        assert bin_win[0] == bin_win[1] == (2, 2)
        centers = self.bounds.centers
        primary_0th = self.stats.center_0th.mean(self.counts) > centers[..., 0]
        primary_1st = self.stats.center_1st.mean(self.counts) > centers[..., 1]
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
            x_prefix = jnp.concat((full(i, 1), x), axis=i)
            x_postfix = jnp.concat((x, full(i, 1)), axis=i)
            x = jnp.where(prefix == 1, x_prefix, x_postfix)
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

    def stat(self):
        summaries = [x.summary(self.counts) for x in self.stats]
        return Summary(*(
                StatContainer(*(getattr(x, i) for x in summaries))
                for i in Summary._fields))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["config", "stack"], meta_fields=[])
@dataclass
class Seives:  # Feature Pyramid
    config: any
    stack: any

    @classmethod
    def create(cls, base, layers=None):
        layers = int(
                math.log2(base.counts.size) / 2 -
                math.log2(base.bounds.config.subdivisions)) \
                    if layers is None else layers - 1
        out = [base]
        for _ in range(layers):
            out.append(out[-1].sifted())
        return cls(base.bounds.config, tuple(out))

    def ruler(self, axis):  # OEIS A001511
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
        if not self.config.early_nms:
            return True

        assert 0 < level < len(self.stack), "layer must store primaries"
        # Only primaries can be candidates, subject to the filter:
        #     Primaries block out 1 level down surroundings
        #         if the primary's upwards path to the common ancestor
        #             is all primaries
        #             or the the common ancestor is the first non-primary
        if level == len(self.stack) - 1:
            return self.stack[level].unshift(self.stack[level].primaries)
        kernel_masks = [
            [0x8000, 0x0008, 0x1000, 0x0001],
            [0x0888, 0x8880, 0x0111, 0x1110],
            [0x7000, 0x0007, 0xe000, 0x000e],
            [0x0117, 0x7110, 0x088e, 0xe880]]
        kernels = [[
            jnp.array([[
                bool(((j >> (n * 4)) & 0xF) & (1 << m))
                for n in range(3, -1, -1)] for m in range(3, -1, -1)])
            for j in i] for i in kernel_masks]
        strides = [
                (lambda *a: (lambda x: jax.lax.slice(x, a, x.shape, (2, 2))))(
                    i, j) for i in range(2) for j in range(2)]
        masks = [None] * 3
        out = ([], [], [], [])
        for a, b in zip(strides, kernels[3]):
            out[3].append(kron_bool(a(self.stack[level + 1].primaries), b))
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
                mask = a(self.pyramids[level]) >= bound
                mask = jnp.logical_and(mask, a(self.stack[level + 1].primaries))
                out[i].append(kron_bool(mask, b))
        ll_all = jnp.logical_or(
            jnp.logical_or(out[0][0], out[1][0]),
            jnp.logical_or(out[2][0], out[3][0]))
        lh_all = jnp.logical_or(
            jnp.logical_or(out[0][1], out[1][1]),
            jnp.logical_or(out[2][1], out[3][1]))
        hl_all = jnp.logical_or(
            jnp.logical_or(out[0][2], out[1][2]),
            jnp.logical_or(out[2][2], out[3][2]))
        hh_all = jnp.logical_or(
            jnp.logical_or(out[0][3], out[1][3]),
            jnp.logical_or(out[2][3], out[3][3]))

        ll = jnp.concatenate((
            ll_all[1:, :],
            jnp.zeros((1, ll_all.shape[1]), dtype=bool)), 0)
        ll = jnp.concatenate((
            ll[:, 1:],
            jnp.zeros((ll.shape[0], 1), dtype=bool)), 1)

        lh = jnp.concatenate((
            lh_all[1:, :],
            jnp.zeros((1, lh_all.shape[1]), dtype=bool)), 0)
        lh = jnp.concatenate((
            jnp.zeros((lh.shape[0], 1), dtype=bool),
            lh[:, :-1]), 1)

        hl = jnp.concatenate((
            jnp.zeros((1, hl_all.shape[1]), dtype=bool),
            hl_all[:-1, :]), 0)
        hl = jnp.concatenate((
            hl[:, 1:],
            jnp.zeros((hl.shape[0], 1), dtype=bool)), 1)

        hh = jnp.concatenate((
            jnp.zeros((1, hh_all.shape[1]), dtype=bool),
            hh_all[:-1, :]), 0)
        hh = jnp.concatenate((
            jnp.zeros((hh.shape[0], 1), dtype=bool),
            hh[:, :-1]), 1)

        reduced = jnp.logical_or(
            jnp.logical_or(ll, hh),
            jnp.logical_or(lh, hl))
        suppressions = self.stack[level + 1].unshift(reduced, 2)
        allowed = jnp.logical_not(suppressions)
        candidates = jnp.logical_and(self.stack[level].primaries, allowed)
        return self.stack[level].unshift(candidates)

    @jax.jit(static_argnames=['candidates'])
    def nominate(self, candidates):
        values = jnp.ravel(self.stack[-1].bounds.metric)
        suppressors = range(len(self.stack) - 1, 0, -1)
        for i, layer in zip(suppressors, self.stack[-2::-1]):
            nominees = jnp.where(self.nms(i), layer.bounds.metric, 0)
            values = jnp.concatenate((values, jnp.ravel(nominees)))
        values, indices = jax.lax.top_k(values, candidates)
        return values, indices

    def flattened(self, indices, f):
        res = None
        for layer in self.stack[-1::-1]:
            adding = jax.tree.map(jnp.ravel, f(layer))
            if res is None:
                res = adding
            else:
                res = jax.tree.map(lambda *x: jnp.concatenate(x), res, adding)
        return jax.tree.map(lambda x: x[indices], res)

    def bound(self, candidates):
        values, indices = self.nominate(candidates)
        return values, self.flattened(indices, lambda x: x.bounds.bounds)

    def stat(self, candidates):
        values, indices = self.nominate(candidates)
        return FlatStat(values, self.flattened(indices, lambda x: x.stat()))

# processed accumulator outputs for a single data field
Summary = namedtuple("SufficientStat", ("mean", "var"))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["confidence", "stats"], meta_fields=[])
@dataclass
class FlatStat:  # features flattened across pyramid layers and image dimensions
    confidence: any
    stats: any

    @property
    def mean(self):
        return SiftedMeans(*self.stats.mean)

    @property
    def var(self):
        return self.stats.var

    @cached_property
    def std(self):
        return StatContainer(*map(jnp.sqrt, self.var))

    @cached_property
    def mode(self):
        # assumes that the stats are a gamma distribution (right skew)
        theta = tuple(i / j for i, j in zip(self.var, self.mean))
        alpha = tuple(i / j for i, j in zip(self.mean, theta))
        return StatContainer(*(
            jnp.where(i >= 1, (i - 1) * j, 0) for i, j in zip(alpha, theta)))

class SiftedMeans(StatContainer):  # convenience property for spacial data
    @property
    def centers(self):
        return jnp.stack((self.center_0th, self.center_1st), -1)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["sum", "sum_sq"], meta_fields=[])
@dataclass
class BinAcc:  # holds intermediate data accumulation state
    sum: any
    sum_sq: any

    def merge(self, offset):
        return self.__class__(self.sum.merge(offset), self.sum_sq.merge(offset))

    def mean(self, counts):
        return self.sum.sum / counts

    def summary(self, counts):
        mean = self.mean(counts)
        return Summary(mean, (
            self.sum_sq.sum - self.sum.sum * mean) / (counts - 1))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["sum"], meta_fields=[])
@dataclass
class BinSum:  # holds sum for 2d binary merges
    sum: any

    def merge(self, offset):
        assert bin_win[0] == bin_win[1] == (2, 2)
        shape = [i - 1 for i in self.sum.shape]
        shifted = jax.lax.dynamic_slice(self.sum, offset, shape)
        return BinSum(jax.lax.reduce_window(shifted, 0., jax.lax.add, *bin_win))

@partial(
        jax.tree_util.register_dataclass,
        data_fields=[
            "config", "depth", "origin", "theta", "depth_mean", "depth_std",
            "radius_mean", "radius_std", "w"],
        meta_fields=["distance"])
@dataclass
class AliasedRay:  # represents the curve centers to fit from
    config: any
    depth: any
    origin: any
    theta: any
    distance: any
    depth_mean: any
    depth_std: any
    radius_mean: any
    radius_std: any
    w: any

    def __post_init__(self, *a, **kw):
        assert self.origin.ndim == 2 and self.origin.shape[-1] == 2

    @classmethod
    def from_binstats(cls, depth, stats, rays, distance):
        w = (
                stats.mean.w * depth.config.gamma +
                stats.mode.w * (1 - depth.config.gamma))
        return cls(
                depth.config, depth.depth, stats.mean.centers,
                jnp.linspace(0, 2 * jnp.pi, rays, endpoint=False), distance,
                stats.mean.depth, stats.std.depth,
                stats.mean.radius, stats.std.radius, w)

    @property
    def candidates(self):
        return self.origin.shape[0]

    @cached_property
    def steps(self):
        offset = self.theta + jnp.pi / 4
        quad = jnp.astype(offset // (jnp.pi / 2), jnp.int32) % 4
        flip = jnp.where(quad % 3, -1, 1)
        slope = jnp.tan(offset % (jnp.pi / 2) - jnp.pi / 4) * flip
        sign, axis = jnp.where(quad // 2, 1, -1)[None, :], quad % 2
        bias = sign * self.origin[:, axis] % 1
        fp = self.origin[:, 1 - axis] + bias * slope
        counting = jnp.int32(self.origin[:, axis] - bias * sign)
        steps = jnp.arange(self.distance, dtype=jnp.int32)[None, None, :]
        indices = counting[..., None] - sign[..., None] * steps
        frac = fp[..., None] + slope[None, :, None] * steps
        lo = jnp.int32(frac)
        hi = lo + 1
        axis = axis[None, None, :, None]
        lo = jnp.where(axis, jnp.stack((lo, indices)), jnp.stack((indices, lo)))
        hi = jnp.where(axis, jnp.stack((hi, indices)), jnp.stack((indices, hi)))
        return (lo, hi)

    @cached_property
    def adjacent(self):
        lo, hi = self.steps
        oob = jnp.array(self.config.resolution)[:, None, None, None]
        oob_lo = jnp.logical_or(lo < 0, lo >= oob)
        oob_lo = jnp.logical_and(oob_lo[0], oob_lo[1])
        oob_hi = jnp.logical_or(hi < 0, hi >= oob)
        oob_hi = jnp.logical_and(oob_hi[0], oob_hi[1])
        lo = jnp.where(oob_lo, jnp.nan, self.depth[*lo])
        hi = jnp.where(oob_hi, jnp.nan, self.depth[*hi])
        return (lo, hi)

    @cached_property
    def occludes(self):
        res = [None, None]
        for i, series in enumerate(self.adjacent):
            hi = self.depth_mean + self.config.alpha * self.depth_std
            lo = self.depth_mean - self.config.beta * self.depth_std
            lim = self.radius_mean + self.config.chi * self.radius_std

            lo, hi = lo[:, None, None], hi[:, None, None]
            lim = lim[:, None, None]
            near, far = series[..., :-1], series[..., 1:]
            valid = jnp.logical_and(near >= lo, near < hi)
            bound = -self.config.delta * self.depth_std[:, None, None]
            lowers = far - near < bound
            expected = jnp.arange(self.distance - 1)[None, None] < lim
            edge = jnp.logical_and(expected, jnp.logical_and(valid, lowers))
            res[i] = jax.lax.reduce_min(jnp.where(
                edge, jnp.arange(self.distance - 1)[None, None],
                jnp.array(self.distance)[(None,) * 3]), (2,))
        return tuple(res)

    # (2, self.candidates, self.theta.size)
    @cached_property
    @patch_tag("mps_gather_shape")
    def points(self):
        (x0, x1), (y0, y1) = self.steps, self.occludes
        z0, z1 = y0[..., None], y1[..., None]
        dims = jax.lax.GatherDimensionNumbers((2,), (), (2,), (0, 1), (0, 1))

        w0 = jnp.stack((
            jax.lax.gather(x0[0], z0 + 0, dims, (1, 1, 1)),
            jax.lax.gather(x0[1], z0 + 0, dims, (1, 1, 1)),
        ))
        w1 = jnp.stack((
            jax.lax.gather(x0[0], z0 + 1, dims, (1, 1, 1)),
            jax.lax.gather(x0[1], z0 + 1, dims, (1, 1, 1)),
        ))
        w2 = jnp.stack((
            jax.lax.gather(x1[0], z1 + 0, dims, (1, 1, 1)),
            jax.lax.gather(x1[1], z1 + 0, dims, (1, 1, 1)),
        ))
        w3 = jnp.stack((
            jax.lax.gather(x1[0], z1 + 1, dims, (1, 1, 1)),
            jax.lax.gather(x1[1], z1 + 1, dims, (1, 1, 1)),
        ))

        return Trace(
                (w0 + w1 + w2 + w3)[..., 0] / 4,
                jnp.logical_and(y0 < self.distance, y1 < self.distance))

    @cached_property
    def fit(self):
        samples = jnp.sum(self.points.valid, 1)
        (y_, x_, r), (y, x) = self.points.fit(), jnp.unstack(self.points.points)
        rmse = jnp.sqrt(jnp.mean((
            jnp.sqrt((x - x_[:, None]) ** 2 + (y - y_[:, None]) ** 2)
            - r[:, None]) ** 2, 1, where=self.points.valid))
        return Circles(y_, x_, r, samples, rmse)

    @cached_property
    def samples(self):
        res = [None, None]
        for i, series in enumerate(self.steps):
            y, x = jnp.unstack(series)
            outside = self.fit.radius[:, None, None] ** 2 < (
                    (self.fit.center_0th[:, None, None] - y) ** 2 +
                    (self.fit.center_1st[:, None, None] - x) ** 2)
            oob = jnp.logical_or(outside, jnp.logical_or(
                    y >= self.config.resolution[0],
                    x >= self.config.resolution[1]))
            out = jax.lax.reduce_min(jnp.where(
                oob, jnp.arange(self.distance)[None, None],
                jnp.array(self.distance)[(None,) * 3]), (2,))
            res[i] = jnp.where(
                    self.occludes[i] < self.distance, self.occludes[i], out)
        return jnp.concat(res, axis=1)

    @cached_property
    def valid(self):
        return jnp.arange(self.distance)[None, None] < self.samples[..., None]

    @cached_property
    def surface(self):
        x = jnp.arange(self.distance)[None, None] - Surface.x_c
        r = self.fit.radius[:, None, None] - Surface.x_c

        y = jnp.concat(self.adjacent, axis=1) * self.w[:, None, None]
        y_c = jnp.mean(
            y - jnp.sqrt(r ** 2 - jnp.minimum(x, r) ** 2), (1, 2),
            where=self.valid)

        residuals = jnp.sqrt(x ** 2 + (y - y_c[:, None, None]) ** 2) - r
        bias = self.skew(residuals)
        residuals = jnp.sqrt(x ** 2 + (y + bias - y_c[:, None, None]) ** 2) - r
        rmse = jnp.sqrt(jnp.mean(residuals ** 2, (1, 2), where=self.valid))
        rmse = jnp.where(jnp.isnan(rmse), jnp.inf, rmse)
        return Surface(self.config, self.fit, y_c / self.w, rmse, bias)

    def skew(self, residuals):
        yx = jnp.concat(self.steps, axis=2)
        data = jnp.concat((yx, residuals[None]))
        center = jnp.mean(data, (2, 3), where=self.valid, keepdims=True)
        centered = jnp.where(self.valid, data - center, 0)
        data = centered.reshape(
                (3, self.candidates, 2 * self.config.rays * self.distance))
        data = jnp.transpose(data, (1, 0, 2))
        data_t = jnp.transpose(data, (0, 2, 1))
        cov = data @ data_t / jnp.sum(self.samples, 1)[:, None, None]
        z = jnp.linalg.det(cov[:, :2, :2])
        y = jnp.linalg.det(cov[:, :2, 1:]) / z
        x = jnp.linalg.det(jnp.stack((cov[:, :2, 0], cov[:, :2, 2]), 2)) / z
        return jnp.sum(centered[:2] * jnp.vstack((y, x))[..., None, None], 0)

    @jax.jit
    def predict(self):
        surface = self.surface
        return surface.confidence[surface.order], surface.bounds[surface.order]

class Trace(namedtuple("Trace", ("points", "valid"))):  # masked 2d drop-offs
    @jax.jit
    def fit(self):
        mean = jnp.mean(self.points, axis=2, where=self.valid)
        y, x = jnp.unstack(
                jnp.where(self.valid, self.points - mean[..., None], 0), axis=0)

        a = jnp.stack([2 * y, 2 * x, self.valid], axis=2)
        b = x ** 2 + y ** 2

        at = jnp.transpose(a, (0, 2, 1))
        ata, atb = at @ a, (at @ b[..., None])[..., 0]
        cols = jnp.unstack(ata, axis=2)
        cramer = lambda idx: jnp.stack(
                cols[:idx] + (atb,) + cols[idx + 1:], axis=2)

        det = jnp.linalg.det(ata)
        # TODO: I think compute graph precision is via MIL pass
        eps = jnp.finfo(det.dtype).eps
        det = jnp.where(jnp.abs(det) < eps, eps, det)

        a_, b_, c = (jnp.linalg.det(cramer(i)) / det for i in range(3))
        return a_ + mean[0], b_ + mean[1], jnp.sqrt(c + a_ ** 2 + b_ ** 2)

class Circles(namedtuple("Circles", (  # 2d fit results
        "center_0th", "center_1st", "radius", "samples", "rmse"))):
    @property
    def bounds(self):
        return jnp.array([
                self.center_0th - self.radius, self.center_1st - self.radius,
                self.center_0th + self.radius, self.center_1st + self.radius]).T

    @property
    def valid(self):
        return self.samples > 3  # avoid vacantly 0 RMSE

class Surface(namedtuple("Surface", (  # depth slice along AliasedRay
        "config", "edge", "center_2nd", "rmse", "skew"))):
    # mean distance from center of a square's perimeter (line not angle)
    x_c = -(math.sqrt(2) + math.log(1 + math.sqrt(2))) / 4

    @property
    def revolutions(self):
        return jnp.concat((jnp.linspace(0, 1, self.config.rays, False),) * 2)

    @property
    def bounds(self):
        return self.edge.bounds

    @property
    def loss(self):
        # patern-matched, not derived
        return jnp.where(
                self.edge.valid, jnp.sqrt(self.config.rays) / self.edge.samples,
                self.config.rays)

    @cached_property
    def order(self):
        return jnp.argsort(self.confidence, descending=True, stable=False)

    @cached_property
    def confidence(self):
        res = jnp.exp(-(
            self.loss +
            self.config.nu * self.rmse +
            self.config.mu * self.edge.rmse
        ))
        return jnp.where(jnp.isnan(res), 0, res)
