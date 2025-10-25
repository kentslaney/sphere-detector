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

@jax.jit
def horizon(cylindrical):
    assert 1 < cylindrical.shape[-1] < 4
    topk = cylindrical.shape[-2]
    halved = jnp.atan2(cylindrical[..., 0], cylindrical[..., 1]) / 2
    transformed = jnp.stack((jnp.cos(halved), jnp.sin(halved)), -1) * \
            jnp.linalg.norm(cylindrical, axis=-1, keepdims=True)
    spread = jnp.broadcast_to(
            jnp.eye(topk)[(None,) * (cylindrical.ndim - 2)],
            cylindrical.shape[:-2] + (topk, topk))
    expanded_T = jnp.concatenate(
            (transformed, cylindrical[..., 2:], spread), -1)
    expanded = jnp.moveaxis(expanded_T, -2, -1)
    return jnp.linalg.slogdet(expanded_T @ expanded).logabsdet

class Raster:
    model = Da2('vits')
    metric = staticmethod(horizon)
    target = None
    f_35mm = None

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
        return Depth(self.cache)

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
        self.full = im
        if cache is not None:
            self.cache = cache

@partial(
        jax.tree_util.register_dataclass, data_fields=["depth"], meta_fields=[])
@dataclass
class Depth(object):
    depth: any

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

    def rasterize(self, continuous):
        assert continuous.shape[-1] == 2
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
        out = jnp.zeros(self.depth.shape, dtype=self.depth.dtype)
        floored = jnp.int32(jnp.floor(continuous))
        remainder = continuous - floored
        for offset in jnp.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]):
            filling = floored + offset
            overlap = 1 - offset + (2 * offset - 1) * remainder
            valid = jnp.all(jnp.logical_and(
                    filling >= 0, filling < self.shape[None]), axis=1)
            filling = jnp.where(valid[:, None], filling, self.shape[None])
            out = out.at[filling[..., 0], filling[..., 1]].add(
                    overlap[..., 0] * overlap[..., 1], mode="drop")
        return out

    def bin(self, continuous, data, priority=None, topk=8):
        slots = data.shape[-1]
        assert continuous.ndim == priority.ndim + 1 == data.ndim
        assert continuous.shape[-1] == 2
        assert continuous.size // 2 == priority.size == data.size // slots
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
            priority = priority.reshape(-1)
            data = data.reshape(-1, slots)

        sources = jnp.vstack((
            jnp.zeros([slots + 1, continuous.shape[0] + 1]),
            -jnp.ones([1, continuous.shape[0] + 1])))
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
        out = jnp.zeros(tuple(self.shape.tolist()) + (topk, slots))
        r, c = coord_flat // self.shape[1], coord_flat % self.shape[1]
        out = out.at[r, c].set(deref)
        return out

    def binned(self, topk=8):
        dz = jnp.where(self.w != 0, 1 / self.w, 1)[..., None]
        # gradient for the spheroid if it was a sphere
        unsquished = jnp.concatenate((self.grad, dz), -1)
        normed = unsquished / jnp.linalg.norm(
                unsquished, axis=-1, keepdims=True)
        return self.bin(
            self.centers[self.inwards], normed[self.inwards], topk=topk)

    def corners(self, topk=8):
        bins = self.binned(topk)
        slots = bins.shape[-1]
        copies = jnp.zeros(tuple(self.shape.tolist()) + (4, topk, slots))
        copies = copies.at[:, :, 0, :, :].set(bins[:, :])
        copies = copies.at[:-1, :, 1, :, :].set(bins[1:, :])
        copies = copies.at[:, :-1, 2, :, :].set(bins[:, 1:])
        copies = copies.at[:-1, :-1, 3, :, :].set(bins[1:, 1:])
        return copies.reshape(tuple(self.shape.tolist()) + (-1, slots))

    @cached_property
    def metered(self):
        return self.metric(self.binned())

    @jax.jit
    def density(self):
        arr = jnp.where(self.inwards[..., None], self.centers, -2)
        return self.rasterize(arr)

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
            return Depth(self.cache)
        # trends the right direction since depths are flipped
        return Depth(self.cache / self.fov_sec)

class M2(Perspective):
    target = (392, 518)
    f_35mm = 18

examples_dir = local / "assets" / "examples"
cache_dir = local / "cache"

if __name__ == "__main__":
    im4 = M2.file(examples_dir / "IMG_0004.HEIC", cache_dir / "out4.npy")
    im5 = M2.file(examples_dir / "IMG_0005.HEIC", cache_dir / "out5.npy")
    im7 = M2.file(examples_dir / "IMG_0007.HEIC", cache_dir / "out7.npy")
    im8 = M2.file(examples_dir / "IMG_0008.HEIC", cache_dir / "out8.npy")

    import matplotlib.pyplot as plt
    plt.imshow(im4.depth.density())
    plt.show()
