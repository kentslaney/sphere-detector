import sys, pathlib
from functools import cached_property

import numpy as np
from scipy.signal import correlate2d

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
        return np.array(self.model.infer_image(np.array(im)))

class Horizon:
    def __init__(self, stretch=1):
        self.stretch = stretch

    def diagonal(self, topk):
        if self.stretch == 1:
            return np.eye(topk)
        stretch = np.eye(topk) * self.stretch ** (-1 / (topk - 1))
        stretch[0, 0] = self.stretch
        initial_basis = np.eye(topk)
        initial_basis[:, 0] = np.ones(topk)
        # Q is orthonormal with a matching direction for the first vector
        rotation, _ = np.linalg.qr(initial_basis)
        symmetric_matrix = rotation @ stretch @ rotation.T
        return symmetric_matrix

    def __call__(self, cylindrical):
        assert 1 < cylindrical.shape[-1] < 4
        topk = cylindrical.shape[-2]
        halved = np.atan2(cylindrical[..., 0], cylindrical[..., 1]) / 2
        transformed = np.stack((np.cos(halved), np.sin(halved)), -1) * \
                np.linalg.norm(cylindrical, axis=-1, keepdims=True)
        spread = np.broadcast_to(
                self.diagonal(topk)[(None,) * (cylindrical.ndim - 2)],
                cylindrical.shape[:-2] + (topk, topk))
        expanded_T = np.concatenate(
                (transformed, cylindrical[..., 2:], spread), -1)
        expanded = np.moveaxis(expanded_T, -2, -1)
        return np.linalg.slogdet(expanded_T @ expanded).logabsdet

class Raster:
    model = Da2('vits')
    metric = Horizon(1)
    target = None
    f_35mm = None

    @classmethod
    def file(cls, path, npy=None):
        cache = None
        if npy is not None:
            npy = pathlib.Path(npy)
            if npy.exists():
                cache = np.load(npy)
                if cls.target is not None and np.all(cache.shape != cls.target):
                    cache = None
        obj = cls(Image.open(path), cache)
        if npy is not None:
            npy.parents[0].mkdir(parents=True, exist_ok=True)
            np.save(npy, obj.cache)
        return obj

    @cached_property
    def cache(self):
        return self.model(self.cropped())

    @property
    def depth(self):
        return self.cache

    @property
    def shape(self):
        return np.array(self.full.size if self.target is None else self.target)

    @property
    def coord(self):
        x, y = np.meshgrid(*map(np.arange, self.shape[::-1]))
        return np.stack((y, x), -1)

    def cropped(self):
        if self.target is None:
            return self.full
        size = np.array(self.full.size)
        scaled = np.int32(np.max(self.shape / size) * size)
        resample = self.full.resize(scaled)
        origin = (scaled - self.shape[::-1]) // 2
        return resample.crop(np.concat((origin, origin + self.shape[::-1])))

    def __init__(self, im, cache=None):
        self.full = im
        if cache is not None:
            self.cache = cache

    @staticmethod
    def scharr(arr):
        kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        l1 = np.sum(np.abs(kernel))
        kw = { 'boundary': 'symm', 'mode': 'same' }
        return np.stack((
                correlate2d(arr, kernel, **kw),
                correlate2d(arr, kernel.T, **kw)), -1) / l1

    @cached_property
    def norm2(self):
        return np.sum(self.grad ** 2, -1)

    @cached_property
    def norm(self):
        return np.sqrt(self.norm2)

    @cached_property
    def grad(self):
        return self.scharr(self.depth)

    @cached_property
    def hessian(self):
        return np.stack((
                self.scharr(self.grad[..., 0]),
                self.scharr(self.grad[..., 1])), -1)

    @cached_property
    def rotated(self):
        norm = self.norm[..., None]
        basis0 = np.divide(
                self.grad, norm, where=norm != 0,
                out=np.zeros_like(self.grad))
        basis1 = basis0[..., ::-1] * np.array([[[-1, 1]]])
        basis = np.stack((basis0, basis1), -1)
        inv = basis * np.array([[[[1, -1], [-1, 1]]]])
        return inv @ self.hessian @ basis

    @cached_property
    def inwards(self):
        convex = np.logical_and(
                np.linalg.det(self.rotated) > 0, self.rotated[..., 0, 0] < 0)
        return np.logical_and(
                convex, self.rotated[..., 0, 0] <= self.rotated[..., 1, 1])

    @cached_property
    def flat_radius_over_norm(self):
        return np.divide(
                1, self.rotated[..., 1, 1], where=self.inwards,
                out=np.zeros(self.shape))

    @cached_property
    def centers(self):
        return self.coord - self.flat_radius_over_norm[..., None] * self.grad

    @cached_property
    def w2(self):
        sec2 = np.divide(
                self.rotated[..., 0, 0],
                self.rotated[..., 1, 1],
                where=self.inwards,

                out=np.ones(self.shape))

        return np.divide(
                sec2 - 1, self.norm2, where=self.norm2 != 0,
                out=np.ones(self.shape))

    @cached_property
    def w(self):
        return np.sqrt(self.w2)

    def rasterize(self, continuous):
        assert continuous.shape[-1] == 2
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
        out = np.zeros(self.shape)
        floored = np.int32(np.floor(continuous))
        remainder = continuous - floored
        for offset in np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]):
            filling = floored + offset
            overlap = 1 - offset + (2 * offset - 1) * remainder
            valid = np.all(np.logical_and(
                    filling >= 0, filling < self.shape[None]), axis=1)
            np.add.at(
                    out,
                    (filling[valid][..., 0], filling[valid][..., 1]),
                    overlap[valid][..., 0] * overlap[valid][..., 1])
        return out

    def bin(self, continuous, data, priority=None, topk=8):
        slots = data.shape[-1]
        if priority is None:
            priority = np.random.uniform(size=continuous.shape[:-1])
        assert continuous.ndim == priority.ndim + 1 == data.ndim
        assert continuous.shape[-1] == 2
        assert continuous.size // 2 == priority.size == data.size // slots
        if continuous.ndim > 2:
            continuous = continuous.reshape(-1, 2)
            priority = priority.reshape(-1)
            data = data.reshape(-1, slots)

        sources = np.vstack((
            np.zeros([slots + 1, continuous.shape[0] + 1]),
            -np.ones([1, continuous.shape[0] + 1])))
        floored = np.int32(np.floor(continuous))

        valid = np.all(np.logical_and(
                floored >= 0, floored < self.shape[None]), axis=1)
        floored, priority, data = floored[valid], priority[valid], data[valid]
        flat_index = floored[..., 0] * self.shape[1] + floored[..., 1]
        sources[:, :flat_index.shape[0]] = \
                np.vstack((data.T, -priority[None], flat_index[None]))

        sources = sources[:, np.lexsort(sources)]
        coord_flat, offset, counts = np.unique(
                sources[-1], return_index=True, return_counts=True)
        keeping = np.arange(topk)[None] < counts[:, None]
        ref = (offset[:, None] + np.arange(topk)[None]) * keeping
        deref = sources[:slots, np.ravel(ref)].reshape(slots, -1, topk)
        deref = np.transpose(deref, axes=(1, 2, 0))
        deref, coord_flat = deref[1:], np.int32(coord_flat[1:])
        out = np.zeros(self.shape.tolist() + [topk, slots])
        out[coord_flat // self.shape[1], coord_flat % self.shape[1]] = deref
        return out

    @cached_property
    def metered(self):
        dz = np.divide(
                1, self.w, where=self.w != 0,
                out=np.ones(self.shape))[..., None]
        # gradient for the spheroid if it was a sphere
        unsquished = np.concatenate((self.grad, dz), -1)
        normed = unsquished / np.linalg.norm(unsquished, axis=-1, keepdims=True)
        return self.metric(self.bin(self.centers, normed))

class Perspective(Raster):
    f_35mm = None

    @property
    def f_px(self):
        return self.f_35mm / 35 * np.linalg.norm(self.shape)

    @property
    def fov_sec(self):
        f_center = (self.coord - self.shape[None] / 2) / self.f_px
        return np.sqrt(1 + np.sum(f_center ** 2, axis=-1))

    @cached_property
    def depth(self):
        if self.f_35mm is None:
            return self.cache
        # trends the right direction since depths are flipped
        return self.cache / self.fov_sec

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
    plt.imshow(im5.metered)
    plt.show()
