import sys
import jax.numpy as jnp
import numpy as np
from functools import cached_property

from .utils import local

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

    def infer_direct(self, im):
        """Run DepthAnythingV2 directly at native 1:1 resolution with zero resizing or bilinear interpolation."""
        import torch
        np_im = np.array(im)
        if np_im.ndim == 2:
            np_im = np.stack([np_im] * 3, axis=-1)
        elif np_im.shape[2] == 4:
            np_im = np_im[:, :, :3]

        h, w = np_im.shape[:2]
        pad_h = (14 - h % 14) % 14
        pad_w = (14 - w % 14) % 14
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

        im_float = np_im.astype(np.float32) / 255.0
        im_padded = np.pad(im_float, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")

        device = next(self.model.parameters()).device
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

        tensor = torch.from_numpy(im_padded).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            depth_out = self.model(tensor)

        depth_np = depth_out.squeeze().cpu().numpy()
        return jnp.array(depth_np[pad_top : pad_top + h, pad_left : pad_left + w])

    def __call__(self, im, direct=False):
        if direct:
            return self.infer_direct(im)
        return jnp.array(self.model.infer_image(np.array(im)))

