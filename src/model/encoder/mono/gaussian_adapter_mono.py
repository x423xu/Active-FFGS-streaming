import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import time

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from ..common.gaussians import build_covariance
from ..common.gaussian_adapter import GaussianAdapterCfg
from ..common.gaussian_adapter import Gaussians

TIMER = True


def _sync_time(device: torch.device) -> float:
    # CUDA ops are asynchronous; synchronize so stage timings are attributed correctly.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time()


class MonoGaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ):
        if TIMER:
            start_time1 = _sync_time(extrinsics.device)
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        if TIMER:
            elapsed_time1 = _sync_time(device) - start_time1
            start_time2 = _sync_time(device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]

        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        if TIMER:
            elapsed_time2 = _sync_time(device) - start_time2
            start_time3 = _sync_time(device)
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        if TIMER:
            elapsed_time3 = _sync_time(device) - start_time3
            start_time4 = _sync_time(device)
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        if TIMER:
            elapsed_time4 = _sync_time(device) - start_time4
            start_time5 = _sync_time(device)
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]
        
        if TIMER:
            elapsed_time5 = _sync_time(device) - start_time5
            start_time6 = _sync_time(device)
            
        gaussian_out = Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

        if TIMER:
            elapsed_time6 = _sync_time(device) - start_time6
            print(f'###### time1: {elapsed_time1:.4f} s')
            print(f'###### time2: {elapsed_time2:.4f} s')
            print(f'###### time3: {elapsed_time3:.4f} s')
            print(f'###### time4: {elapsed_time4:.4f} s')
            print(f'###### time5: {elapsed_time5:.4f} s')
            print(f'###### time6: {elapsed_time6:.4f} s')
            print(f'###### total time: {elapsed_time1 + elapsed_time2 + elapsed_time3 + elapsed_time4 + elapsed_time5 + elapsed_time6:.4f} s')
        return gaussian_out

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh
