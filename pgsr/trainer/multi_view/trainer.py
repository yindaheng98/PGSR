from dataclasses import dataclass
from typing import Optional

import torch

from gaussian_splatting import Camera

from ...utils.reproj import reconstruction


@dataclass(frozen=True)
class CameraCache:
    K: torch.Tensor
    R_c2w: torch.Tensor
    T_c2w: torch.Tensor
    depth: torch.Tensor

    @classmethod
    def from_camera(
            cls,
            camera: Camera,
            depth: torch.Tensor,
            alpha: Optional[torch.Tensor] = None,
            scale: int = 1,
    ) -> "CameraCache":
        K = camera.K.detach().clone()
        K[:2] /= scale
        c2w = torch.linalg.inv(camera.world_view_transform.detach())
        depth = depth.detach().squeeze()[::scale, ::scale].contiguous()
        if alpha is not None:
            depth = depth.masked_fill(
                alpha.detach().squeeze()[::scale, ::scale] <= 0,
                0,
            )
        return cls(
            K=K,
            R_c2w=c2w[:3, :3].transpose(-1, -2),
            T_c2w=c2w[3, :3],
            depth=depth,
        )

    def reconstruction(self) -> torch.Tensor:
        return reconstruction(self.K, self.R_c2w, self.T_c2w, self.depth)
