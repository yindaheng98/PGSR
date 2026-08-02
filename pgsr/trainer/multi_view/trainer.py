from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

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
            scale_factor: float = 1.0,
            alpha_threshold: float = 1e-4,
    ) -> "CameraCache":
        K = camera.K.detach().clone()
        K[:2] *= scale_factor
        c2w = torch.linalg.inv(camera.world_view_transform.detach())
        depth = depth.detach().squeeze()[None, None]
        if scale_factor != 1:
            depth = F.interpolate(
                depth,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        depth = depth[0, 0].contiguous()
        if alpha is not None:
            alpha = alpha.detach().squeeze()[None, None]
            if scale_factor != 1:
                alpha = F.interpolate(
                    alpha,
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
            depth = depth.masked_fill(alpha[0, 0] < alpha_threshold, 0)
        return cls(
            K=K,
            R_c2w=c2w[:3, :3].transpose(-1, -2),
            T_c2w=c2w[3, :3],
            depth=depth,
        )

    def reconstruction(self) -> torch.Tensor:
        return reconstruction(self.K, self.R_c2w, self.T_c2w, self.depth)
