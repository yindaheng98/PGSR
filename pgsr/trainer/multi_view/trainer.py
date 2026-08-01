from dataclasses import dataclass

import torch

from gaussian_splatting import Camera

from ...utils.reproj import reconstruction


@dataclass(frozen=True)
class CameraCache:
    K: torch.Tensor
    R_c2w: torch.Tensor
    T_c2w: torch.Tensor

    @classmethod
    def from_camera(cls, camera: Camera, scale: int = 1) -> "CameraCache":
        K = camera.K.detach().clone()
        K[:2] /= scale
        c2w = torch.linalg.inv(camera.world_view_transform.detach())
        return cls(
            K=K,
            R_c2w=c2w[:3, :3].transpose(-1, -2),
            T_c2w=c2w[3, :3],
        )

    def reconstruction(self, depth: torch.Tensor) -> torch.Tensor:
        return reconstruction(self.K, self.R_c2w, self.T_c2w, depth)
