from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper

from ...utils.reproj import reconstruction, visibility
from .abc import AbstractMultiViewRegularizer


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

    def reconstruction(self, min_depth: float = 0.01, max_depth: float = 100.0) -> torch.Tensor:
        # Keep only depths inside the trusted range before returning xyz samples.
        valid = (self.depth > min_depth) & (self.depth < max_depth)
        xyz = reconstruction(self.K, self.R_c2w, self.T_c2w, self.depth)
        return xyz[valid]

    def visibility(
            self,
            xyz: torch.Tensor,
            relative_depth_tolerance: float,
            min_depth: float = 0.01, max_depth: float = 100.0,
    ) -> torch.Tensor:
        # Test the input world-space points against this cache's camera and depth map.
        return visibility(
            self.K, self.R_c2w, self.T_c2w, self.depth,
            xyz,
            relative_depth_tolerance,
            min_depth, max_depth,
        )


class MultiViewRegularizationTrainer(TrainerWrapper):

    def __init__(
            self,
            base_trainer: AbstractTrainer,
            dataset: CameraDataset,
            regularizer: AbstractMultiViewRegularizer,
            neighbor_view_n_max=8,
            neighbor_view_update_interval=1000,
            neighbor_view_depth_tolerance_ratio=0.05,
            neighbor_view_depth_scale_factor=0.25,
    ):
        super().__init__(base_trainer)
        self.dataset = dataset
        self.regularizer = regularizer
        self.neighbor_view_n_max = neighbor_view_n_max
        self.neighbor_view_update_interval = neighbor_view_update_interval
        self.neighbor_view_depth_tolerance_ratio = neighbor_view_depth_tolerance_ratio
        self.neighbor_view_depth_scale_factor = neighbor_view_depth_scale_factor
        camera_count = len(dataset)
        self.camera_indices = {dataset[idx].ground_truth_image_path: idx for idx in range(camera_count)}
        self.camera_cache: list[Optional[CameraCache]] = [None] * camera_count
        self.nearest_indices = [[] for _ in range(camera_count)]
        self.nearest_update_steps = [-neighbor_view_update_interval for _ in range(camera_count)]
