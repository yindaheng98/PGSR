from functools import partial
from typing import Callable, Optional

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer

from ....utils import reconstruct_pixels
from ..trainer import MultiViewRegularizationTrainer
from .abc import (
    AbstractMultiViewReprojectionRegularizer,
    MultiViewReprojectionRegularizerWrapper,
    NoopMultiViewReprojectionRegularizer,
)


class MultiViewGeometricRegularizer(MultiViewReprojectionRegularizerWrapper):

    def __init__(
            self,
            base_regularizer: AbstractMultiViewReprojectionRegularizer,
            dataset: CameraDataset,
            geo_weight=0.03,
            virtual_camera_translation_min_scale=0.1,
            virtual_camera_translation_max_scale=1.0,
            camera_distance_update_interval=1000,
            visible_sample_count=4096,
            visible_sample_min_depth=0.01,
            visible_sample_max_depth=100.0,
            visible_sample_min_alpha=1.0e-4,
    ):
        super().__init__(base_regularizer)
        self.dataset = dataset
        self.geo_weight = geo_weight
        self.virtual_camera_translation_min_scale = virtual_camera_translation_min_scale
        self.virtual_camera_translation_max_scale = virtual_camera_translation_max_scale
        self.camera_distance_update_interval = camera_distance_update_interval
        self.visible_sample_count = visible_sample_count
        self.visible_sample_min_depth = visible_sample_min_depth
        self.visible_sample_max_depth = visible_sample_max_depth
        self.visible_sample_min_alpha = visible_sample_min_alpha
        self.camera_indices = {
            dataset[idx].ground_truth_image_path: idx
            for idx in range(len(dataset))
        }
        self.camera_min_distances: torch.Tensor
        self.camera_min_distances_step = 0
        self.update_camera_min_distances(0)

    def update_camera_min_distances(self, step: int):
        if (step > 0 and self.camera_distance_update_interval > 0
                and step - self.camera_min_distances_step < self.camera_distance_update_interval):
            return

        cameras = [self.dataset[idx] for idx in range(len(self.dataset))]
        centers = torch.stack([
            camera.camera_center.detach()
            for camera in cameras
        ])
        distances = torch.cdist(centers, centers)
        distances.fill_diagonal_(float("inf"))
        self.camera_min_distances = distances.min(dim=1).values
        self.camera_min_distances_step = step

    def camera_min_distance(self, camera: Camera, step: int) -> torch.Tensor:
        self.update_camera_min_distances(step)
        camera_idx = self.camera_indices[camera.ground_truth_image_path]
        return self.camera_min_distances[camera_idx].to(
            device=camera.camera_center.device,
            dtype=camera.camera_center.dtype,
        )

    def sample_translation(self, camera: Camera, step: int) -> Optional[torch.Tensor]:
        min_distance = self.camera_min_distance(camera, step)
        min_radius = min_distance * self.virtual_camera_translation_min_scale
        max_radius = min_distance * self.virtual_camera_translation_max_scale

        direction = torch.randn(3, device=min_distance.device, dtype=min_distance.dtype)
        direction = direction / torch.linalg.norm(direction)
        # Sample radius uniformly by shell volume, not uniformly by radius.
        radius = (
            torch.rand((), device=max_radius.device, dtype=max_radius.dtype)
            * (max_radius ** 3 - min_radius ** 3) + min_radius ** 3
        ).pow(1.0 / 3.0)
        return direction * radius

    def estimate_visible_region_median(self, out: dict, camera: Camera) -> Optional[torch.Tensor]:
        """Estimate the median world point of the rendered visible region.

        Reconstructing every depth pixel is expensive, so this randomly samples
        visible pixels, reconstructs only those samples, and takes their xyz median.
        """
        depth = out["depth"].detach().squeeze()
        # Compute valid indices in one pass.
        valid = (depth > self.visible_sample_min_depth) & (depth < self.visible_sample_max_depth)
        if "render_alphas" in out:
            valid = valid & (out["render_alphas"].detach().squeeze() > self.visible_sample_min_alpha)
        valid_indices = valid.reshape(-1).nonzero().squeeze(-1)
        if valid_indices.shape[0] > self.visible_sample_count:
            valid_indices = valid_indices[
                torch.randperm(valid_indices.shape[0], device=valid_indices.device)[:self.visible_sample_count]
            ]
        # Flatten indices into grid coordinates.
        height, width = depth.shape
        grid_y = torch.div(valid_indices, width, rounding_mode="floor")
        grid_x = valid_indices % width
        # Sample depth and pixels.
        sampled_depth = depth.reshape(-1)[valid_indices]
        sampled_pixels = torch.stack((
            grid_x.to(dtype=depth.dtype),
            grid_y.to(dtype=depth.dtype),
        ), dim=-1)
        # Reconstruct xyz samples.
        c2w = torch.linalg.inv(camera.world_view_transform.detach())
        xyz = reconstruct_pixels(
            camera.K.detach(),
            c2w[:3, :3].transpose(-1, -2),
            c2w[3, :3],
            sampled_pixels,
            sampled_depth,
        )
        # Take median.
        return xyz.median(dim=0).values

    def look_at_rotation(
            self,
            source_c2w: torch.Tensor,
            camera_center: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        forward = torch.nn.functional.normalize(target - camera_center, dim=0)
        right = torch.nn.functional.normalize(torch.cross(source_c2w[:3, 1], forward, dim=0), dim=0)
        return torch.stack((right, torch.cross(forward, right, dim=0), forward), dim=1)

    def compute_loss(
            self,
            out, camera,
            nearest_out, nearest_camera,
            pixels,
            source_reprojected_uv, source_reprojected_z,
            valid_reprojection_ratio,
            step):
        loss = super().compute_loss(
            out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z,
            valid_reprojection_ratio, step,
        )
        if pixels.shape[0] > 0:
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L234
            pixel_noise = torch.norm(source_reprojected_uv[:, :2] - pixels, dim=-1)
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L237
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L269-L271
            geo_loss = self.geo_weight * valid_reprojection_ratio * (weights * pixel_noise).mean()
            loss += geo_loss
        return loss


def MultiViewGeometricRegularizerWrapper(
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel, dataset: CameraDataset, *args,
        geo_weight=0.03,
        **configs) -> MultiViewGeometricRegularizer:
    return MultiViewGeometricRegularizer(
        base_regularizer_constructor(
            model, dataset, *args,
            **configs,
        ),
        geo_weight=geo_weight,
    )


def MultiViewGeometricRegularizationTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> MultiViewRegularizationTrainer:
    return MultiViewRegularizationTrainer.from_regularizer_constructor(
        base_trainer_constructor,
        partial(MultiViewGeometricRegularizerWrapper, base_regularizer_constructor),
        model, dataset, *args,
        **configs,
    )


def MultiViewGeometricTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> MultiViewRegularizationTrainer:
    return MultiViewGeometricRegularizationTrainerWrapper(
        base_trainer_constructor,
        NoopMultiViewReprojectionRegularizer,
        model, dataset, *args,
        **configs,
    )
