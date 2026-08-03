from functools import partial
from typing import Callable, Optional

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer

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
    ):
        super().__init__(base_regularizer)
        self.dataset = dataset
        self.geo_weight = geo_weight
        self.virtual_camera_translation_min_scale = virtual_camera_translation_min_scale
        self.virtual_camera_translation_max_scale = virtual_camera_translation_max_scale
        self.camera_distance_update_interval = camera_distance_update_interval
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

    def compute_loss(
            self,
            out, camera,
            nearest_out, nearest_camera,
            pixels,
            source_reprojected_uv, source_reprojected_z,
            step):
        loss = super().compute_loss(
            out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z, step,
        )
        if pixels.shape[0] > 0:
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L234
            pixel_noise = torch.norm(source_reprojected_uv[:, :2] - pixels, dim=-1)
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L237
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L269-L271
            geo_loss = self.geo_weight * (weights * pixel_noise).mean()
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
