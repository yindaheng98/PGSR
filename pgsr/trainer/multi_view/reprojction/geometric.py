from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer

from ...reprojection import reprojection_loss
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
            geo_weight=0.03,
    ):
        super().__init__(base_regularizer)
        self.geo_weight = geo_weight

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
        return loss + self.geo_weight * valid_reprojection_ratio * reprojection_loss(pixels, source_reprojected_uv)


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
