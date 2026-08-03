from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer

from ..trainer import MultiViewRegularizationTrainer
from .abc import (
    AbstractMultiViewReprojectionRegularizer,
    NoopMultiViewReprojectionRegularizer,
)
from .geometric import MultiViewGeometricRegularizerWrapper
from .photometric import MultiViewPhotometricRegularizerWrapper


def MultiViewPhotometricGeometricRegularizer(
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> AbstractMultiViewReprojectionRegularizer:
    return MultiViewPhotometricRegularizerWrapper(
        partial(MultiViewGeometricRegularizerWrapper, base_regularizer_constructor),
        model, dataset, *args, **configs)


def BaseMultiViewPhotometricGeometricRegularizer(model: GaussianModel, dataset: CameraDataset, *args, **configs):
    return MultiViewPhotometricGeometricRegularizer(
        NoopMultiViewReprojectionRegularizer, model, dataset, *args, **configs)


def MultiViewPhotometricGeometricRegularizationTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> MultiViewRegularizationTrainer:
    return MultiViewRegularizationTrainer.from_regularizer_constructor(
        base_trainer_constructor,
        partial(MultiViewPhotometricGeometricRegularizer, base_regularizer_constructor),
        model, dataset, *args,
        **configs,
    )


def MultiViewPhotometricGeometricTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> MultiViewRegularizationTrainer:
    return MultiViewPhotometricGeometricRegularizationTrainerWrapper(
        base_trainer_constructor,
        NoopMultiViewReprojectionRegularizer,
        model, dataset, *args,
        **configs,
    )
