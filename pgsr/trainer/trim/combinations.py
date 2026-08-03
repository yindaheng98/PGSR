from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import (
    AbstractDensifier,
    DensificationTrainer,
    NoopDensifier,
)

from .trainer import MultiViewTrimmerDensificationDensifierWrapper


def MultiViewTrimDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> DensificationTrainer:
    return DensificationTrainer.from_densifier_constructor(
        partial(MultiViewTrimmerDensificationDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs,
    )


def BaseMultiViewTrimDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs) -> DensificationTrainer:
    return MultiViewTrimDensificationTrainerWrapper(NoopDensifier, model, dataset, **configs)
