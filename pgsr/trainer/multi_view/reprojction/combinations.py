from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import (
    AbstractMultiViewReprojectionRegularizer,
    NoopMultiViewReprojectionRegularizer,
)
from .geometric import MultiViewGeometricRegularizerWrapper
from .photometric import MultiViewPhotometricRegularizerWrapper


def PGSRMultiViewRegularizerWrapper(
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel, dataset: CameraDataset,
        *args, **configs) -> AbstractMultiViewReprojectionRegularizer:
    return MultiViewPhotometricRegularizerWrapper(
        partial(MultiViewGeometricRegularizerWrapper, base_regularizer_constructor),
        model, dataset, *args, **configs)


def BasePGSRMultiViewRegularizer(model: GaussianModel, dataset: CameraDataset, *args, **configs):
    return PGSRMultiViewRegularizerWrapper(NoopMultiViewReprojectionRegularizer, model, dataset, *args, **configs)
