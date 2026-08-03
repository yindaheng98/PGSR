from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer, BaseTrainer

from .multi_view.reprojction import MultiViewPhotometricGeometricTrainerWrapper
from .reprojection import VirtualCameraReprojectionTrainerWrapper
from .scale import PlanarScaleTrainerWrapper
from .depth_normal_consistency import DepthNormalConsistencyTrainerWrapper


def PGSRTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> AbstractTrainer:
    base_trainer_constructor = partial(PlanarScaleTrainerWrapper, base_trainer_constructor)
    base_trainer_constructor = partial(DepthNormalConsistencyTrainerWrapper, base_trainer_constructor)
    base_trainer_constructor = partial(MultiViewPhotometricGeometricTrainerWrapper, base_trainer_constructor)
    base_trainer_constructor = partial(VirtualCameraReprojectionTrainerWrapper, base_trainer_constructor)
    return base_trainer_constructor(model, dataset, *args, **configs)


def BasePGSRTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return PGSRTrainerWrapper(BaseTrainer, model, dataset, **configs)
