from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import (
    AbstractTrainer,
    BaseCameraTrainer,
    BaseTrainer,
    CameraTrainerWrapper,
    OpacityResetTrainerWrapper,
    SHLifter,
)

from .multi_view.reprojction import MultiViewPhotometricGeometricTrainerWrapper
from .trim import BaseMultiViewTrimDensificationTrainer
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


def BasePGSRCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return PGSRTrainerWrapper(BaseCameraTrainer, model, dataset, **configs)


def MultiViewTrimOpacityResetDensificationTrainerWrapper(model: GaussianModel, dataset: CameraDataset, **configs):
    return OpacityResetTrainerWrapper(BaseMultiViewTrimDensificationTrainer, model, dataset, **configs)


def MultiViewTrimOpacityResetDensificationCameraTrainerWrapper(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(MultiViewTrimOpacityResetDensificationTrainerWrapper, model, dataset, **configs)


def BasePGSRMultiViewTrimOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return PGSRTrainerWrapper(MultiViewTrimOpacityResetDensificationTrainerWrapper, model, dataset, **configs)


def BasePGSRMultiViewTrimOpacityResetDensificationCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return PGSRTrainerWrapper(MultiViewTrimOpacityResetDensificationCameraTrainerWrapper, model, dataset, **configs)


def BaseSHLiftPGSRTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BasePGSRTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree,
    )


def BaseSHLiftPGSRCameraTrainer(
        model: GaussianModel,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BasePGSRCameraTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree,
    )


def BaseSHLiftPGSRMultiViewTrimOpacityResetDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BasePGSRMultiViewTrimOpacityResetDensificationTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree,
    )


def BaseSHLiftPGSRMultiViewTrimOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BasePGSRMultiViewTrimOpacityResetDensificationCameraTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree,
    )


# Aliases for default trainers
PGSRTrainer = BasePGSRTrainer
PGSRCameraTrainer = BasePGSRCameraTrainer
PGSROpacityResetDensificationTrainer = BasePGSRMultiViewTrimOpacityResetDensificationTrainer
PGSROpacityResetDensificationCameraTrainer = BasePGSRMultiViewTrimOpacityResetDensificationCameraTrainer
SHLiftPGSRTrainer = BaseSHLiftPGSRTrainer
SHLiftPGSRCameraTrainer = BaseSHLiftPGSRCameraTrainer
SHLiftPGSROpacityResetDensificationTrainer = BaseSHLiftPGSRMultiViewTrimOpacityResetDensificationTrainer
SHLiftPGSROpacityResetDensificationCameraTrainer = BaseSHLiftPGSRMultiViewTrimOpacityResetDensificationCameraTrainer
