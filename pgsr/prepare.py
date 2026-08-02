from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.dataset.colmap import colmap_init
from gaussian_splatting.trainer import (
    AbstractTrainer,
    BaseCameraTrainer,
    BaseOpacityResetDensificationCameraTrainer,
    BaseOpacityResetDensificationTrainer,
    BaseSHLiftCameraTrainer,
    BaseSHLiftOpacityResetDensificationCameraTrainer,
    BaseSHLiftOpacityResetDensificationTrainer,
    BaseSHLiftTrainer,
    BaseTrainer,
)

from .trainer import PGSRTrainerWrapper


backends = ["gsplat", "gsplat-2dgs"]
basemodes = {
    "base": partial(PGSRTrainerWrapper, BaseTrainer),
    "densify": partial(PGSRTrainerWrapper, BaseOpacityResetDensificationTrainer),
    "camera": partial(PGSRTrainerWrapper, BaseCameraTrainer),
    "camera-densify": partial(PGSRTrainerWrapper, BaseOpacityResetDensificationCameraTrainer),
}
shliftmodes = {
    "base": partial(PGSRTrainerWrapper, BaseSHLiftTrainer),
    "densify": partial(PGSRTrainerWrapper, BaseSHLiftOpacityResetDensificationTrainer),
    "camera": partial(PGSRTrainerWrapper, BaseSHLiftCameraTrainer),
    "camera-densify": partial(PGSRTrainerWrapper, BaseSHLiftOpacityResetDensificationCameraTrainer),
}


def get_gaussian_model_class(backend: str, trainable_camera: bool = False) -> Callable[[int], GaussianModel]:
    match backend:
        case "gsplat":
            from .models.gsplat import GsplatPGSRGaussianModel, CameraTrainableGsplatPGSRGaussianModel
            return GsplatPGSRGaussianModel if not trainable_camera else CameraTrainableGsplatPGSRGaussianModel
        case "gsplat-2dgs":
            from .models.gsplat_2dgs import Gsplat2DGSPGSRGaussianModel, CameraTrainableGsplat2DGSPGSRGaussianModel
            return Gsplat2DGSPGSRGaussianModel if not trainable_camera else CameraTrainableGsplat2DGSPGSRGaussianModel
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def prepare_gaussians(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None, backend: str = "gsplat-2dgs") -> GaussianModel:
    gaussians = get_gaussian_model_class(backend, trainable_camera=trainable_camera)(sh_degree).to(device)
    gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    return gaussians


def prepare_trainer(gaussians: GaussianModel, dataset: CameraDataset, mode: str, load_ply: str = None, configs={}) -> AbstractTrainer:
    modes = basemodes if load_ply else shliftmodes
    trainer = modes[mode](gaussians, dataset, **configs)
    return trainer
