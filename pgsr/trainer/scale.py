from typing import Callable

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper


class PlanarScaleTrainer(TrainerWrapper):

    def __init__(self, base_trainer: AbstractTrainer, scale_loss_weight=100.0):
        super().__init__(base_trainer)
        self.scale_loss_weight = scale_loss_weight

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = super().loss(out, camera)
        visibility_filter = out["visibility_filter"].reshape(-1)
        if visibility_filter.numel() == 0:
            return loss

        # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L177-L182
        scale = self.model.get_scaling[visibility_filter]
        sorted_scale, _ = torch.sort(scale, dim=-1)
        min_scale_loss = sorted_scale[..., 0]
        return loss + self.scale_loss_weight * min_scale_loss.mean()


def PlanarScaleTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        scale_loss_weight=100.0,
        **configs) -> PlanarScaleTrainer:
    return PlanarScaleTrainer(
        base_trainer_constructor(model, dataset, *args, **configs),
        scale_loss_weight=scale_loss_weight,
    )
