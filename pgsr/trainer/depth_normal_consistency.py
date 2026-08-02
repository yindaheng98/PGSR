from typing import Callable

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper

from ..utils import get_img_grad_weight


class DepthNormalConsistencyTrainer(TrainerWrapper):

    def __init__(
            self,
            base_trainer: AbstractTrainer,
            depth_normal_consistency_weight=0.015,
            depth_normal_consistency_from_iter=7000,
            depth_normal_consistency_edge_aware=True,
    ):
        super().__init__(base_trainer)
        self.depth_normal_consistency_weight = depth_normal_consistency_weight
        self.depth_normal_consistency_from_iter = depth_normal_consistency_from_iter
        self.depth_normal_consistency_edge_aware = depth_normal_consistency_edge_aware
        self.model.render_depth_normal = True

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if not all(name in out for name in ("render_normals", "normals_from_depth")):
            raise KeyError(
                "DepthNormalConsistencyTrainer requires PGSR normal outputs; "
                f"missing {[name for name in ('render_normals', 'normals_from_depth') if name not in out]}"
            )

        loss = super().loss(out, camera)
        if self.curr_step <= self.depth_normal_consistency_from_iter:
            return loss

        # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L183-L196
        weight = self.depth_normal_consistency_weight
        normal = out["render_normals"]
        depth_normal = out["normals_from_depth"]

        image_weight = (1.0 - get_img_grad_weight(camera.ground_truth_image))
        image_weight = (image_weight).clamp(0, 1).detach() ** 2
        if self.depth_normal_consistency_edge_aware:
            normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
        else:
            normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
        loss += normal_loss

        return loss


def DepthNormalConsistencyTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel, dataset: CameraDataset, *args,
        depth_normal_consistency_weight=0.015,
        depth_normal_consistency_from_iter=7000,
        depth_normal_consistency_edge_aware=True,
        **configs) -> DepthNormalConsistencyTrainer:
    return DepthNormalConsistencyTrainer(
        base_trainer_constructor(model, dataset, *args, **configs),
        depth_normal_consistency_weight=depth_normal_consistency_weight,
        depth_normal_consistency_from_iter=depth_normal_consistency_from_iter,
        depth_normal_consistency_edge_aware=depth_normal_consistency_edge_aware,
    )
