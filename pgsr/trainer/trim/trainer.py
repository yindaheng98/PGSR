from functools import partial
from typing import Callable, Optional

import torch
from tqdm import tqdm

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import (
    AbstractDensifier,
    DensificationInstruct,
    DensifierWrapper,
)


class MultiViewTrimmer(DensifierWrapper):

    def __init__(
            self,
            base_densifier: AbstractDensifier,
            dataset: CameraDataset,
            trim_from_iter: int = 0,
            trim_until_iter: Optional[int] = 15000,
            trim_interval: int = 1000,
            trim_observe_threshold: int = 2,
    ):
        super().__init__(base_densifier)
        self.dataset = dataset
        self.trim_from_iter = trim_from_iter
        self.trim_until_iter = trim_until_iter
        self.trim_interval = trim_interval
        self.trim_observe_threshold = trim_observe_threshold

    def trim_mask(self) -> torch.Tensor:
        observe_count = torch.zeros(self.model.get_xyz.shape[0], device=self.model.get_xyz.device, dtype=torch.int32)
        for camera_idx in tqdm(range(len(self.dataset)), total=len(self.dataset), desc="Multi-view trim", leave=False, position=1):
            with torch.no_grad():
                out = self.model(self.dataset[camera_idx])
            observe_count[out["visibility_filter"].reshape(-1)] += 1
        return observe_count < self.trim_observe_threshold

    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        ret = super().densify_and_prune(loss, out, camera, step)
        if self.trim_from_iter <= step < self.trim_until_iter and step % self.trim_interval == 0:
            trim_mask = self.trim_mask()
            remove_mask = trim_mask if ret.remove_mask is None else torch.logical_or(ret.remove_mask, trim_mask)
            ret = ret._replace(remove_mask=remove_mask)
        return ret


def MultiViewTrimmerDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        trim_from_iter: int = 0,
        trim_until_iter: Optional[int] = None,
        trim_interval: int = 1000,
        trim_observe_threshold: int = 2,
        **configs) -> MultiViewTrimmer:
    if trim_until_iter is None:
        trim_until_iter = configs.get("densify_until_iter", 15000)
    return MultiViewTrimmer(
        base_densifier_constructor(model, dataset, *args, **configs),
        dataset,
        trim_from_iter=trim_from_iter,
        trim_until_iter=trim_until_iter,
        trim_interval=trim_interval,
        trim_observe_threshold=trim_observe_threshold,
    )


def MultiViewTrimmerDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> MultiViewTrimmer:
    from gaussian_splatting.trainer import DensificationDensifierWrapper

    return MultiViewTrimmerDensifierWrapper(
        partial(DensificationDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs,
    )
