from typing import Callable

import torch

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .reproj import AbstractMultiViewReprojectionRegularizer, MultiViewReprojectionRegularizerWrapper


class MultiViewGeometricRegularizer(MultiViewReprojectionRegularizerWrapper):

    def __init__(
            self,
            base_regularizer: AbstractMultiViewReprojectionRegularizer,
            multi_view_geo_weight=0.03,
    ):
        super().__init__(base_regularizer)
        self.multi_view_geo_weight = multi_view_geo_weight

    def compute_loss(
            self,
            loss,
            out, camera,
            nearest_out, nearest_camera,
            pixels,
            source_reprojected_uv, source_reprojected_z,
            step):
        loss = super().compute_loss(
            loss, out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z, step,
        )
        if pixels.shape[0] > 0:
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L234
            pixel_noise = torch.norm(source_reprojected_uv[:, :2] - pixels, dim=-1)
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L237
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L269-L271
            geo_loss = self.multi_view_geo_weight * (weights * pixel_noise).mean()
            loss = loss + geo_loss
        return loss


def MultiViewGeometricRegularizerWrapper(
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        multi_view_geo_weight=0.03,
        max_reprojection_error=1.0,
        **configs) -> MultiViewGeometricRegularizer:
    return MultiViewGeometricRegularizer(
        base_regularizer_constructor(
            model,
            dataset,
            *args,
            max_reprojection_error=max_reprojection_error,
            **configs,
        ),
        multi_view_geo_weight=multi_view_geo_weight,
    )
