from abc import abstractmethod

import torch

from gaussian_splatting import Camera, GaussianModel

from ...utils import reprojection
from .abc import AbstractMultiViewRegularizer


class AbstractMultiViewReprojectionRegularizer(AbstractMultiViewRegularizer):

    def __init__(self, max_reprojection_error: float = 1.0):
        super().__init__()
        # Maximum allowed round-trip reprojection error in pixels before a correspondence is rejected.
        self.max_reprojection_error = max_reprojection_error

    def regularize(
            self,
            loss: torch.Tensor,
            out: dict, camera: Camera,
            nearest_out: dict, nearest_camera: Camera,
            step: int,
    ) -> torch.Tensor:
        c2w = torch.linalg.inv(camera.world_view_transform)
        nearest_c2w = torch.linalg.inv(nearest_camera.world_view_transform)
        pixels, source_reprojected_uv, source_reprojected_z = reprojection(
            source_K=camera.K,
            source_R_c2w=c2w[:3, :3].transpose(-1, -2),
            source_T_c2w=c2w[3, :3],
            source_depth=out["depth"].squeeze(),
            target_K=nearest_camera.K,
            target_R_c2w=nearest_c2w[:3, :3].transpose(-1, -2),
            target_T_c2w=nearest_c2w[3, :3],
            target_depth=nearest_out["depth"].squeeze(),
        )
        reprojection_error = torch.norm(source_reprojected_uv[:, :2] - pixels, dim=-1)
        valid_reprojection = reprojection_error < self.max_reprojection_error
        pixels = pixels[valid_reprojection]
        source_reprojected_uv = source_reprojected_uv[valid_reprojection]
        source_reprojected_z = source_reprojected_z[valid_reprojection]
        return self.compute_loss(
            loss, out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z,
            step,
        )

    @abstractmethod
    def compute_loss(
            self,
            loss: torch.Tensor,
            out: dict, camera: Camera,
            nearest_out: dict, nearest_camera: Camera,
            pixels: torch.Tensor,
            source_reprojected_uv: torch.Tensor,
            source_reprojected_z: torch.Tensor,
            step: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class MultiViewReprojectionRegularizerWrapper(AbstractMultiViewReprojectionRegularizer):
    '''
    This class is designed to wrap a multi-view regularizer and add additional functionality.
    Without this class, you should modify the regularizer class directly.
    '''

    def __init__(self, base_regularizer: AbstractMultiViewReprojectionRegularizer):
        super().__init__(max_reprojection_error=base_regularizer.max_reprojection_error)
        self.base_regularizer = base_regularizer

    @property
    def model(self) -> GaussianModel:
        return self.base_regularizer.model

    def compute_loss(
            self, loss, out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z, step) -> torch.Tensor:
        return self.base_regularizer.compute_loss(
            loss, out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z, step,
        )


class NoopMultiViewReprojectionRegularizer(AbstractMultiViewReprojectionRegularizer):
    '''
    This class is designed to do nothing.
    It is used as the base of all multi-view regularizer wrapper.
    '''

    def __init__(self, model: GaussianModel, *args, max_reprojection_error=1.0, **configs):
        super().__init__(max_reprojection_error=max_reprojection_error)
        self._model = model

    @property
    def model(self) -> GaussianModel:
        return self._model

    def compute_loss(
            self, loss, out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z, step) -> torch.Tensor:
        return loss
