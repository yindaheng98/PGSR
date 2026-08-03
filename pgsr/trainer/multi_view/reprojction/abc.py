from abc import abstractmethod

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset

from ...reprojection import compute_valid_reprojection_and_ratio
from ..abc import AbstractMultiViewRegularizer


class AbstractMultiViewReprojectionRegularizer(AbstractMultiViewRegularizer):

    def __init__(self, max_reprojection_error: float = 1.0):
        super().__init__()
        # Maximum allowed round-trip reprojection error in pixels before a correspondence is rejected.
        self.max_reprojection_error = max_reprojection_error

    def compute_reprojection(
            self,
            out: dict, camera: Camera,
            nearest_out: dict, nearest_camera: Camera,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return compute_valid_reprojection_and_ratio(
            out, camera, nearest_out, nearest_camera,
            max_reprojection_error=self.max_reprojection_error,
        )

    def regularize_with_nearest_gt_camera(
            self,
            out: dict, camera: Camera,
            nearest_out: dict, nearest_camera: Camera,
            step: int,
    ) -> torch.Tensor:
        pixels, source_reprojected_uv, source_reprojected_z, valid_reprojection_ratio = self.compute_reprojection(
            out, camera, nearest_out, nearest_camera,
        )
        return self.compute_loss(
            out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z,
            valid_reprojection_ratio,
            step,
        )

    @abstractmethod
    def compute_loss(
            self,
            out: dict, camera: Camera,
            nearest_out: dict, nearest_camera: Camera,
            pixels: torch.Tensor,
            source_reprojected_uv: torch.Tensor,
            source_reprojected_z: torch.Tensor,
            valid_reprojection_ratio: torch.Tensor,
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
            self, out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z,
            valid_reprojection_ratio, step) -> torch.Tensor:
        return self.base_regularizer.compute_loss(
            out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z,
            valid_reprojection_ratio, step,
        )


class NoopMultiViewReprojectionRegularizer(AbstractMultiViewReprojectionRegularizer):
    '''
    This class is designed to do nothing.
    It is used as the base of all multi-view regularizer wrapper.
    '''

    def __init__(self, model: GaussianModel, dataset: CameraDataset, max_reprojection_error=1.0):
        super().__init__(max_reprojection_error=max_reprojection_error)
        self._model = model

    @property
    def model(self) -> GaussianModel:
        return self._model

    def compute_loss(
            self, out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z,
            valid_reprojection_ratio, step) -> torch.Tensor:
        return out["render"].new_zeros(())
