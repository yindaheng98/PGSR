from abc import ABC, abstractmethod

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset


class AbstractMultiViewRegularizer(ABC):

    @property
    @abstractmethod
    def model(self) -> GaussianModel:
        raise ValueError("Model is not set")

    @abstractmethod
    def regularize(
            self,
            loss: torch.Tensor,
            out: dict, camera: Camera,
            nearest_out: dict, nearest_camera: Camera,
            step: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class MultiViewRegularizerWrapper(AbstractMultiViewRegularizer):
    '''
    This class is designed to wrap a multi-view regularizer and add additional functionality.
    Without this class, you should modify the regularizer class directly.
    '''

    def __init__(self, base_regularizer: AbstractMultiViewRegularizer):
        super().__init__()
        self.base_regularizer = base_regularizer

    @property
    def model(self) -> GaussianModel:
        return self.base_regularizer.model

    def regularize(self, loss, out, camera, nearest_out, nearest_camera, step: int) -> torch.Tensor:
        return self.base_regularizer.regularize(
            loss, out, camera, nearest_out, nearest_camera, step
        )


class NoopMultiViewRegularizer(AbstractMultiViewRegularizer):
    '''
    This class is designed to do nothing.
    It is used as the base of all multi-view regularizer wrapper.
    '''

    def __init__(self, model: GaussianModel, dataset: CameraDataset):
        super().__init__()
        self._model = model

    @property
    def model(self) -> GaussianModel:
        return self._model

    def regularize(self, loss, out, camera, nearest_out, nearest_camera, step: int) -> torch.Tensor:
        return loss
