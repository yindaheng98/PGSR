import torch
import torch.nn.functional as F

from gaussian_splatting import Camera
from .reproj import AbstractMultiViewReprojectionRegularizer, MultiViewReprojectionRegularizerWrapper


class MultiViewPhotometricRegularizer(MultiViewReprojectionRegularizerWrapper):

    def __init__(
            self,
            base_regularizer: AbstractMultiViewReprojectionRegularizer,
            multi_view_ncc_weight=0.15,
            multi_view_patch_size=3,
            multi_view_sample_num=102400,
            multi_view_ncc_scale_factor=1.0,
    ):
        super().__init__(base_regularizer)
        if multi_view_ncc_scale_factor <= 0:
            raise ValueError("multi_view_ncc_scale_factor must be positive")
        self.multi_view_ncc_weight = multi_view_ncc_weight
        self.multi_view_patch_size = multi_view_patch_size
        self.multi_view_sample_num = multi_view_sample_num
        self.multi_view_ncc_scale_factor = multi_view_ncc_scale_factor
        self.ncc_grayscale_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def ncc_grayscale_and_k(self, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = camera.ground_truth_image_path
        cached = self.ncc_grayscale_cache.get(image_path)
        if cached is not None:
            return cached
        with torch.no_grad():
            # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/cameras.py#L32-L48
            resized_image_rgb = camera.ground_truth_image.detach()
            if self.multi_view_ncc_scale_factor != 1.0:
                resized_image_rgb = F.interpolate(
                    resized_image_rgb[None],
                    scale_factor=self.multi_view_ncc_scale_factor,
                    mode="bilinear",
                    align_corners=True,
                )[0]
            gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]

            # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/cameras.py#L128-L132
            K = camera.K.detach().clone()
            K[:2] *= self.multi_view_ncc_scale_factor
        self.ncc_grayscale_cache[image_path] = (gray_image, K)
        return gray_image, K
