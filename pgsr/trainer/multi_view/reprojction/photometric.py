from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer

from ....utils import lncc, patch_offsets, patch_warp
from ..trainer import MultiViewRegularizationTrainer
from .abc import (
    AbstractMultiViewReprojectionRegularizer,
    MultiViewReprojectionRegularizerWrapper,
    NoopMultiViewReprojectionRegularizer,
)


class MultiViewPhotometricRegularizer(MultiViewReprojectionRegularizerWrapper):

    def __init__(
            self,
            base_regularizer: AbstractMultiViewReprojectionRegularizer,
            ncc_weight=0.15,
            ncc_patch_size=3,
            ncc_sample_num=102400,
            ncc_scale_factor=1.0,
    ):
        super().__init__(base_regularizer)
        if ncc_scale_factor <= 0:
            raise ValueError("ncc_scale_factor must be positive")
        self.ncc_weight = ncc_weight
        self.ncc_patch_size = ncc_patch_size
        self.ncc_sample_num = ncc_sample_num
        self.ncc_scale_factor = ncc_scale_factor
        self.ncc_grayscale_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def ncc_grayscale_and_k(self, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = camera.ground_truth_image_path
        cached = self.ncc_grayscale_cache.get(image_path)
        if cached is not None:
            return cached
        with torch.no_grad():
            # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/cameras.py#L32-L48
            resized_image_rgb = camera.ground_truth_image.detach()
            if self.ncc_scale_factor != 1.0:
                resized_image_rgb = F.interpolate(
                    resized_image_rgb[None],
                    scale_factor=self.ncc_scale_factor,
                    mode="bilinear",
                    align_corners=True,
                )[0]
            gray_image = 0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2]

            # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/cameras.py#L128-L132
            K = camera.K.detach().clone()
            K[:2] *= self.ncc_scale_factor
        self.ncc_grayscale_cache[image_path] = (gray_image, K)
        return gray_image, K

    def compute_loss(
            self,
            out: dict, camera: Camera,
            nearest_out: dict, nearest_camera: Camera,
            pixels: torch.Tensor,
            source_reprojected_uv: torch.Tensor, source_reprojected_z: torch.Tensor,
            valid_reprojection_ratio: torch.Tensor,
            step: int):
        loss = super().compute_loss(
            out, camera, nearest_out, nearest_camera,
            pixels, source_reprojected_uv, source_reprojected_z,
            valid_reprojection_ratio, step,
        )
        if pixels.shape[0] == 0:
            return loss

        patch_size = self.ncc_patch_size
        sample_num = self.ncc_sample_num
        total_patch_size = (patch_size * 2 + 1) ** 2
        ncc_weight = self.ncc_weight
        ncc_scale_factor = self.ncc_scale_factor
        with torch.no_grad():
            # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L273-L281
            # sample mask
            valid_indices = torch.arange(pixels.shape[0], device=pixels.device)
            if valid_indices.shape[0] > sample_num:
                valid_indices = valid_indices[torch.randperm(valid_indices.shape[0], device=pixels.device)[:sample_num]]
            pixel_noise = torch.norm(source_reprojected_uv[:, :2] - pixels, dim=-1)
            weights = torch.exp(-pixel_noise[valid_indices]).detach()

            # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L282-L295
            # sample ref frame patch
            pixels = pixels.reshape(-1, 2)[valid_indices]
            offsets = patch_offsets(patch_size, pixels.device)
            ori_pixels_patch = pixels.reshape(-1, 1, 2) * ncc_scale_factor + offsets.float()

            gt_image_gray, ref_K = self.ncc_grayscale_and_k(camera)
            H, W = gt_image_gray.shape
            pixels_patch = ori_pixels_patch.clone()
            pixels_patch[..., 0] = 2.0 * pixels_patch[..., 0] / (W - 1) - 1.0
            pixels_patch[..., 1] = 2.0 * pixels_patch[..., 1] / (H - 1) - 1.0
            ref_gray_val = F.grid_sample(
                gt_image_gray[None, None],
                pixels_patch.view(1, -1, 1, 2),
                align_corners=True,
            ).reshape(valid_indices.shape[0], total_patch_size)

            ref_to_neareast_r = (
                nearest_camera.world_view_transform[:3, :3].transpose(-1, -2)
                @ camera.world_view_transform[:3, :3]
            )
            ref_to_neareast_t = (
                -ref_to_neareast_r @ camera.world_view_transform[3, :3]
                + nearest_camera.world_view_transform[3, :3]
            )
            nearest_image_gray, nearest_K = self.ncc_grayscale_and_k(nearest_camera)

        # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L297-L312
        # compute Homography
        ref_local_n = out["render_normals"].permute(1, 2, 0).reshape(-1, 3)[valid_indices]
        ref_local_d = out["rendered_distance"].reshape(-1)[valid_indices]
        H_ref_to_neareast = ref_to_neareast_r[None] - (
            ref_to_neareast_t[None, :, None] @ ref_local_n[:, None, :]
        ) / ref_local_d[:, None, None]
        H_ref_to_neareast = nearest_K[None] @ H_ref_to_neareast @ torch.linalg.inv(ref_K)

        # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L314-L320
        # compute neareast frame patch
        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
        nearest_H, nearest_W = nearest_image_gray.shape
        grid[..., 0] = 2.0 * grid[..., 0] / (nearest_W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (nearest_H - 1) - 1.0
        sampled_gray_val = F.grid_sample(
            nearest_image_gray[None, None],
            grid.reshape(1, -1, 1, 2),
            align_corners=True,
        ).reshape(valid_indices.shape[0], total_patch_size)

        # Code source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py#L322-L330
        # compute loss
        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
        mask = ncc_mask.reshape(-1)
        ncc = ncc.reshape(-1) * weights
        ncc = ncc[mask].squeeze()

        if mask.sum() > 0:
            ncc_loss = ncc_weight * valid_reprojection_ratio * ncc.mean()
            loss = loss + ncc_loss
        return loss


def MultiViewPhotometricRegularizerWrapper(
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel, dataset: CameraDataset, *args,
        ncc_weight=0.15,
        ncc_patch_size=3,
        ncc_sample_num=102400,
        ncc_scale_factor=1.0,
        **configs) -> MultiViewPhotometricRegularizer:
    return MultiViewPhotometricRegularizer(
        base_regularizer_constructor(
            model, dataset, *args,
            **configs,
        ),
        ncc_weight=ncc_weight,
        ncc_patch_size=ncc_patch_size,
        ncc_sample_num=ncc_sample_num,
        ncc_scale_factor=ncc_scale_factor,
    )


def MultiViewPhotometricRegularizationTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        base_regularizer_constructor: Callable[..., AbstractMultiViewReprojectionRegularizer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> MultiViewRegularizationTrainer:
    return MultiViewRegularizationTrainer.from_regularizer_constructor(
        base_trainer_constructor,
        partial(MultiViewPhotometricRegularizerWrapper, base_regularizer_constructor),
        model, dataset, *args,
        **configs,
    )


def MultiViewPhotometricTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> MultiViewRegularizationTrainer:
    return MultiViewPhotometricRegularizationTrainerWrapper(
        base_trainer_constructor,
        NoopMultiViewReprojectionRegularizer,
        model, dataset, *args,
        **configs,
    )
