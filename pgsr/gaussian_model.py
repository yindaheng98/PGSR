import torch
import torch.nn.functional as F

from gaussian_splatting import Camera
from gaussian_splatting.utils import quaternion_to_matrix


def plane_params(
    viewpoint_camera: Camera,
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Build camera-facing [normal.xyz, plane_distance] per Gaussian."""
    # https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/gaussian_model.py#L145-L151
    rotation_matrices = quaternion_to_matrix(F.normalize(rotations, dim=-1))
    smallest_axis_idx = scales.argmin(dim=-1)[:, None, None].expand(-1, 3, 1)

    # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/gaussian_model.py#L153-L158
    global_normal = rotation_matrices.gather(2, smallest_axis_idx).squeeze(2)
    camera_center = viewpoint_camera.camera_center.to(device=means3D.device, dtype=means3D.dtype)
    gaussian_to_cam_global = camera_center - means3D
    neg_mask = (global_normal * gaussian_to_cam_global).sum(-1, keepdim=True) < 0.0
    global_normal = torch.where(neg_mask, -global_normal, global_normal)

    # https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/gaussian_renderer/__init__.py#L132-L133
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]

    # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/gaussian_renderer/__init__.py#L135-L139
    local_distance = (local_normal * pts_in_cam).sum(-1, keepdim=True).abs()
    input_all_map = torch.cat((local_normal, local_distance), dim=-1)
    return input_all_map
