import torch
import torch.nn.functional as F

from gaussian_splatting import Camera
from gaussian_splatting.utils import quaternion_to_matrix

from .utils import normal_from_depth_image


def render_normal(
    viewpoint_camera: Camera,
    depth: torch.Tensor,
    offset: torch.Tensor | None = None,
    # normal: torch.Tensor | None = None,
    scale: int = 1,
) -> torch.Tensor:
    """Render depth-derived normals in PGSR's [C, H, W] convention."""
    # Source: https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/gaussian_renderer/__init__.py#L21-L33
    intrinsic_matrix = viewpoint_camera.K.to(device=depth.device, dtype=depth.dtype).clone()
    intrinsic_matrix[0] /= scale
    intrinsic_matrix[1] /= scale
    extrinsic_matrix = viewpoint_camera.world_view_transform.T.contiguous().to(
        device=depth.device, dtype=depth.dtype
    )
    st = max(int(scale / 2) - 1, 0)
    if offset is not None:
        offset = offset[st::scale, st::scale]
    normal_ref = normal_from_depth_image(
        depth[st::scale, st::scale],
        intrinsic_matrix,
        extrinsic_matrix,
        offset,
    )
    normal_ref = normal_ref.permute(2, 0, 1)
    return normal_ref


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
