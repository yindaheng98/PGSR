import torch
import torch.nn.functional as F
from gsplat import spherical_harmonics
from gsplat.rendering import rasterization_2dgs

from gaussian_splatting import Camera
from gaussian_splatting.models.gsplat_2dgs import Gsplat2DGSGaussianModel
from gaussian_splatting.utils import quaternion_to_matrix

from ..gaussian_model import render_normal
from .gsplat import CameraTrainableGsplatPGSRGaussianModel, render_plane


def plane_params(
    viewpoint_camera: Camera,
    means3D: torch.Tensor,
    # scales: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Build camera-facing [normal.xyz, plane_distance] per 2D Gaussian."""
    rotation_matrices = quaternion_to_matrix(F.normalize(rotations, dim=-1))
    global_normal = rotation_matrices[..., :, 2]
    camera_center = viewpoint_camera.camera_center.to(device=means3D.device, dtype=means3D.dtype)
    gaussian_to_cam_global = camera_center - means3D
    neg_mask = (global_normal * gaussian_to_cam_global).sum(-1, keepdim=True) < 0.0
    global_normal = torch.where(neg_mask, -global_normal, global_normal)

    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]

    local_distance = (local_normal * pts_in_cam).sum(-1, keepdim=True).abs()
    input_all_map = torch.cat((local_normal, local_distance), dim=-1)
    return input_all_map


class Gsplat2DGSPGSRGaussianModel(Gsplat2DGSGaussianModel):
    """PGSR maps using the same alpha compositing as the 2DGS renderer."""

    def __init__(self, sh_degree, min_scale=1e-6, render_depth_normal: bool = False):
        super().__init__(sh_degree, min_scale=min_scale)
        self.render_depth_normal = render_depth_normal

    def render(
        self,
        viewpoint_camera,
        means3D,
        opacity,
        scales,
        rotations,
        shs,
        colors_precomp=None,
        cov3D_precomp=None,
    ):
        # Base render flow adapted from https://github.com/yindaheng98/gaussian-splatting/blob/3c996b3f007a268353d73902f4efd04425dda5f1/gaussian_splatting/models/gsplat_2dgs.py
        width = int(viewpoint_camera.image_width)
        height = int(viewpoint_camera.image_height)
        device = means3D.device

        viewmats = viewpoint_camera.world_view_transform.T[None]

        Ks = viewpoint_camera.K.to(device=device, dtype=means3D.dtype)[None]

        input_all_map = plane_params(viewpoint_camera, means3D, rotations)

        camera_center = torch.linalg.inv(viewmats[0])[:3, 3]
        rgb = torch.clamp_min(spherical_harmonics(self.active_sh_degree, means3D - camera_center, shs) + 0.5, 0.0)

        (
            render_colors,
            render_alphas,
            _render_normals,
            _normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means3D,
            rotations,
            scales,
            opacity,
            # Gsplat-style renderers have no extra_signals argument, so append
            # PGSR plane channels after RGB and let the rasterizer alpha-composite
            # them together.
            torch.cat((rgb, input_all_map), dim=-1)[None],
            viewmats,
            Ks,
            width,
            height,
            sh_degree=None,
            # Distortion regularization requires a depth render mode.  The
            # appended expected-depth channel also pads RGB + four PGSR plane
            # channels to the CUDA rasterizer's supported eight channels.
            render_mode="RGB+ED",
            packed=False,
            # Only RGB uses the image background; PGSR's appended channels use
            # zero background so uncovered pixels contribute no plane signal.
            backgrounds=torch.cat((viewpoint_camera.bg_color.to(means3D), torch.zeros(4, device=device, dtype=means3D.dtype)))[None],
            distloss=True,
            depth_mode="expected",
        )

        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)
        out_all_map = render_colors[..., 3:7]

        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)
        rendered_image = rendered_image.clamp(0, 1)

        radii = info["radii"][0].max(dim=-1).values

        gradient_2dgs = info["gradient_2dgs"]
        try:
            gradient_2dgs.retain_grad()
        except:
            pass

        depth, rendered_normal, rendered_distance, render_alphas = render_plane(out_all_map, render_alphas, viewpoint_camera.K)
        plane_outputs = {
            "depth": depth,
            "invdepth": 1 / depth,
            "render_normals": rendered_normal,
            "render_alphas": render_alphas,
            "rendered_distance": rendered_distance,
        }
        if self.render_depth_normal:
            # https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/gaussian_renderer/__init__.py#L173-L175
            depth_normal = render_normal(viewpoint_camera, depth.squeeze(0)) * render_alphas.detach()
            plane_outputs["normals_from_depth"] = depth_normal
        return {
            "render": rendered_image,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "get_viewspace_grad": lambda out: out["gradient_2dgs"].grad.squeeze(0) * out["gradient_2dgs"].new_tensor([[width, height]]) / 2.0,
            "gradient_2dgs": gradient_2dgs,
            "render_distort": render_distort,
            "render_median": render_median,
            **plane_outputs,
        }


class CameraTrainableGsplat2DGSPGSRGaussianModel(Gsplat2DGSPGSRGaussianModel):
    def forward(self, viewpoint_camera: Camera):
        # https://github.com/yindaheng98/gaussian-splatting/blob/3c996b3f007a268353d73902f4efd04425dda5f1/gaussian_splatting/models/gsplat_2dgs.py#L168-L170
        return CameraTrainableGsplatPGSRGaussianModel.forward(self, viewpoint_camera)
