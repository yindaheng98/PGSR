import torch
from gsplat import rasterization, spherical_harmonics

from gaussian_splatting import Camera
from gaussian_splatting.models.gsplat import (
    CameraTrainableGsplatGaussianModel,
    GsplatGaussianModel,
)

from ..gaussian_model import plane_params, render_normal


def render_plane(
    out_all_map: torch.Tensor,
    render_alphas: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build PGSR normal, distance and ray-plane depth outputs."""
    # https://github.com/zju3dv/PGSR/blob/e83f5cb41a49cc512964af11a794502aaa32cc8d/submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu#L303-L304
    _, height, width, _ = out_all_map.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=out_all_map.device, dtype=out_all_map.dtype),
        torch.arange(width, device=out_all_map.device, dtype=out_all_map.dtype),
        indexing="ij",
    )
    rays = torch.stack(
        (
            (x - K[0, 2]) / K[0, 0],
            (y - K[1, 2]) / K[1, 1],
            torch.ones_like(x),
        ),
        dim=-1,
    )

    rendered_normal = out_all_map[..., :3]
    rendered_distance = out_all_map[..., 3:4]
    # https://github.com/zju3dv/PGSR/blob/e83f5cb41a49cc512964af11a794502aaa32cc8d/submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu#L404
    plane_depth = rendered_distance / -((rendered_normal * rays[None]).sum(-1, keepdim=True) + 1.0e-8)
    # Match original PGSR: zero-initialized plane maps leave uncovered depth
    # at zero; no validity policy is applied by the renderer.
    depth = plane_depth[0].permute(2, 0, 1)

    # https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/gaussian_renderer/__init__.py#L153-L165
    return depth, rendered_normal[0].permute(2, 0, 1), rendered_distance[0].permute(2, 0, 1), render_alphas[0].permute(2, 0, 1)


class GsplatPGSRGaussianModel(GsplatGaussianModel):
    """PGSR geometry maps rendered as alpha-composited gsplat features."""

    def __init__(self, sh_degree, render_depth_normal: bool = False):
        super().__init__(sh_degree)
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
        # Base render flow adapted from https://github.com/yindaheng98/gaussian-splatting/blob/3c996b3f007a268353d73902f4efd04425dda5f1/gaussian_splatting/models/gsplat.py
        width = int(viewpoint_camera.image_width)
        height = int(viewpoint_camera.image_height)
        device = means3D.device

        viewmats = viewpoint_camera.world_view_transform.T[None]

        Ks = viewpoint_camera.K.to(device=device, dtype=means3D.dtype)[None]

        input_all_map = plane_params(viewpoint_camera, means3D, scales, rotations)

        camera_center = torch.linalg.inv(viewmats[0])[:3, 3]
        rgb = torch.clamp_min(spherical_harmonics(self.active_sh_degree, means3D - camera_center, shs) + 0.5, 0.0)

        render_colors, render_alphas, info = rasterization(
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
            render_mode="RGB+ED",
            packed=False,
            absgrad=True,
            rasterize_mode="antialiased" if self.antialiasing else "classic",
            # Only RGB uses the image background; PGSR's appended channels use
            # zero background so uncovered pixels contribute no plane signal.
            backgrounds=torch.cat((viewpoint_camera.bg_color.to(means3D), torch.zeros(4, device=device, dtype=means3D.dtype)))[None],
        )

        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)
        out_all_map = render_colors[..., 3:7]

        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)
        rendered_image = rendered_image.clamp(0, 1)

        radii = info["radii"][0].max(dim=-1).values

        means2d = info["means2d"]
        try:
            means2d.retain_grad()
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
            depth_normal = render_normal(viewpoint_camera, plane_outputs["depth"].squeeze()) * plane_outputs["render_alphas"].detach()
            plane_outputs.update({"normals_from_depth": depth_normal})
        return {
            "render": rendered_image,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "get_viewspace_grad": lambda out: out["means2d"].grad.squeeze(0) * out["means2d"].new_tensor([[width, height]]) / 2.0,
            "means2d": means2d,
            **plane_outputs,
        }


class CameraTrainableGsplatPGSRGaussianModel(GsplatPGSRGaussianModel):
    def forward(self, viewpoint_camera: Camera):
        # https://github.com/yindaheng98/gaussian-splatting/blob/3c996b3f007a268353d73902f4efd04425dda5f1/gaussian_splatting/models/gsplat.py#L117-L140
        return CameraTrainableGsplatGaussianModel.forward(self, viewpoint_camera)
