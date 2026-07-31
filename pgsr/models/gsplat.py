import torch
from gsplat import rasterization

from gaussian_splatting.models.gsplat import GsplatGaussianModel

from ..gaussian_model import plane_params, render_normal


def render_plane(
    out_all_map: torch.Tensor,
    render_alphas: torch.Tensor,
    K: torch.Tensor,
) -> dict:
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
    depth = plane_depth[0].permute(2, 0, 1)

    # https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/gaussian_renderer/__init__.py#L153-L165
    return {
        "render_normals": rendered_normal[0].permute(2, 0, 1),
        "render_alphas": render_alphas[0].permute(2, 0, 1),
        "depth": depth,
        "invdepth": 1 / depth,

        # PGSR specific output
        "rendered_distance": rendered_distance[0].permute(2, 0, 1),
    }


class GsplatPGSRGaussianModel(GsplatGaussianModel):
    """PGSR geometry maps rendered through gsplat ``extra_signals``."""

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

        render_colors, render_alphas, info = rasterization(
            means3D,
            rotations,
            scales,
            opacity,
            shs,
            viewmats,
            Ks,
            width,
            height,
            sh_degree=self.active_sh_degree,
            render_mode="RGB",
            packed=False,
            absgrad=True,
            rasterize_mode="antialiased" if self.antialiasing else "classic",
            backgrounds=viewpoint_camera.bg_color[None],
            # gsplat extra_signals carries PGSR's input_all_map through the RGB
            # visibility, coverage and alpha compositing path.
            extra_signals=input_all_map[None],
        )

        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)

        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)
        rendered_image = rendered_image.clamp(0, 1)

        radii = info["radii"][0].max(dim=-1).values

        means2d = info["means2d"]
        try:
            means2d.retain_grad()
        except:
            pass

        return_dict = {
            "render": rendered_image,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "get_viewspace_grad": lambda out: out["means2d"].grad.squeeze(0) * out["means2d"].new_tensor([[width, height]]) / 2.0,
            "means2d": means2d,
        }
        return_dict.update(
            render_plane(info["render_extra_signals"], render_alphas, viewpoint_camera.K)
        )
        if self.render_depth_normal:
            # https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/gaussian_renderer/__init__.py#L173-L175
            depth_normal = render_normal(viewpoint_camera, return_dict["depth"].squeeze()) * return_dict["render_alphas"].detach()
            return_dict.update({"normals_from_depth": depth_normal})
        return return_dict
