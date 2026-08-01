import torch
import torch.nn.functional as F


# Source: https://github.com/yindaheng98/PostRenderPerspectiveAlign/blob/86967a863d01f8eb5c56d82a1283d9c3e2f94bdb/prpa/reproj.py#L5-L14
def reconstruction(K: torch.Tensor, R_c2w: torch.Tensor, T_c2w: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Reconstruct point cloud from camera and depth map"""
    height, width = depth.shape
    uv = torch.ones((height, width, 3), dtype=depth.dtype, device=depth.device)
    uv[..., 0] = torch.arange(0, width, dtype=depth.dtype, device=depth.device).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=depth.dtype, device=depth.device).unsqueeze(1).expand(-1, width)
    xyz_camera = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
    # xyz_camera = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)).T*1000
    xyz_world = R_c2w @ xyz_camera + T_c2w.unsqueeze(1)
    return xyz_world.T.reshape(*uv.shape)


# Source: https://github.com/yindaheng98/PostRenderPerspectiveAlign/blob/86967a863d01f8eb5c56d82a1283d9c3e2f94bdb/prpa/reproj.py#L17-L24
def projection(K: torch.Tensor, R_c2w: torch.Tensor, T_c2w: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project point cloud to camera"""
    height, width = xyz.shape[:2]
    xyz_world = xyz.reshape(-1, 3).T
    xyz_camera = torch.inverse(R_c2w) @ (xyz_world - T_c2w.unsqueeze(1))
    uvz = K @ xyz_camera
    uv = (uvz/uvz[-1, ...]).T.reshape(height, width, 3)
    return uv, uvz[-1, ...].reshape(height, width)


def visibility(
        K: torch.Tensor,
        R_c2w: torch.Tensor,
        T_c2w: torch.Tensor,
        depth: torch.Tensor,
        xyz: torch.Tensor,
        depth_threshold: float,
        min_depth: float = 0.01,  # Same as Gaussian Splatting Camera.znear.
) -> torch.Tensor:
    """Return whether world-space points are visible in the camera depth map."""
    uv, z = projection(K, R_c2w, T_c2w, xyz)
    pixels = uv[..., :2]
    height, width = depth.shape

    grid = pixels.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
    rendered_depth = F.grid_sample(
        depth[None, None],
        grid.reshape(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0, :, 0].reshape(z.shape)

    return (
        (z > min_depth)
        & (pixels[..., 0] > 0)
        & (pixels[..., 0] < width)
        & (pixels[..., 1] > 0)
        & (pixels[..., 1] < height)
        & torch.isfinite(rendered_depth)
        & (rendered_depth > 0)
        & ((z - rendered_depth) < depth_threshold * rendered_depth)
    )
