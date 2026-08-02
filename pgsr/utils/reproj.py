from typing import Optional

import torch
import torch.nn.functional as F


def reconstruction(K: torch.Tensor, R_c2w: torch.Tensor, T_c2w: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Reconstruct point cloud from camera and depth map"""
    height, width = depth.shape
    uv = torch.empty((height, width, 2), dtype=depth.dtype, device=depth.device)
    uv[..., 0] = torch.arange(0, width, dtype=depth.dtype, device=depth.device).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=depth.dtype, device=depth.device).unsqueeze(1).expand(-1, width)
    return reconstruct_pixels(K, R_c2w, T_c2w, uv, depth)


def reconstruct_pixels(
        K: torch.Tensor,
        R_c2w: torch.Tensor,
        T_c2w: torch.Tensor,
        pixels: torch.Tensor,
        depth: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct world-space points from pixel coordinates and matching depths."""
    shape = pixels.shape[:-1]
    if shape != depth.shape:
        raise ValueError(f"pixels leading shape {shape} must match depth shape {depth.shape}")
    depth_flat = depth.reshape(-1)
    pixels_h = torch.cat((
        pixels.reshape(-1, 2),
        torch.ones((depth_flat.shape[0], 1), device=depth.device, dtype=depth.dtype),
    ), dim=-1)
    xyz_camera = torch.inverse(K) @ pixels_h.T * depth_flat
    xyz_world = R_c2w @ xyz_camera + T_c2w.unsqueeze(1)
    return xyz_world.T.reshape(*shape, 3)


# Source: https://github.com/yindaheng98/PostRenderPerspectiveAlign/blob/86967a863d01f8eb5c56d82a1283d9c3e2f94bdb/prpa/reproj.py#L17-L24
def projection(K: torch.Tensor, R_c2w: torch.Tensor, T_c2w: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to camera pixels, preserving all leading dimensions."""
    shape = xyz.shape[:-1]
    xyz_world = xyz.reshape(-1, 3).T
    xyz_camera = torch.inverse(R_c2w) @ (xyz_world - T_c2w.unsqueeze(1))
    uvz = K @ xyz_camera
    uv = (uvz / uvz[-1, ...]).T.reshape(*shape, 3)
    return uv, uvz[-1, ...].reshape(*shape)


def reprojection(
        source_K: torch.Tensor,
        source_R_c2w: torch.Tensor,
        source_T_c2w: torch.Tensor,
        source_depth: torch.Tensor,
        target_K: torch.Tensor,
        target_R_c2w: torch.Tensor,
        target_T_c2w: torch.Tensor,
        target_depth: torch.Tensor,
        min_depth: float = 0.01,
        max_depth: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project source depth pixels to a target view, sample target depth, and project back."""
    height, width = source_depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=source_depth.device, dtype=source_depth.dtype),
        torch.arange(width, device=source_depth.device, dtype=source_depth.dtype),
        indexing="ij",
    )
    source_pixels = torch.stack((x, y), dim=-1).reshape(-1, 2)

    source_world = reconstruction(source_K, source_R_c2w, source_T_c2w, source_depth).reshape(-1, 3)
    target_uv, target_z = projection(target_K, target_R_c2w, target_T_c2w, source_world)
    target_pixels = target_uv[:, :2]

    target_height, target_width = target_depth.shape
    valid_mask = (
        (target_pixels[:, 0] > 0) & (target_pixels[:, 0] < target_width)
        & (target_pixels[:, 1] > 0) & (target_pixels[:, 1] < target_height)
        & (target_z > min_depth) & (target_z < max_depth)
    )
    source_pixels = source_pixels[valid_mask]
    target_pixels = target_pixels[valid_mask]
    if target_pixels.shape[0] == 0:
        return source_pixels, source_pixels.new_empty((0, 3)), source_pixels.new_empty((0,))

    grid = target_pixels.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / (target_width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (target_height - 1) - 1.0
    target_depth_samples = F.grid_sample(
        target_depth[None, None],
        grid.view(1, -1, 1, 2),
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )[0, 0, :, 0]
    valid_mask = (target_depth_samples > min_depth) & (target_depth_samples < max_depth)
    source_pixels = source_pixels[valid_mask]
    target_pixels = target_pixels[valid_mask]
    target_depth_samples = target_depth_samples[valid_mask]
    if target_pixels.shape[0] == 0:
        return source_pixels, source_pixels.new_empty((0, 3)), source_pixels.new_empty((0,))

    reprojected_world = reconstruct_pixels(
        target_K, target_R_c2w, target_T_c2w, target_pixels, target_depth_samples,
    )
    source_reprojected_uv, source_reprojected_z = projection(
        source_K, source_R_c2w, source_T_c2w, reprojected_world,
    )
    return source_pixels, source_reprojected_uv, source_reprojected_z


def visibility(
        K: torch.Tensor,
        R_c2w: torch.Tensor,
        T_c2w: torch.Tensor,
        depth: torch.Tensor,
        xyz: torch.Tensor,
        relative_depth_tolerance: float,
        min_depth: float = 0.01,  # Same as Gaussian Splatting Camera.znear.
        max_depth: float = 100.0,  # Depth values farther than this are treated as invalid.
        depth_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return whether world-space points are visible in the camera depth map.

    relative_depth_tolerance is a fraction of the sampled rendered depth.
    """
    uv, z = projection(K, R_c2w, T_c2w, xyz)
    pixels = uv[..., :2]
    height, width = depth.shape

    # grid_sample expects normalized coordinates in [-1, 1].
    grid = pixels.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
    # Sample the rendered depth map at each projected point location.
    rendered_depth = F.grid_sample(
        depth[None, None],
        grid.reshape(1, -1, 1, 2),
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )[0, 0, :, 0].reshape(z.shape)
    rendered_mask = torch.ones_like(rendered_depth, dtype=torch.bool)
    if depth_mask is not None:
        rendered_mask = F.grid_sample(
            depth_mask.float()[None, None],
            grid.reshape(1, -1, 1, 2),
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, :, 0].reshape(z.shape) > 0.5

    return (  # A point is visible only if all conditions below are satisfied.
        (z > min_depth) & (z < max_depth)  # The point must be inside the valid depth range.
        & (pixels[..., 0] > 0) & (pixels[..., 0] < width)  # The projected x coordinate must be inside the image.
        & (pixels[..., 1] > 0) & (pixels[..., 1] < height)  # The projected y coordinate must be inside the image.
        & rendered_mask
        & (rendered_depth > min_depth) & (rendered_depth < max_depth)  # The sampled depth must also be valid.
        & ((z - rendered_depth) < relative_depth_tolerance * rendered_depth)  # Reject points much farther than the depth map.
    )
