import os
from typing import Tuple

import torch

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset
from gaussian_splatting.render import rendering

from pgsr.prepare import backends, prepare_gaussians


def prepare_rendering(
        sh_degree: int, source: str, device: str,
        trainable_camera: bool = False, load_ply: str = None, load_camera: str = None,
        load_mask=True, load_depth=True, backend: str = "gsplat") -> Tuple[CameraDataset, GaussianModel]:
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_mask=load_mask, load_depth=load_depth)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply, backend=backend)
    return dataset, gaussians


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--backend", choices=backends, default="gsplat")
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("--no_rescale_depth_gt", action="store_true")
    parser.add_argument("--save_depth_pcd", action="store_true")
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera,
            load_mask=not args.no_image_mask, load_depth=args.save_depth_pcd, backend=args.backend)
        rendering(dataset, gaussians, save, save_pcd=args.save_depth_pcd, rescale_depth_gt=not args.no_rescale_depth_gt)
