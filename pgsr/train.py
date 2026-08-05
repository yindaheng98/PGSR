import os
from argparse import ArgumentParser
from typing import Tuple

import torch

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset
from gaussian_splatting.train import save_cfg_args, training
from gaussian_splatting.trainer import AbstractTrainer

from pgsr.prepare import backends, basemodes, prepare_gaussians, prepare_trainer, shliftmodes


def prepare_training(
        sh_degree: int, source: str, device: str, mode: str,
        trainable_camera: bool = False, load_ply: str = None, load_camera: str = None,
        load_mask=True, load_depth=True, backend: str = "gsplat",
        configs={}) -> Tuple[CameraDataset, GaussianModel, AbstractTrainer]:
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_mask=load_mask, load_depth=load_depth)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply, backend=backend)
    trainer = prepare_trainer(gaussians=gaussians, dataset=dataset, mode=mode, load_ply=load_ply, configs=configs)
    return dataset, gaussians, trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--backend", choices=backends, default="gsplat")
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "-m", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", "--iterations", default=30000, type=int)
    parser.add_argument("-l", "--load_ply", default=None, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("--no_depth_data", action="store_true")
    parser.add_argument("--mode", choices=sorted(list(set(list(basemodes.keys()) + list(shliftmodes.keys())))), default="densify")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--empty_cache_every_step", action='store_true')
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()

    save_cfg_args(args.destination, args.sh_degree, args.source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, gaussians, trainer = prepare_training(
        sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode, trainable_camera="camera" in args.mode,
        load_ply=args.load_ply, load_camera=args.load_camera,
        load_mask=not args.no_image_mask, load_depth=not args.no_depth_data, backend=args.backend,
        configs=configs)
    dataset.save_cameras(os.path.join(args.destination, "cameras.json"))
    torch.cuda.empty_cache()
    training(
        dataset=dataset, gaussians=gaussians, trainer=trainer,
        destination=args.destination, iteration=args.iteration, save_iterations=args.save_iterations,
        empty_cache_every_step=args.empty_cache_every_step)
