import os
from argparse import ArgumentParser

from gaussian_splatting.mesh import extract_mesh

from pgsr.prepare import backends
from pgsr.render import prepare_rendering


if __name__ == "__main__":
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
    parser.add_argument("-o", "--option", default=[], action="append", type=str)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    dataset, gaussians = prepare_rendering(
        sh_degree=args.sh_degree, source=args.source, device=args.device,
        trainable_camera=args.mode == "camera", load_ply=load_ply, load_camera=args.load_camera,
        load_mask=not args.no_image_mask, load_depth=False, backend=args.backend,
    )
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    extract_mesh(dataset, gaussians, os.path.join(args.destination, f"ours_{args.iteration}"), **configs)
