import os

import torch

from gaussian_splatting.viewer import viewer_render_fn, viewing

from pgsr.prepare import backends, prepare_gaussians


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--backend", choices=backends, default="gsplat")
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    with torch.no_grad():
        gaussians = prepare_gaussians(
            sh_degree=args.sh_degree, source=args.destination, device=args.device,
            load_ply=load_ply, backend=args.backend)
        viewing(gaussians, device=args.device, port=args.port)
