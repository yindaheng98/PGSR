# PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction (Python Package Version)

[![PyPI version](https://img.shields.io/pypi/v/pgsr.svg?logo=pypi)](https://pypi.org/project/pgsr/)
[![Downloads](https://api.pepy.tech/personalized-badge/pgsr?period=month&left_color=grey&right_color=brightgreen&left_text=monthly%20downloads)](https://pepy.tech/project/pgsr)
[![Total downloads](https://api.pepy.tech/personalized-badge/pgsr?period=total&left_color=grey&right_color=brightgreen&left_text=total%20downloads)](https://pepy.tech/project/pgsr)
[![Build](https://github.com/yindaheng98/PGSR/actions/workflows/build-release.yml/badge.svg)](https://github.com/yindaheng98/PGSR/actions/workflows/build-release.yml)

This repository contains the **refactored Python package for [PGSR](https://github.com/zju3dv/PGSR)**. It is ported from commit [de24f1a38b350387e8d8fe381b2cd70c1ae946e7](https://github.com/zju3dv/PGSR/tree/de24f1a38b350387e8d8fe381b2cd70c1ae946e7). The original components have been reorganized into a standard Python package and adapted to the reusable APIs provided by [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting).

## Features

* [x] Code organized as a standard Python package
* [x] `gsplat` and `gsplat-2dgs` rendering backends
* [x] Planar scale regularization
* [x] Depth-normal consistency
* [x] Multi-view photometric and geometric regularization
* [x] Virtual-camera reprojection
* [x] Multi-view trimming and opacity-reset densification
* [x] Optional camera-pose optimization
* [x] Rendering, mesh extraction, and interactive viewing

## Prerequisites

* [PyTorch](https://pytorch.org/) with CUDA support
* A CUDA Toolkit version compatible with the installed PyTorch build
* Python 3.10 or later

Optional features can be installed through package extras:

```shell
pip install --upgrade "pgsr[mesh,viewer]"
```

If you have trouble installing [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting), install it from source:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
```

## PyPI Install

```shell
pip install --upgrade pgsr
```

Or install the latest version from source:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/PGSR.git@main --no-build-isolation
```

### Development Install

```shell
git clone https://github.com/yindaheng98/PGSR.git
cd PGSR
pip install --editable .
```

## Quick Start

1. Prepare a dataset in the COLMAP format used by [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting). For example, download the Tanks and Temples + Deep Blending dataset:

```shell
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip -P ./data
unzip data/tandt_db.zip -d data/
```

2. Train PGSR with densification:

```shell
python -m pgsr.train -s data/truck -d output/truck -i 30000 --mode densify --backend gsplat --no_image_mask --no_depth_data
```

3. Render the trained model:

```shell
python -m pgsr.render -s data/truck -d output/truck -i 30000 --backend gsplat --no_image_mask
```

4. Extract a mesh (requires the `mesh` extra):

```shell
python -m pgsr.mesh -s data/truck -d output/truck -i 30000 --backend gsplat --no_image_mask -o max_depth=10.0 -o voxel_size=0.01
```

5. Open the interactive viewer (requires the `viewer` extra):

```shell
python -m pgsr.viewer -d output/truck -i 30000 --backend gsplat --port 8080
```

> 💡 This package does not include dataset preprocessing or evaluation scripts. Refer to the [original PGSR repository](https://github.com/zju3dv/PGSR) for the DTU, Tanks and Temples, and Mip-NeRF 360 workflows.

> 💡 See [.vscode/launch.json](.vscode/launch.json) for more examples. Run `python -m pgsr.train --help`, `python -m pgsr.render --help`, `python -m pgsr.mesh --help`, or `python -m pgsr.viewer --help` for all command-line options.

## Backends and Training Modes

Two rendering backends are available:

* `gsplat` (default)
* `gsplat-2dgs`

The training entry point supports the following modes:

* `base`: PGSR regularization without densification
* `densify` (default): PGSR regularization with multi-view trimming, opacity reset, and densification
* `camera`: `base` with trainable camera poses
* `camera-densify`: `densify` with trainable camera poses

Use repeated `-o key=value` arguments to override trainer configuration values:

```shell
python -m pgsr.train -s data/truck -d output/truck -o densify_grad_threshold=0.0001 -o opacity_cull_threshold=0.05
```

## API Usage

This project builds on [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) and provides PGSR Gaussian models and composed trainers. Refer to that package for the core Gaussian model, dataset, trainer, and training-loop concepts.

The high-level factory prepares the dataset, Gaussian model, and trainer:

```python
from pgsr.train import prepare_training

dataset, gaussians, trainer = prepare_training(
    sh_degree=3,
    source="data/truck",
    device="cuda",
    mode="densify",
    backend="gsplat",
    load_mask=False,
    load_depth=False,
    configs={"densify_grad_threshold": 0.0001},
)
```

The lower-level factories can also be used independently:

```python
from gaussian_splatting.prepare import prepare_dataset
from pgsr.prepare import prepare_gaussians, prepare_trainer

dataset = prepare_dataset(
    source="data/truck",
    device="cuda",
    load_mask=False,
    load_depth=False,
)
gaussians = prepare_gaussians(
    sh_degree=3,
    source="data/truck",
    device="cuda",
    backend="gsplat",
)
trainer = prepare_trainer(
    gaussians=gaussians,
    dataset=dataset,
    mode="densify",
    configs={"densify_grad_threshold": 0.0001},
)
```

# PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction
Danpeng Chen, Hai Li, [Weicai Ye](https://ywcmaike.github.io/), Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/)
### [Project Page](https://zju3dv.github.io/pgsr/) | [arXiv](https://arxiv.org/abs/2406.06521)
![Teaser image](assets/teaser.jpg)

We present a Planar-based Gaussian Splatting Reconstruction representation for efficient and high-fidelity surface reconstruction from multi-view RGB images without any geometric prior (depth or normal from pre-trained model).  

## Updates
- [2024.07.18]: We fine-tuned the hyperparameters based on the original paper. The Chamfer Distance on the DTU dataset decreased to 0.47.

The Chamfer Distance↓ on the DTU dataset
|     | 24| 37| 40| 55| 63| 65| 69| 83| 97|105|106|110|114|118|122|Mean|Time|
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|PGSR(Paper)|0.34|0.58|0.29|0.29|0.78|0.58|0.54|1.01|0.73|0.51|0.49|0.69|0.31|0.37|0.38|0.53|0.6h|
|PGSR(Code_V1.0)|0.33|0.51|0.29|0.28|0.75|0.53|0.46|0.92|0.62|0.48|0.45|0.55|0.29|0.33|0.31|0.47|0.5h|
|PGSR(Remove ICP)|0.36|0.57|0.38|0.33|0.78|0.58|0.50|1.08|0.63|0.59|0.46|0.54|0.30|0.38|0.34|0.52|0.5h|

The F1 Score↑ on the TnT dataset
||PGSR(Paper)|PGSR(Code_V1.0)
|-|-|-|
|Barn|0.66|0.65
|Caterpillar|0.41|0.44
|Courthouse|0.21|0.20
|Ignatius|0.80|0.81
|Meetingroom|0.29|0.32
|Truck|0.60|0.66
|Mean|0.50|0.51
|Time|1.2h|45m

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Densify is based on [AbsGau](https://ty424.github.io/AbsGS.github.io/) and [GOF](https://github.com/autonomousvision/gaussian-opacity-fields?tab=readme-ov-file). DTU and Tanks and Temples dataset preprocess are based on [Neuralangelo scripts](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md). Evaluation scripts for DTU and Tanks and Temples dataset are based on [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation) respectively. We thank all the authors for their great work and repos. 


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{chen2024pgsr,
  title={PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction},
  author={Chen, Danpeng and Li, Hai and Ye, Weicai and Wang, Yifan and Xie, Weijian and Zhai, Shangjin and Wang, Nan and Liu, Haomin and Bao, Hujun and Zhang, Guofeng},
  journal={arXiv preprint arXiv:2406.06521},
  year={2024}
}
```
