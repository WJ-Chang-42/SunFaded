# SunFaded

**Illumination-Aware Gaussian Splatting for Dark Scenes with Camera-Mounted Active Lighting**

Wenjie Chang, Tianle Ding, Wenfei Yang, Tianzhu Zhang

CVPR 2026 · [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Chang_SunFaded_Illumination-Aware_Gaussian_Splatting_for_Dark_Scenes_with_Camera-Mounted_Active_CVPR_2026_paper.html)

## Installation

Tested environment: Linux, Python 3.10, PyTorch 2.3.1, CUDA toolkit 12.1, NVIDIA RTX
3090. CPU-only training is not supported. Install a matching NVIDIA driver and
CUDA toolkit/compiler before building the extensions.

Clone this repository with `--recurse-submodules` into a directory named
`SunFaded`. If you have already cloned it, initialize the pinned dependencies
with the command below.

```bash
cd SunFaded
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install setuptools==65.5.0 wheel==0.45.1
python -m pip install -r requirements-release.txt
python -m pip install --no-build-isolation --no-deps 'depth-pro @ git+https://github.com/apple/ml-depth-pro.git@9efe5c1def37a26c5367a71df664b18e1306c708'
python -m pip install --no-build-isolation --no-deps ./submodules/diff-surfel-rasterization ./submodules/simple-knn
python -m pip check
```

The two CUDA dependencies are Git submodules; GLM is a nested submodule of
the rasterizer. The main repository records their upstream URLs and fixed
commits, not their source files. Source ZIP downloads do not include these
dependencies; use a recursive Git clone for installation. Use
`requirements-release.txt` for SunFaded; `environment.yml` belongs to the
upstream project. `CUDA_HOME` must point to the matching toolkit. Set
`TORCH_CUDA_ARCH_LIST` for your GPU when cross-compiling (8.6 for RTX 3090).

Download the official Depth Pro checkpoint separately, subject to its own terms:

```bash
mkdir -p checkpoints
curl -L --fail https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
```

The default path is relative to this repository, not the launch directory.
Override it with `--depth_pro_checkpoint /path/to/depth_pro.pt`. Rendering a saved
complete state does not require the Depth Pro checkpoint. LPIPS uses pretrained
AlexNet/LPIPS weights and may download them on first use.

## Data

We use the dataset provided by [DarkGS](https://github.com/tyz1030/darkgs#data).
For details of our self-captured data, please refer to our
[supplementary material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Chang_SunFaded_Illumination-Aware_Gaussian_CVPR_2026_supplemental.pdf).

## Training

```bash
python train.py -s /path/to/scene -m output/my_scene -r 1 \
  --config config/training_config.json --seed 0 \
  --depth_pro_checkpoint /path/to/depth_pro.pt
```

The example uses 40,000 iterations and seed 0. Select a GPU with
`CUDA_VISIBLE_DEVICES` if needed.

For a short installation check, add `--iterations 20 --test_iterations 20
--save_iterations 20 --skip_export`. Short checks do not establish convergence.
`--test_iterations` controls the reconstruction-reporting schedule, not a
held-out test set. Intermediate reports sample five training views; the final
export covers every training view. Pass `--config config/training_config.json`
to use the supplied schedule. Omitting `--config` selects a different built-in
schedule.

Non-empty output directories are rejected. Each run stores the effective
configuration, seed, camera manifest, native visualization files and a complete
`point_cloud/iteration_40000/inference_state.pth`. `--checkpoint_iterations`
saves additional complete **inference** snapshots; `--start_checkpoint` fails
explicitly because exact training resume is not supported.

## Citation

If you use SunFaded in your research, please cite:

```bibtex
@InProceedings{Chang_2026_CVPR,
    author    = {Chang, Wenjie and Ding, Tianle and Yang, Wenfei and Zhang, Tianzhu},
    title     = {SunFaded: Illumination-Aware Gaussian Splatting for Dark Scenes with Camera-Mounted Active Lighting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {40876-40885}
}
```

## Acknowledgements

Built on [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting), with
[Depth Pro](https://github.com/apple/ml-depth-pro).
Retain the original citations and attribution when using the underlying work.

We thank [DarkGS](https://github.com/tyz1030/darkgs) and Hakushi Hasegawa
for their great work.

## License

SunFaded is distributed under the [Gaussian-Splatting license](LICENSE.md)
for non-commercial research and evaluation. Third-party code, pretrained models
and datasets have their own terms. See [third-party notices](THIRD_PARTY.md).
No dataset or pretrained checkpoint is included.
