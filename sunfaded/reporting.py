# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact george.drettakis@inria.fr

"""Training reconstruction reporting; original sampling and metric scheduling are retained."""

import os
import uuid
from argparse import Namespace
import torch
from scene import Scene
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.general_utils import colormap


def prepare_output_and_logger(args):
    """Create a fresh output directory and record model arguments. TensorBoard remains disabled as in the source revision."""
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print(f"Output directory: {args.model_path}")
    if os.path.isdir(args.model_path) and os.listdir(args.model_path):
        raise FileExistsError(
            f"Refusing to overwrite non-empty model output: {args.model_path}"
        )
    os.makedirs(args.model_path, exist_ok=True)
    config_path = os.path.join(args.model_path, "cfg_args")
    with open(config_path, "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    light_list=None,
    normal_list=None,
):
    """Report only training reconstruction, retaining the original preview sampling and metric-computation thresholds."""
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/reg_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )
    if iteration in testing_iterations:
        print(
            f"\n[Iteration {iteration}] Training-set reconstruction metrics (no held-out views)."
        )
        torch.cuda.empty_cache()
        if iteration < 39999:
            validation_configs = (
                {
                    "name": "train_preview",
                    "cameras": [
                        scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                        for idx in range(5, 30, 5)
                    ],
                },
            )
        else:
            validation_configs = (
                {"name": "train_all", "cameras": scene.getTrainCameras()},
            )
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    with torch.no_grad():
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    albedo = render_pkg["albedo"]
                    if tb_writer and idx < 5:
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/depth",
                            depth[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/render",
                            image[None],
                            global_step=iteration,
                        )
                        try:
                            if "rend_alpha" in render_pkg:
                                rend_alpha = render_pkg["rend_alpha"]
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/rend_alpha",
                                    rend_alpha[None],
                                    global_step=iteration,
                                )
                            if "rend_normal" in render_pkg:
                                rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/rend_normal",
                                    rend_normal[None],
                                    global_step=iteration,
                                )
                            if "surf_normal" in render_pkg:
                                surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/surf_normal",
                                    surf_normal[None],
                                    global_step=iteration,
                                )
                            if "rend_dist" in render_pkg:
                                rend_dist = render_pkg["rend_dist"]
                                rend_dist = colormap(rend_dist.cpu().numpy()[0])
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/rend_dist",
                                    rend_dist[None],
                                    global_step=iteration,
                                )
                        except Exception as e:
                            pass
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if iteration > 25000:
                        ssim_test += ssim(image, gt_image).mean().double()
                        lpips_test += lpips(image, gt_image).mean().double()
                num_views = len(config["cameras"])
                psnr_test /= num_views
                l1_test /= num_views
                ssim_test /= num_views
                lpips_test /= num_views
                print(
                    f"\n[Iteration {iteration}] Training reconstruction {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.2f} SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f}"
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    if iteration > 35000:
                        tb_writer.add_scalar(
                            f"{config['name']}/loss_viewpoint - ssim",
                            ssim_test,
                            iteration,
                        )
                        tb_writer.add_scalar(
                            f"{config['name']}/loss_viewpoint - lpips",
                            lpips_test,
                            iteration,
                        )
        torch.cuda.empty_cache()
