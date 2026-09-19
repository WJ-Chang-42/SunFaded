"""
SunFaded optimization over all registered COLMAP views.

Configuration, reporting, depth priors, lighting components and inference
persistence live in separate modules. See README.md for setup and usage.

Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.

This software is free for non-commercial, research and evaluation use
under the terms of the LICENSE.md file.

For inquiries contact george.drettakis@inria.fr
"""

import os
import json
from random import randint
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import imageio
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import render_net_image
from sunfaded.priors import create_depth_model
from sunfaded.state import save_inference_state
from utils.point_utils import depth_to_normal


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    config_path=None,
):
    """Optimize the scene and lighting over all registered COLMAP views."""
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    training_config = load_training_config(config_path) if config_path else None
    if training_config:
        config_save_path = os.path.join(dataset.model_path, "training_config.json")
        with open(config_save_path, "w", encoding="utf-8") as handle:
            json.dump(training_config, handle, indent=2)
        print(f"Training configuration saved to: {config_save_path}")
    else:
        print("Using the default training configuration")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        raise ValueError(
            "Training resume is not supported: legacy checkpoints omit learned state. Use a complete inference snapshot with render.py."
        )
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    gaussian_num = False
    texture_rendering = False
    uncertainty = False
    gaussian_update = True
    env_list = {}
    light_list = {}
    normal_list = {}
    uncertainty_list = {}
    after_iters = 0
    frozen_lr_cache = None
    flag = None
    for iteration in range(first_iter, opt.iterations + 1):
        iter_config = get_iteration_config(training_config, iteration)
        gaussian_num = iter_config["gaussian_num"]
        texture_rendering = iter_config["texture_rendering"]
        uncertainty = iter_config["uncertainty"]
        gaussian_update = iter_config["gaussian_update"]
        frozen_lr_cache = apply_legacy_freeze(
            gaussians.optimizer, iteration, frozen_lr_cache
        )
        iter_start.record()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        with torch.autograd.set_detect_anomaly(False):
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                texture_rendering=texture_rendering,
                grad_update=gaussian_update,
            )
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            albedo = render_pkg["albedo"]
            albedo_light = render_pkg["albedo_light"]
            albedo_env = render_pkg["albedo_env"]
            light = render_pkg["light"]
            shadow = render_pkg["shadow"].clamp(min=0.1)
            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth = viewpoint_cam.depth_map
            coefficient = -10
            image_weight = torch.exp(coefficient * viewpoint_cam.light)
            if texture_rendering:
                if iteration > 15000:
                    L_uncertainty = torch.tensor([0]).to(image.device)
                    if viewpoint_cam.uid in uncertainty_list:
                        uncer_ = uncertainty_list[viewpoint_cam.uid]
                    M = torch.ones_like(shadow)
                else:
                    L_uncertainty = torch.tensor([0]).to(image.device)
                    M = torch.ones_like(shadow)
                if env_list:
                    depth_loss = 1 - pearson_corrcoef_min(
                        render_pkg["surf_depth"].reshape(-1, 1),
                        env_list[viewpoint_cam.uid].reshape(-1, 1),
                    )
                else:
                    depth_loss = render_pkg["surf_depth"].new_tensor(0.0)
                Ll1 = l1_loss(image * M, gt_image * M)
                ssim_loss = 1.0 - ssim(image * M, gt_image * M)
                loss = (
                    (1.0 - opt.lambda_dssim) * Ll1
                    + opt.lambda_dssim * ssim_loss
                    + L_uncertainty
                    + 0.001 * depth_loss
                )
            else:
                Ll1 = l1_loss(albedo_env * image_weight, gt_image * image_weight)
                M = torch.ones_like(shadow)
                depth_loss = 1 - pearson_corrcoef_min(
                    render_pkg["surf_depth"].reshape(-1, 1), gt_depth.reshape(-1, 1)
                )
                loss = 5 * Ll1 + 0.001 * depth_loss
                L_uncertainty = torch.tensor([0]).to(image.device)
            lambda_normal_start = (
                training_config["regularization"]["lambda_normal_start_iteration"]
                if training_config
                else 7000
            )
            lambda_dist_start = (
                training_config["regularization"]["lambda_dist_start_iteration"]
                if training_config
                else 3000
            )
            lambda_normal = (
                opt.lambda_normal if iteration > lambda_normal_start else 0.0
            )
            lambda_dist = opt.lambda_dist if iteration > lambda_dist_start else 0.0
            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            if env_list:
                viewpoint_normal = normal_list[viewpoint_cam.uid] / (
                    render_pkg["rend_alpha"] + 1e-08
                )
            else:
                viewpoint_normal = viewpoint_cam.normal_map / (
                    render_pkg["rend_alpha"] + 1e-08
                )
            normal_error = (1 - (viewpoint_normal * surf_normal).sum(dim=0))[None][
                ..., 1:-1, 1:-1
            ]
            normal_loss = (
                0.5
                * lambda_normal
                * (1 - (rend_normal * surf_normal).sum(dim=0))[None].mean()
                + 0.5 * lambda_normal * normal_error.mean()
            )
            dist_loss = lambda_dist * rend_dist.mean()
            depth_loss = render_pkg["surf_depth"].new_tensor(0.0)
            total_loss = loss + dist_loss + normal_loss + depth_loss
            total_loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "after_iters": f"{after_iters}",
                    "shadow": f"{shadow.mean().item():.5f}",
                    "depth": f"{ema_depth_for_log:.5f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "Offset": f"{gaussians._offset[0, 0, 0]:.3f},{gaussians._offset[1, 0, 0]:.3f},{gaussians._offset[2, 0, 0]:.3f},{gaussians._offset[3, 0, 0]:.3f},{gaussians._offset[4, 0, 0]:.3f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train_loss_patches/dist_loss", ema_dist_for_log, iteration
                )
                tb_writer.add_scalar(
                    "train_loss_patches/normal_loss", ema_normal_for_log, iteration
                )
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, texture_rendering),
                light_list,
                normal_list,
            )
            if iteration in saving_iterations:
                print(f"\n[Iteration {iteration}] Saving Gaussian model...")
                scene.save(iteration, pipe=pipe, texture_rendering=texture_rendering)
            save_iterations = (
                training_config["special_operations"]["image_saving"]["iterations"]
                if training_config
                else [5000, 7000, 10000, 15000, 20000, 25000, 30000]
            )
            image_save_enabled = (
                training_config["special_operations"]["image_saving"]["enabled"]
                if training_config
                else True
            )
            if (
                iteration % 5000 == 0 or iteration in save_iterations
            ) and image_save_enabled:
                name = iteration
                image_dir = os.path.join(scene.model_path, "image")
                os.makedirs(image_dir, exist_ok=True)
                print(f"\n[Iteration {iteration}] Saving visualization images...")
                normal = render_pkg["rend_normal"]
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_normal_learn.png"),
                    ((0.5 + 0.5 * normal) * 255)
                    .type(torch.uint8)
                    .cpu()
                    .permute(1, 2, 0),
                )
                normal = render_pkg["surf_normal"]
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_normal_surf.png"),
                    ((0.5 + 0.5 * normal) * 255)
                    .type(torch.uint8)
                    .cpu()
                    .permute(1, 2, 0),
                )
                normal = viewpoint_cam.normal_map
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_normal_predict.png"),
                    ((0.5 + 0.5 * normal) * 255)
                    .type(torch.uint8)
                    .cpu()
                    .permute(1, 2, 0),
                )
                plt.figure(figsize=(10, 8))
                plt.imshow(
                    render_pkg["surf_depth"].squeeze().detach().cpu(),
                    vmax=torch.quantile(render_pkg["surf_depth"], q=0.95).item(),
                )
                plt.colorbar()
                plt.title(f"Rendered Depth - Iteration {name}")
                plt.savefig(os.path.join(image_dir, f"{name}_depth.png"))
                plt.close("all")
                light_iter = (
                    training_config["special_operations"]["light_computation"][
                        "iteration"
                    ]
                    if training_config
                    else 7000
                )
                light_enabled = (
                    training_config["special_operations"]["light_computation"][
                        "enabled"
                    ]
                    if training_config
                    else True
                )
                if iteration == 30000 and True:
                    (model, transform) = create_depth_model(
                        dataset.depth_pro_checkpoint
                    )
                    for idx, viewpoint in enumerate(scene.getTrainCameras().copy()):
                        render_pkg_light = render(
                            viewpoint,
                            gaussians,
                            pipe,
                            background,
                            texture_rendering=True,
                        )
                        img = render_pkg_light["albedo"].detach().cpu()
                        img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
                        img = img.permute(1, 2, 0).contiguous()
                        pil_img = Image.fromarray(img.numpy(), mode="RGB")
                        image = transform(pil_img)
                        prediction = model.infer(image.cuda(), f_px=None)
                        depth = prediction["depth"]
                        light_list[viewpoint.uid] = depth.cuda()
                        normal_list[viewpoint.uid] = depth_to_normal(
                            viewpoint, light_list[viewpoint.uid]
                        ).permute(2, 0, 1)
                    del model, transform
                if env_list and viewpoint_cam.uid in env_list:
                    imageio.imwrite(
                        os.path.join(image_dir, f"{name}_env.png"),
                        (env_list[viewpoint_cam.uid].detach().cpu() * 255)
                        .type(torch.uint8)
                        .permute(1, 2, 0),
                    )
                    if viewpoint_cam.uid in uncertainty_list:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(
                            uncertainty_list[viewpoint_cam.uid].squeeze().detach().cpu()
                        )
                        plt.colorbar()
                        plt.title(f"Uncertainty Prediction - Iteration {name}")
                        plt.savefig(os.path.join(image_dir, f"{name}_uncertainty.png"))
                        plt.close("all")
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_rgb.png"),
                    (image.detach().cpu() * 255).type(torch.uint8).permute(1, 2, 0),
                )
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_rgb_origin.png"),
                    (gt_image.detach().cpu() * 255).type(torch.uint8).permute(1, 2, 0),
                )
                plt.imshow(viewpoint_cam.depth_map.detach().cpu())
                plt.colorbar()
                plt.savefig(os.path.join(image_dir, f"{name}_depth_estimation.png"))
                plt.close("all")
                if light_list:
                    plt.imshow(light_list[viewpoint_cam.uid].detach().cpu())
                    plt.colorbar()
                    plt.savefig(os.path.join(image_dir, f"{name}_depth_light_list.png"))
                    plt.close("all")
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_albedo.png"),
                    (albedo.detach().cpu() * 255).type(torch.uint8).permute(1, 2, 0),
                )
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_albedo_env.png"),
                    (albedo_env.detach().cpu() * 255)
                    .type(torch.uint8)
                    .permute(1, 2, 0),
                )
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_albedo_light.png"),
                    (albedo_light.detach().cpu() * 255)
                    .type(torch.uint8)
                    .permute(1, 2, 0),
                )
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_light.png"),
                    (light.detach().cpu() * 255).type(torch.uint8).permute(1, 2, 0),
                )
                plt.figure(figsize=(10, 8))
                plt.imshow(image_weight.detach().squeeze().cpu())
                plt.colorbar()
                plt.title(f"Lighting Weights - Iteration {name}")
                plt.savefig(os.path.join(image_dir, f"{name}_shadow.png"))
                plt.close("all")
                imageio.imwrite(
                    os.path.join(image_dir, f"{name}_shadow*light.png"),
                    ((shadow * light).detach().cpu() * 255)
                    .type(torch.uint8)
                    .permute(1, 2, 0),
                )
                np.save(
                    os.path.join(image_dir, f"{name}_light.npy"),
                    light.detach().cpu().permute(1, 2, 0).numpy(),
                )
            if iteration < opt.densify_until_iter and gaussian_update:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = 20
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )
                if iteration > 0:
                    should_reset = iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    )
                    if should_reset:
                        gaussians.reset_opacity()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                after_iters += 1
            if iteration in checkpoint_iterations:
                print(
                    f"\n[Iteration {iteration}] Saving complete inference state (not training resume)."
                )
                save_inference_state(
                    gaussians,
                    os.path.join(scene.model_path, f"inference_{iteration}.pth"),
                    iteration,
                    scene=scene,
                    pipe=pipe,
                    texture_rendering=texture_rendering,
                )
        with torch.no_grad():
            if getattr(dataset, "viewer", False) and network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    (
                        custom_cam,
                        do_training,
                        keep_alive,
                        scaling_modifer,
                        render_mode,
                    ) = network_gui.receive()
                    if custom_cam is not None:
                        render_pkg = render(
                            custom_cam,
                            gaussians,
                            pipe,
                            background,
                            texture_rendering=texture_rendering,
                            scaling_modifier=scaling_modifer,
                        )
                        net_image = render_net_image(
                            render_pkg, dataset.render_items, render_mode, custom_cam
                        )
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log,
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and (
                        iteration < int(opt.iterations) or not keep_alive
                    ):
                        break
                except Exception as e:
                    network_gui.conn = None
    return scene


from sunfaded.config import (
    load_training_config,
    get_iteration_config,
    apply_legacy_freeze,
)
from sunfaded.reporting import prepare_output_and_logger, training_report
from sunfaded.losses import pearson_corrcoef_min
