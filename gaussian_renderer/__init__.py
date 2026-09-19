#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from sunfaded.shading import embedding_func, phong_lighting_image
from sunfaded.shading import (
    phong_lighting as phong_lighting,
    rotmat2qvec as rotmat2qvec,
)

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
import torch.nn.functional as F


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    texture_rendering=False,
    grad_update=False,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        splat2world = pc.get_covariance(scaling_modifier)
        (W, H) = (viewpoint_camera.image_width, viewpoint_camera.image_height)
        (near, far) = (viewpoint_camera.znear, viewpoint_camera.zfar)
        ndc2pix = (
            torch.tensor(
                [
                    [W / 2, 0, 0, (W - 1) / 2],
                    [0, H / 2, 0, (H - 1) / 2],
                    [0, 0, far - near, near],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .cuda()
            .T
        )
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]])
            .permute(0, 2, 1)
            .reshape(-1, 9)
        )
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    pipe.convert_SHs_python = True
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            (albedo, specular, shininess) = (
                pc.get_albedo[:, :3].clamp(min=0, max=1),
                pc.get_albedo[:, 3:6].clamp(min=1e-06),
                pc.get_albedo[:, 6:].clamp(min=1e-06),
            )
            colors_precomp = albedo
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    if not grad_update:
        with torch.no_grad():
            (rendered_image, radii, allmap) = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
    else:
        (rendered_image, radii, allmap) = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    rets = {"viewspace_points": means2D, "visibility_filter": radii > 0, "radii": radii}
    render_alpha = allmap[1:2]
    render_normal_origin = allmap[2:5]
    render_normal = render_normal_origin
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    render_depth_expected = allmap[0:1]
    render_depth_expected = render_depth_expected / (render_alpha + 1e-08)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    render_dist = allmap[6:7]
    surf_depth = (
        render_depth_expected * (1 - pipe.depth_ratio)
        + pipe.depth_ratio * render_depth_median
    )
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    surf_normal = surf_normal * render_alpha
    env_light = pc.get_env
    rendered_albedo = rendered_image[:3]
    if texture_rendering:
        with torch.no_grad():
            (W, H) = (viewpoint_camera.image_width, viewpoint_camera.image_height)
            (grid_x, grid_y) = torch.meshgrid(
                torch.arange(W, device="cuda").float(),
                torch.arange(H, device="cuda").float(),
                indexing="xy",
            )
            points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
            rays_d = (points @ viewpoint_camera.intrins_matrix).permute(2, 0, 1)
            camera_direction_normalization = rays_d / rays_d.norm(dim=0, keepdim=True)
            points_cloud = rays_d * surf_depth
            (block_W, block_H) = (W // 16, H // 16)
            points_cloud_block = F.interpolate(
                points_cloud.unsqueeze(0),
                size=(block_H, block_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        light_direction_block = points_cloud_block - pc._offset[:3]
        light_distance_block = light_direction_block.norm(dim=0, keepdim=True).reshape(
            -1, 1
        )
        light_direction_block = light_direction_block / (
            light_direction_block.norm(dim=0, keepdim=True) + 1e-05
        )
        light_direction_block = light_direction_block.permute(1, 2, 0).reshape(-1, 3)
        (x, y, z) = (
            light_direction_block[:, 0:1],
            light_direction_block[:, 1:2],
            light_direction_block[:, 2:3],
        )
        theta = torch.asin(y)
        phi = torch.atan2(x, z + 1e-05)
        light_direction = points_cloud - pc._offset[:3]
        light_direction_normalization = light_direction / (
            light_direction.norm(dim=0, keepdim=True) + 1e-05
        )
        pose = torch.cat([theta, phi], dim=-1)
        camera_pose = light_distance_block
        embedding = embedding_func(pose)
        embedding_camera = embedding_func(camera_pose)
        (light_block, _) = pc.mlp(embedding, embedding_camera)
        light_block = light_block.reshape(1, block_H, block_W, 3)
        light = F.interpolate(
            light_block.permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        shadow = torch.zeros(
            (1, *rendered_albedo.shape[1:]), device=rendered_albedo.device
        )
        rendered_light = phong_lighting_image(
            rendered_albedo,
            None,
            None,
            render_normal_origin,
            -light_direction_normalization,
            -camera_direction_normalization,
            light,
        )
    else:
        rendered_light = torch.zeros(
            rendered_albedo.shape, device=rendered_albedo.device
        )
        light = torch.zeros(rendered_albedo.shape, device=rendered_albedo.device)
        light_block = torch.zeros(rendered_albedo.shape, device=rendered_albedo.device)
        shadow = torch.zeros(
            (1, *rendered_albedo.shape[1:]), device=rendered_albedo.device
        )
    if torch.isnan(rendered_light.sum()) > 0:
        raise FloatingPointError("Non-finite lighting encountered while rendering")
    rendered_albedo_env = rendered_albedo * env_light
    all_rendered = rendered_albedo_env + rendered_light
    rendered_albedo_light = rendered_light
    rets.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
            "albedo": rendered_albedo,
            "albedo_env": rendered_albedo_env,
            "albedo_light": rendered_albedo_light,
            "light": light,
            "light_block": light_block,
            "render": all_rendered,
            "shadow": shadow,
        }
    )
    return rets
