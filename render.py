# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact george.drettakis@inria.fr

"""Reload a complete SunFaded state and export reconstruction metrics/images."""

import argparse
from functools import partial
from pathlib import Path

from arguments import ModelParams, PipelineParams, get_combined_args


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--inference_state",
        help="Path to a complete SunFaded inference snapshot",
    )
    parser.add_argument("--output", help="New reconstruction output directory")
    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Save float RGB, target and depth NPZ files",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip reconstruction export (mesh/trajectory only)",
    )
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--render_path",
        action="store_true",
        help="Inherited experimental camera trajectory",
    )
    parser.add_argument(
        "--render_mode", default="ellipse", choices=["ellipse", "interpolated"]
    )
    parser.add_argument("--voxel_size", default=-1.0, type=float)
    parser.add_argument("--depth_trunc", default=-1.0, type=float)
    parser.add_argument("--sdf_trunc", default=-1.0, type=float)
    parser.add_argument("--num_cluster", default=50, type=int)
    parser.add_argument("--unbounded", action="store_true")
    parser.add_argument("--mesh_res", default=1024, type=int)
    return parser, model, pipeline


def main():
    parser, model, pipeline = build_parser()
    args = get_combined_args(parser)
    if not args.source_path:
        parser.error("-s/--source_path is required")
    if not args.model_path and not args.inference_state:
        parser.error("Supply -m/--model_path or --inference_state")
    import torch
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from sunfaded.state import read_inference_state
    from sunfaded.evaluation import export_reconstruction
    from utils.system_utils import searchForMaxIteration

    state_path = args.inference_state
    if state_path is None:
        iteration = (
            searchForMaxIteration(str(Path(args.model_path) / "point_cloud"))
            if args.iteration == -1
            else args.iteration
        )
        state_path = (
            Path(args.model_path)
            / "point_cloud"
            / f"iteration_{iteration}"
            / "inference_state.pth"
        )
    state = read_inference_state(state_path)
    # State settings are authoritative: resizing or a different lighting mode changes the result.
    for key, value in state.get("data_settings", {}).items():
        if key in ("resolution", "white_background"):
            setattr(args, key, value)
    args.sh_degree = state["max_sh_degree"]
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False, inference_state=state_path)
    pipe = pipeline.extract(args)
    settings = scene.render_settings
    for key in ("depth_ratio", "compute_cov3D_python", "convert_SHs_python"):
        setattr(pipe, key, settings[key])
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    output = (
        Path(args.output)
        if args.output
        else Path(args.model_path or Path(state_path).parent)
        / "rendered"
        / str(scene.loaded_iter)
    )
    if not args.skip_train:
        export_reconstruction(
            scene.getTrainCameras(),
            gaussians,
            pipe,
            background,
            output,
            texture_rendering=settings["texture_rendering"],
            save_raw=args.save_raw,
        )
    if args.skip_mesh and not args.render_path:
        return
    from utils.mesh_utils import GaussianExtractor, post_process_mesh
    from utils.render_utils import generate_path, create_videos
    import open3d as o3d

    render_fn = partial(render, texture_rendering=settings["texture_rendering"])
    extractor = GaussianExtractor(gaussians, render_fn, pipe, bg_color=bg_color)
    if args.render_path:
        trajectory_dir = output / "trajectory"
        if trajectory_dir.exists():
            raise FileExistsError(trajectory_dir)
        trajectory_dir.mkdir(parents=True)
        trajectory = generate_path(
            scene.getTrainCameras(), n_frames=240, mode=args.render_mode
        )
        extractor.reconstruction(trajectory)
        extractor.export_image(str(trajectory_dir))
        create_videos(
            str(trajectory_dir), str(trajectory_dir), "render_traj", num_frames=240
        )
    if not args.skip_mesh:
        mesh_dir = output / "mesh"
        if mesh_dir.exists():
            raise FileExistsError(mesh_dir)
        mesh_dir.mkdir(parents=True)
        extractor.gaussians.active_sh_degree = 0
        extractor.reconstruction(scene.getTrainCameras())
        if args.unbounded:
            name = "fuse_unbounded.ply"
            mesh = extractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = "fuse.ply"
            depth_trunc = (
                extractor.radius * 2.0 if args.depth_trunc < 0 else args.depth_trunc
            )
            voxel_size = (
                depth_trunc / args.mesh_res if args.voxel_size < 0 else args.voxel_size
            )
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = extractor.extract_mesh_bounded(
                voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc
            )
        o3d.io.write_triangle_mesh(str(mesh_dir / name), mesh)
        post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(
            str(mesh_dir / name.replace(".ply", "_post.ply")), post
        )


if __name__ == "__main__":
    main()
