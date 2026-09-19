# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact george.drettakis@inria.fr

"""Train SunFaded on every registered COLMAP view."""

import argparse
import json
from pathlib import Path

from arguments import ModelParams, OptimizationParams, PipelineParams


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument(
        "--config",
        default=None,
        help="Training JSON; recommended: config/training_config.json",
    )
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[5000, 7000, 10000, 20000, 30000, 40000],
        help="Training reconstruction reporting iterations; never held-out testing",
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[],
        help="Save additional complete inference snapshots, not resumable training",
    )
    parser.add_argument(
        "--start_checkpoint",
        default=None,
        help="Unsupported legacy option; fails explicitly",
    )
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable the inherited experimental SIBR viewer",
    )
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", default=6009, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip final all-view image/metric export (e.g. smoke tests)",
    )
    return parser, model, optimization, pipeline


def main(argv=None):
    parser, model, optimization, pipeline = build_parser()
    args = parser.parse_args(argv)
    if not args.source_path:
        parser.error("-s/--source_path is required")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.start_checkpoint:
        parser.error(
            "Training resume is not supported; render complete inference snapshots with render.py"
        )
    if not Path(args.depth_pro_checkpoint).is_file():
        parser.error("Depth Pro weights are missing; supply --depth_pro_checkpoint")
    if args.config and not Path(args.config).is_file():
        parser.error("--config file does not exist")
    import random
    import numpy as np
    import torch
    from sunfaded import training as trainer
    from sunfaded.evaluation import export_reconstruction

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and the two compiled CUDA extensions are required")
    dataset = model.extract(args)
    dataset.viewer = args.viewer
    if args.viewer:
        trainer.network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    pipe = pipeline.extract(args)
    scene = trainer.training(
        dataset,
        optimization.extract(args),
        pipe,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        None,
        args.config,
    )
    # Capture after the final iteration's bookkeeping, including densification if requested.
    texture = trainer.get_iteration_config(
        trainer.load_training_config(args.config) if args.config else None,
        args.iterations,
    )["texture_rendering"]
    scene.save(args.iterations, pipe=pipe, texture_rendering=texture)
    with open(Path(scene.model_path) / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)
    if not args.skip_export:
        background = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        export_reconstruction(
            scene.getTrainCameras(),
            scene.gaussians,
            pipe,
            background,
            Path(scene.model_path) / "reconstruction",
            texture_rendering=texture,
        )
    print(
        f"Training complete: {scene.model_path} (all-view reconstruction, no held-out test)"
    )


if __name__ == "__main__":
    main()
