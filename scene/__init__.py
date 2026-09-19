# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact george.drettakis@inria.fr

"""COLMAP-only scene loading and complete SunFaded inference persistence."""

import json
import random
from pathlib import Path

import torch
from arguments import ModelParams
from scene.dataset_readers import readCustomSceneInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration
from sunfaded.priors import attach_depth_priors
from sunfaded.state import (
    camera_manifest,
    read_inference_state,
    restore_inference_state,
    save_inference_state,
)


class Scene:
    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=(1.0,),
        inference_state=None,
    ):
        self.args = args
        self.model_path = args.model_path
        self.gaussians = gaussians
        self.loaded_iter = None
        self.render_settings = None
        if inference_state is not None:
            state = read_inference_state(inference_state)
            self.loaded_iter = state["iteration"]
        elif load_iteration is not None:
            self.loaded_iter = (
                searchForMaxIteration(str(Path(self.model_path) / "point_cloud"))
                if load_iteration == -1
                else load_iteration
            )
            path = (
                Path(self.model_path)
                / "point_cloud"
                / f"iteration_{self.loaded_iter}"
                / "inference_state.pth"
            )
            state = read_inference_state(path)
        else:
            state = None

        scene_info = readCustomSceneInfo(args.source_path, args.images)
        if shuffle:
            random.shuffle(scene_info.train_cameras)
        self.train_cameras = {}
        # Empty compatibility container for inherited callers; never populated.
        self.test_cameras = {scale: [] for scale in resolution_scales}
        for scale in resolution_scales:
            self.train_cameras[scale] = cameraList_from_camInfos(
                scene_info.train_cameras, scale, args
            )
        if not self.getTrainCameras():
            raise ValueError("No registered COLMAP images available for training")
        self.cameras_extent = 10.0
        if state is not None:
            restore_inference_state(self.gaussians, state)
            self.render_settings = state["render_settings"]
        else:
            attach_depth_priors(self.getAllCameras(), args.depth_pro_checkpoint)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            manifest = dict(
                protocol="train_all",
                held_out_views=0,
                seed=args.seed,
                optimized_names=[
                    camera.image_name for camera in self.getTrainCameras()
                ],
                cameras=camera_manifest(self.getTrainCameras()),
            )
            with open(Path(self.model_path) / "camera_manifest.json", "w") as handle:
                json.dump(manifest, handle, indent=2)

    def save(self, iteration, pipe=None, texture_rendering=True):
        folder = Path(self.model_path) / "point_cloud" / f"iteration_{iteration}"
        folder.mkdir(parents=True, exist_ok=True)
        self.gaussians.save_ply(str(folder / "point_cloud.ply"))
        torch.save(self.gaussians.mlp.state_dict(), folder / "mlp.pth")
        torch.save(self.gaussians.get_env, folder / "env.pth")
        save_inference_state(
            self.gaussians,
            folder / "inference_state.pth",
            iteration,
            scene=self,
            pipe=pipe,
            texture_rendering=texture_rendering,
        )

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getAllCameras(self, scale=1.0):
        return self.getTrainCameras(scale) + self.getTestCameras(scale)
