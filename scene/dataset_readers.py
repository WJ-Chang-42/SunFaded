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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
import glob
import torch

# Apply the same world-scale factor to input points and camera translations.
# A factor of 0.1 reduces world-space distances by ten.
WORLD_SCALE = 0.1

def estimate_illumination(img, sigma=250):
    """Estimate an illumination map using Gaussian blur."""
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image = np.float32(gray_image) + 1.0  # Keep intensities strictly positive.
    gray_image = gray_image/255.
    illumination = cv2.GaussianBlur(gray_image, (0, 0), sigma)
    return illumination


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    R_gt: np.array
    T_gt: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    light: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = torch.cat(cam_centers, dim=1)
        avg_cam_center = torch.mean(cam_centers, dim=1, keepdim=True)
        center = avg_cam_center
        dist = torch.norm(cam_centers - center, dim=0, keepdim=True)
        diagonal = torch.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View(cam.R, cam.T)
        C2W = torch.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    translate = translate.cpu().numpy()
    radius = radius.cpu().numpy()

    return {"translate": translate, "radius": radius}

## TODO: Add support for reading images from a json file ##
def readJsonCameras(cameras, cameras_gt, images_folder):
    cam_infos = []
    for idx, cam in enumerate(cameras):
        sys.stdout.write('\rReading camera {}/{}'.format(idx+1, len(cameras)))
        sys.stdout.flush()

        R_gt = None
        T_gt = None
        
        # Only try to get GT poses if cameras_gt is available
        if cameras_gt is not None:
            for key in cameras_gt:
                cam_gt = cameras_gt[key]
                if cam["image_name"] == cam_gt.name.split(".")[0]:
                    R_gt = np.transpose(qvec2rotmat(cam_gt.qvec))
                    T_gt = np.array(cam_gt.tvec) * WORLD_SCALE
                    break
        
        R = np.array(cam["R"])
        T = np.array(cam["T"]) * WORLD_SCALE
        FovY = focal2fov(cam["Focaly"], cam["height"])
        FovX = focal2fov(cam["Focalx"], cam["width"])

        pattern = os.path.join(images_folder, cam["image_name"] + ".*")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"Image file not found for {cam['image_name']}")
        image_path = files[0]

        image = Image.open(image_path)

        cam_info = CameraInfo(uid=idx, R=R, T=T, R_gt=R_gt, T_gt=T_gt,
                               FovY=FovY, FovX=FovX, image=image,
                               image_path=image_path, image_name=cam["image_name"],
                               width=image.size[0], height=image.size[1])
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readUnposedCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec) * WORLD_SCALE

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        light = estimate_illumination(np.array(image),100)
        cam_info = CameraInfo(uid=idx, R=None, T=None, R_gt=R, T_gt=T, FovY=FovY, FovX=FovX, image=image,light=light,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readUnposedCameras2(images_folder):
    cam_infos = []
    image_names = os.listdir(images_folder)
    image_names = sorted(image_names)
    for image_name in image_names:
        image_path = os.path.join(images_folder, image_name)
        image = Image.open(image_path)
        light = estimate_illumination(np.array(image),100)
        width, height = image.size
        image_name = image_name.split(".")[0]
        cam_info = CameraInfo(uid=len(cam_infos), R=None, T=None, R_gt=None, T_gt=None, FovY=None, FovX=None, image=image,light=light,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec) * WORLD_SCALE

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        light = estimate_illumination(np.array(image),100)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,light=light,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # Apply WORLD_SCALE to the PLY world-space coordinates.
    return BasicPointCloud(points=positions * WORLD_SCALE, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            # translation is in world units -- scale to configured WORLD_SCALE
            T = w2c[:3, 3] * WORLD_SCALE

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCustomSceneInfo(path, images):
    """Load every registered COLMAP view without a held-out split.

    Keep the original point-cloud scaling, camera order, illumination estimate,
    binary/text fallbacks, and cached PLY conversion.
    """
    sparse = os.path.join(path, "sparse", "0")
    try:
        extrinsics = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
        intrinsics = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    except (OSError, ValueError, EOFError, IndexError):
        try:
            extrinsics = read_extrinsics_text(os.path.join(sparse, "images.txt"))
            intrinsics = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
        except (OSError, ValueError, IndexError) as exc:
            raise ValueError("Valid COLMAP cameras and poses are required under sparse/0") from exc
    cameras = readUnposedCameras(extrinsics, intrinsics, os.path.join(path, images or "images"))
    cameras = sorted(cameras.copy(), key=lambda camera: camera.image_name)
    names = [camera.image_name for camera in cameras]
    if not names or len(set(names)) != len(names):
        raise ValueError("COLMAP image names must be non-empty and unique after basename/stem conversion")
    ply_path = os.path.join(sparse, "points3D.ply")
    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = read_points3D_binary(os.path.join(sparse, "points3D.bin"))
        except (OSError, ValueError, EOFError):
            xyz, rgb, _ = read_points3D_text(os.path.join(sparse, "points3D.txt"))
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)
    if pcd is None or len(pcd.points) == 0 or not np.isfinite(pcd.points).all():
        raise ValueError("COLMAP point cloud is empty or contains non-finite coordinates")
    return SceneInfo(point_cloud=pcd, train_cameras=cameras, test_cameras=[],
                     nerf_normalization=None, ply_path=ply_path)


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Custom" : readCustomSceneInfo,
}
