# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm

import torch

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def _safe_normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  """Normalize a vector with numeric stability."""
  norm = np.linalg.norm(v)
  if norm < eps:
    return np.zeros_like(v)
  return v / norm

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform
  # points = np.random.rand(3,100)
  # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
  # (poses_recentered @ points_h)[0]
  # (transform @ pad_poses(poses) @ points_h)[0,:3]

  # # Just make sure it's it in the [-1, 1]^3 cube
  # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  # poses_recentered[:, :3, 3] *= scale_factor
  # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  # return poses_recentered, transform


def _camera_box_and_orientation(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Compute a bounding box and stable orientation from camera poses."""
  positions = poses[:, :3, 3]
  bbox_min = positions.min(axis=0)
  bbox_max = positions.max(axis=0)
  center = (bbox_min + bbox_max) * 0.5

  forward = _safe_normalize(poses[:, :3, 2].mean(axis=0))
  if np.allclose(forward, 0):
    forward = np.array([0., 0., 1.], dtype=np.float64)

  up = _safe_normalize(poses[:, :3, 1].mean(axis=0))
  if np.allclose(up, 0):
    up = np.array([0., 1., 0.], dtype=np.float64)

  right = np.cross(forward, up)
  if np.linalg.norm(right) < 1e-6:
    right = np.array([1., 0., 0.], dtype=np.float64)
  right = _safe_normalize(right)
  up = _safe_normalize(np.cross(right, forward))

  offsets = positions - center
  right_extent = np.percentile(np.abs(offsets @ right), 90)
  up_extent = np.percentile(np.abs(offsets @ up), 90)
  forward_offset = np.percentile(offsets @ forward, 50)

  min_extent = np.max(np.linalg.norm(offsets, axis=1))
  scale_fallback = 1.0 if min_extent < 1e-6 else min_extent * 0.25
  right_extent = right_extent if right_extent > 1e-3 else scale_fallback
  up_extent = up_extent if up_extent > 1e-3 else scale_fallback

  return center, np.array([right_extent, up_extent, forward_offset]), forward, up, right


def _generate_orbit_from_box(poses: np.ndarray, n_frames: int) -> np.ndarray:
  """Generate a smooth orbit using a bounding box derived from poses."""
  center, extent, forward, up, right = _camera_box_and_orientation(poses)
  radius_right, radius_up, forward_offset = extent

  theta = np.linspace(0, 2. * np.pi, n_frames, endpoint=False)
  positions = []
  for t in theta:
    pos = (center + right * (radius_right * np.cos(t))
                  + up * (radius_up * np.sin(t))
                  + forward * forward_offset)
    positions.append(pos)

  return np.stack([viewmatrix(center - p, up, p) for p in positions])

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
        z_variation * (z_low[2] + (z_high - z_low)[2] *
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  #if const_speed:

  # # Resample theta angles so that the velocity is closer to constant.
  # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
  # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
  # positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  return np.stack([viewmatrix(p - center, up, p) for p in positions])


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
  """Convert a rotation matrix to a quaternion in (w, x, y, z) format."""
  m = matrix
  trace = np.trace(m)
  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (m[2, 1] - m[1, 2]) * s
    y = (m[0, 2] - m[2, 0]) * s
    z = (m[1, 0] - m[0, 1]) * s
  else:
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
      s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
      w = (m[2, 1] - m[1, 2]) / s
      x = 0.25 * s
      y = (m[0, 1] + m[1, 0]) / s
      z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
      s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
      w = (m[0, 2] - m[2, 0]) / s
      x = (m[0, 1] + m[1, 0]) / s
      y = 0.25 * s
      z = (m[1, 2] + m[2, 1]) / s
    else:
      s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
      w = (m[1, 0] - m[0, 1]) / s
      x = (m[0, 2] + m[2, 0]) / s
      y = (m[1, 2] + m[2, 1]) / s
      z = 0.25 * s
  quat = np.array([w, x, y, z], dtype=np.float64)
  return quat / np.linalg.norm(quat)


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
  """Convert a (w, x, y, z) quaternion to a rotation matrix."""
  w, x, y, z = q
  return np.array([
      [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
  ], dtype=np.float64)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
  q0 = q0 / np.linalg.norm(q0)
  q1 = q1 / np.linalg.norm(q1)
  dot = np.dot(q0, q1)
  if dot < 0.0:
    q1 = -q1
    dot = -dot
  if dot > 0.9995:
    result = q0 + t * (q1 - q0)
    return result / np.linalg.norm(result)
  theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
  theta = theta_0 * t
  q2 = q1 - q0 * dot
  q2 /= np.linalg.norm(q2)
  return q0 * np.cos(theta) + q2 * np.sin(theta)


def _catmull_rom_spline(points: np.ndarray, n_frames: int) -> np.ndarray:
  points = np.asarray(points)
  if len(points) < 2:
    return np.repeat(points, n_frames, axis=0)
  extended = np.concatenate([points[-1][None], points, points[:2]], axis=0)
  ts = np.linspace(0, len(points), n_frames, endpoint=False)
  result = []
  for t in ts:
    i = int(np.floor(t))
    local_t = t - i
    p0, p1, p2, p3 = extended[i], extended[i + 1], extended[i + 2], extended[i + 3]
    res = 0.5 * ((2 * p1) + (-p0 + p2) * local_t +
                 (2 * p0 - 5 * p1 + 4 * p2 - p3) * (local_t ** 2) +
                 (-p0 + 3 * p1 - 3 * p2 + p3) * (local_t ** 3))
    result.append(res)
  return np.stack(result)


def _interpolate_poses(pose_recenter: np.ndarray,
                       n_frames: int,
                       sort_by_azimuth: bool = False) -> np.ndarray:
  positions = pose_recenter[:, :3, 3]
  rotations = pose_recenter[:, :3, :3]

  if sort_by_azimuth:
    azimuth = np.arctan2(positions[:, 1], positions[:, 0])
    order = np.argsort(azimuth)
    positions = positions[order]
    rotations = rotations[order]

  interp_positions = _catmull_rom_spline(positions, n_frames)

  quats = [_rotation_matrix_to_quaternion(r) for r in rotations]
  ts = np.linspace(0, len(rotations), n_frames, endpoint=False)
  interp_rotations = []
  for t in ts:
    i = int(np.floor(t))
    local_t = t - i
    q0 = quats[i % len(quats)]
    q1 = quats[(i + 1) % len(quats)]
    interp_rotations.append(_quaternion_to_rotation_matrix(_slerp(q0, q1, local_t)))

  return np.stack([
      np.concatenate([r, p[:, None]], axis=1) for r, p in zip(interp_rotations, interp_positions)
  ])


def generate_path(viewpoint_cameras, n_frames=480, mode: str = "ellipse", sort_by_azimuth: bool = False):
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

  if mode == "ellipse":
    new_poses = _generate_orbit_from_box(poses=pose_recenter, n_frames=n_frames)
  elif mode == "interpolated":
    new_poses = _interpolate_poses(pose_recenter, n_frames=n_frames, sort_by_azimuth=sort_by_azimuth)
  else:
    raise ValueError(f"Unknown render path mode: {mode}")

  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

  traj = []
  for c2w in new_poses:
      c2w = c2w @ np.diag([1, -1, -1, 1])
      cam = copy.deepcopy(viewpoint_cameras[0])
      cam.image_height = int(cam.image_height / 2) * 2
      cam.image_width = int(cam.image_width / 2) * 2
      cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
      cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
      cam.camera_center = cam.world_view_transform.inverse()[3, :3]
      traj.append(cam)

  return traj

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def create_videos(base_dir, input_dir, out_name, num_frames=480):
  """Creates videos out of the images saved to disk."""
  # Last two parts of checkpoint path are experiment name and scene name.
  video_prefix = f'{out_name}'

  zpad = max(5, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  os.makedirs(base_dir, exist_ok=True)
  render_dist_curve_fn = np.log
  
  # Load one example frame to get image shape and depth range.
  depth_file = os.path.join(input_dir, 'vis', f'depth_{idx_to_str(0)}.tiff')
  depth_frame = load_img(depth_file)
  shape = depth_frame.shape
  p = 3
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
  lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
  print(f'Video shape is {shape[:2]}')

  video_kwargs = {
      'shape': shape[:2],
      'codec': 'h264',
      'fps': 60,
      'crf': 18,
  }
  
  for k in ['depth', 'normal', 'color']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
    input_format = 'gray' if k == 'alpha' else 'rgb'
    

    file_ext = 'png' if k in ['color', 'normal'] else 'tiff'
    idx = 0

    if k == 'color':
      file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')
    else:
      file0 = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

    if not os.path.exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      for idx in tqdm(range(num_frames)):
        # img_file = os.path.join(input_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
        if k == 'color':
          img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
        else:
          img_file = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

        if not os.path.exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = load_img(img_file)
        if k in ['color', 'normal']:
          img = img / 255.
        elif k.startswith('depth'):
          img = render_dist_curve_fn(img)
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
          img = cm.get_cmap('turbo')(img)[..., :3]

        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(frame)
        idx += 1

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')
