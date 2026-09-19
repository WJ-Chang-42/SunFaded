# Copyright (c) 2025
# Longsplat-style Camera (compatible with your pipeline)
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View, getProjectionMatrix, fov2focal, focal2fov

class Camera(nn.Module):
    """
    Camera adapter for the SunFaded COLMAP workflow:
    - world_view_transform uses the supplied COLMAP R_gt/T_gt
    - K from FoV; also exposes intrins_matrix = (K^{-1})^T for compatibility
    - supports low-res training -> high-res evaluation via to_final()
    - keeps light & gt_alpha_mask fields
    """
    def __init__(self,
                 colmap_id,
                 R, T,
                 FoVx, FoVy,
                 image,
                 light=None,
                 gt_alpha_mask=None,
                 image_name=None,
                 uid=None,
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0,
                 data_device="cuda",
                 R_gt=None, T_gt=None):

        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.image_name = image_name
        self.is_registered = False

        # device
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # Save GT / Pred for reference (longsplat pattern)
        self.R_gt = torch.tensor(R_gt) if R_gt is not None else None
        self.T_gt = torch.tensor(T_gt) if T_gt is not None else None
        self.R_pred = torch.tensor(R) if R is not None else None
        self.T_pred = torch.tensor(T) if T is not None else None

        # Runtime pose starts from provided R/T if any; otherwise identity
        if R is not None and T is not None:
            self.R = torch.as_tensor(R, dtype=torch.float32, device=self.data_device)
            self.T = torch.as_tensor(T, dtype=torch.float32, device=self.data_device)
        else:
            t = torch.eye(4, device=self.data_device)
            self.R = t[:3, :3].clone()
            self.T = t[:3, 3].clone()

        # Preserve the original width-based 1024 cap and interpolation rounding.
        with torch.no_grad():
            if image.shape[2] > 1024:
                resize_factor = 1024.0 / float(image.shape[2])
                image_resize = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    scale_factor=resize_factor,
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
            else:
                resize_factor = 1.0
                image_resize = image

        self.original_image = image_resize.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = int(self.original_image.shape[2])
        self.image_height = int(self.original_image.shape[1])

        self.original_image_final = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width_final = int(self.original_image_final.shape[2])
        self.image_height_final = int(self.original_image_final.shape[1])

        # Mask handling: multiply like longsplat, also keep the mask for reference
        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            self.original_image = self.original_image * self.gt_alpha_mask
        else:
            self.gt_alpha_mask = None
        # Optional light field (kept for compatibility with your pipeline)
        if light is not None:
            temp = light / (light.max() + 1e-8)


            self.light = torch.as_tensor(temp, dtype=torch.float32, device=self.data_device).unsqueeze(0)

            self.light  = torch.nn.functional.interpolate(
                    self.light.unsqueeze(0),
                    scale_factor=resize_factor,
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
        else:
            self.light = None

        self.zfar = 100.0
        self.znear = 0.01

        # keep trans/scale for data normalization outside camera matrices
        self.trans = np.array(trans, dtype=np.float32)
        self.scale = float(scale)

        # caches for external modules
        self.kp0 = None
        self.kp1 = None
        self.depth_map = None
        self.pre_depth_map = None
        self.pts3d = None

        # Intrinsics from FoV -> focal (pixels)
        if FoVx is None or FoVy is None:
            self.FoVx = None
            self.FoVy = None
            self.Focalx = None
            self.Focaly = None
            self.intrinsic = None
            self.intrins_matrix = None
            self.projection_matrix = None
        else:
            self.FoVx = float(FoVx)
            self.FoVy = float(FoVy)
            self.Focalx = float(fov2focal(self.FoVx, self.image_width))
            self.Focaly = float(fov2focal(self.FoVy, self.image_height))
            self.intrinsic = torch.tensor(
                [[self.Focalx, 0.0, self.image_width / 2.0],
                 [0.0, self.Focaly, self.image_height / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=self.data_device
            )
            # compatibility: (K^{-1})^T to match your previous "intrins_matrix"
            self.intrins_matrix = torch.inverse(self.intrinsic).T
            self.projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar,
                fovX=self.FoVx, fovY=self.FoVy
            ).transpose(0, 1).to(self.data_device)

        # learnable deltas (longsplat pattern)
        self.cam_rot_delta = nn.Parameter(torch.zeros(3, device=self.data_device))
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, device=self.data_device))

    # ---------- properties ----------
    @property
    def world_view_transform(self):
        # NOTE: if you implement SO(3) from cam_rot_delta, apply it here
        return getWorld2View(self.R_gt, self.T_gt).transpose(0, 1).to(self.data_device)

    @property
    def view_world_transform(self):
        return torch.inverse(self.world_view_transform)

    @property
    def full_proj_transform(self):
        if self.projection_matrix is None:
            return None
        return (self.world_view_transform.unsqueeze(0)
                .bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    @property
    def camera_center(self):
        return self.view_world_transform[3, :3]

    # ---------- runtime updates ----------
    def to_final(self):
        """Switch to full-res image & update intrinsics/depth size accordingly."""
        self.original_image = self.original_image_final
        self.image_width = self.image_width_final
        self.image_height = self.image_height_final

        if self.FoVx is not None and self.FoVy is not None:
            self.Focalx = float(fov2focal(self.FoVx, self.image_width))
            self.Focaly = float(fov2focal(self.FoVy, self.image_height))
            self.intrinsic = torch.tensor(
                [[self.Focalx, 0.0, self.image_width / 2.0],
                 [0.0, self.Focaly, self.image_height / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=self.data_device
            )
            self.intrins_matrix = torch.inverse(self.intrinsic).T
        if self.depth_map is not None:
            self.depth_map = torch.nn.functional.interpolate(
                self.depth_map.unsqueeze(0).unsqueeze(0),
                size=(self.image_height, self.image_width),
                mode='bilinear', align_corners=True
            ).squeeze(0).squeeze(0)

    def update_RT(self, R, t):
        self.R = torch.as_tensor(R, dtype=torch.float32, device=self.data_device)
        self.T = torch.as_tensor(t, dtype=torch.float32, device=self.data_device)

    def update_focal(self, focal_length):
        """Set square pixels fx=fy=f; keep FOV consistent."""
        self.FoVx = float(focal2fov(focal_length, self.image_width))
        self.FoVy = float(focal2fov(focal_length, self.image_height))
        self.Focalx = float(focal_length)
        self.Focaly = float(focal_length)
        self.intrinsic = torch.tensor(
            [[self.Focalx, 0.0, self.image_width / 2.0],
             [0.0, self.Focaly, self.image_height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32, device=self.data_device
        )
        self.intrins_matrix = torch.inverse(self.intrinsic).T
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar,
            fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to(self.data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVy = float(fovy)
        self.FoVx = float(fovx)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
