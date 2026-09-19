"""Versioned complete inference snapshots. These are not training-resume checkpoints."""

from pathlib import Path
import os

import torch
from torch import nn

STATE_FIELDS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_albedo",
    "_scaling",
    "_rotation",
    "_opacity",
    "_env",
    "_offset",
    "max_radii2D",
    "xyz_gradient_accum",
    "denom",
)
PARAMETERS = set(STATE_FIELDS[:9])


def camera_manifest(cameras):
    return [
        dict(name=c.image_name, uid=c.uid, width=c.image_width, height=c.image_height)
        for c in cameras
    ]


def capture_inference_state(
    gaussians, iteration, scene=None, pipe=None, texture_rendering=True
):
    values = {
        key: getattr(gaussians, key).detach().cpu().clone() for key in STATE_FIELDS
    }
    values["mlp"] = {
        key: value.detach().cpu().clone()
        for key, value in gaussians.mlp.state_dict().items()
    }
    result = dict(
        format_version=1,
        iteration=int(iteration),
        parameters=values,
        active_sh_degree=gaussians.active_sh_degree,
        max_sh_degree=gaussians.max_sh_degree,
        spatial_lr_scale=gaussians.spatial_lr_scale,
        purpose="Complete inference state; not exact training-resume state.",
        render_settings=dict(
            texture_rendering=bool(texture_rendering),
            depth_ratio=float(getattr(pipe, "depth_ratio", 0.0)),
            compute_cov3D_python=bool(getattr(pipe, "compute_cov3D_python", False)),
            convert_SHs_python=bool(getattr(pipe, "convert_SHs_python", False)),
        ),
    )
    if scene is not None:
        result["cameras"] = camera_manifest(scene.getTrainCameras())
        result["data_settings"] = dict(
            resolution=scene.args.resolution,
            white_background=scene.args.white_background,
            world_scale=0.1,
        )
    validate_inference_state(result)
    return result


def validate_inference_state(state):
    if not isinstance(state, dict) or "parameters" not in state:
        raise ValueError(
            "Not a complete inference snapshot; native legacy PLY/MLP/env files omit learned parameters."
        )
    if state.get("format_version", 0) not in (0, 1):
        raise ValueError("Unsupported inference snapshot version")
    required = set(STATE_FIELDS) | {"mlp"}
    missing = required - state["parameters"].keys()
    if missing:
        raise ValueError(f"Incomplete inference snapshot; missing {sorted(missing)}")
    for key in ("iteration", "active_sh_degree", "max_sh_degree", "spatial_lr_scale"):
        if key not in state:
            raise ValueError(f"Missing snapshot metadata: {key}")
    for key, value in state["parameters"].items():
        tensors = value.values() if key == "mlp" else [value]
        if any(
            not isinstance(t, torch.Tensor) or not bool(torch.isfinite(t).all())
            for t in tensors
        ):
            raise ValueError(f"Invalid or non-finite snapshot tensors: {key}")
    count = state["parameters"]["_xyz"].shape[0]
    shapes = {
        "_xyz": (count, 3),
        "_albedo": (count, 7),
        "_scaling": (count, 2),
        "_rotation": (count, 4),
        "_opacity": (count, 1),
        "_env": (3, 1, 1),
        "_offset": (5, 1, 1),
    }
    for key, shape in shapes.items():
        if tuple(state["parameters"][key].shape) != shape:
            raise ValueError(f"Unexpected shape for {key}: expected {shape}")


def save_inference_state(gaussians, path, iteration, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = capture_inference_state(gaussians, iteration, **kwargs)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_inference_state(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Complete inference snapshot missing: {path}. Legacy PLY/MLP/env alone cannot restore offset."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    validate_inference_state(state)
    if "render_settings" not in state:
        # Compatibility settings for unversioned 40K all-view snapshots.
        if state["iteration"] != 40000:
            raise ValueError(
                "Legacy snapshot lacks render settings; only the 40K all-view format is supported"
            )
        state = dict(
            state,
            render_settings=dict(
                texture_rendering=True,
                depth_ratio=0.0,
                compute_cov3D_python=False,
                convert_SHs_python=True,
            ),
            data_settings=dict(resolution=1, white_background=False, world_scale=0.1),
        )
    return state


def restore_inference_state(gaussians, state, device="cuda"):
    validate_inference_state(state)
    if gaussians.max_sh_degree != state["max_sh_degree"]:
        raise ValueError("Snapshot SH degree differs from the configured model")
    for key in STATE_FIELDS:
        value = state["parameters"][key].to(device).clone()
        setattr(gaussians, key, nn.Parameter(value) if key in PARAMETERS else value)
    gaussians.mlp.load_state_dict(state["parameters"]["mlp"], strict=True)
    gaussians.active_sh_degree = state["active_sh_degree"]
    gaussians.spatial_lr_scale = state["spatial_lr_scale"]
