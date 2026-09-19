"""Float-tensor training reconstruction metrics; PNGs are previews, not metric inputs."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gaussian_renderer import render
from lpipsPyTorch import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def write_preview(path, tensor):
    array = (
        (tensor.detach().clamp(0, 1) * 255)
        .round()
        .to(torch.uint8)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def reconstruction_metrics(image, target, criterion):
    """Preserve CHW per-channel PSNR, upstream SSIM and AlexNet LPIPS on [0, 1]."""
    image, target = image.clamp(0, 1), target.clamp(0, 1)
    result = dict(
        l1=float(l1_loss(image, target).mean()),
        psnr=float(psnr(image, target).mean()),
        ssim=float(ssim(image, target).mean()),
        lpips=float(criterion(image, target).mean()),
    )
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError(
            "Non-finite reconstruction metric (including infinite PSNR for identical tensors)"
        )
    return result


@torch.no_grad()
def export_reconstruction(
    cameras, gaussians, pipe, background, output, texture_rendering=True, save_raw=False
):
    cameras = list(cameras)
    if not cameras:
        raise ValueError("Cannot evaluate an empty reconstruction set")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite reconstruction output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("render", "gt", "albedo", "normal") + (("raw",) if save_raw else ()):
        (output / folder).mkdir()
    criterion = LPIPS("alex", "0.1").cuda().eval()
    rows, seen = [], set()
    for camera in cameras:
        name = camera.image_name
        if name in seen or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Unsafe or duplicate image name: {name}")
        seen.add(name)
        package = render(
            camera, gaussians, pipe, background, texture_rendering=texture_rendering
        )
        for key in ("render", "albedo", "rend_normal", "surf_depth"):
            if not bool(torch.isfinite(package[key]).all()):
                raise FloatingPointError(f"Non-finite {key} for {name}")
        image = package["render"].clamp(0, 1)
        target = camera.original_image.cuda().clamp(0, 1)
        rows.append(
            dict(
                name=name,
                uid=camera.uid,
                width=camera.image_width,
                height=camera.image_height,
                **reconstruction_metrics(image, target, criterion),
            )
        )
        for key, value in (
            ("render", image),
            ("gt", target),
            ("albedo", package["albedo"]),
            ("normal", package["rend_normal"] * 0.5 + 0.5),
        ):
            write_preview(output / key / f"{name}.png", value)
        if save_raw:
            np.savez_compressed(
                output / "raw" / f"{name}.npz",
                render=image.cpu().numpy(),
                target=target.cpu().numpy(),
                depth=package["surf_depth"].cpu().numpy(),
            )
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("l1", "psnr", "ssim", "lpips")
    }
    result = dict(
        protocol="training_reconstruction",
        evaluated_views=len(rows),
        held_out_views=0,
        aggregates=aggregate,
        per_view=rows,
        metric_definition=dict(
            psnr="Mean RGB-channel PSNR for each CHW image, then mean over images",
            ssim="Unmodified repository implementation",
            lpips="AlexNet v0.1 on [0,1] tensors",
            support="Full frames at the training camera resolution; no added crop/mask",
            previews="Rounded 8-bit PNGs are not metric inputs",
        ),
    )
    with open(output / "metrics.json", "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print("Training reconstruction: " + json.dumps(aggregate))
    return result
