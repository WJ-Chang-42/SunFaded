"""Depth Pro construction with an explicit checkpoint and unchanged preprocessing."""

from dataclasses import replace
from pathlib import Path


def create_depth_model(checkpoint):
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Depth Pro checkpoint not found: {path}. Set --depth_pro_checkpoint."
        )
    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(path))
    model, transform = depth_pro.create_model_and_transforms(config=config)
    model = model.to("cuda")
    model.eval()
    return model, transform


def attach_depth_priors(cameras, checkpoint):
    import torch
    from PIL import Image
    from utils.point_utils import depth_to_normal

    model, transform = create_depth_model(checkpoint)
    for camera in cameras:
        img = camera.original_image.detach().cpu()
        img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
        img = img.permute(1, 2, 0).contiguous()
        pil_img = Image.fromarray(img.numpy(), mode="RGB")
        prediction = model.infer(transform(pil_img).cuda(), f_px=None)
        camera.depth_map = prediction["depth"].cuda()
        camera.normal_map = depth_to_normal(camera, camera.depth_map).permute(2, 0, 1)
    del model, transform
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
