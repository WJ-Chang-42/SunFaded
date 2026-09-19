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

import torch


def embedding_func(pose):
    out = []
    N_freqs = 6
    freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
    for freq in freq_bands:
        for func in [torch.sin, torch.cos]:
            out += [func(freq * pose)]
    return torch.cat(out, -1)


def phong_lighting_image(albedo, specular, shininess, normal, dir_light, dir_pp, light):
    """Diffuse-only image-space lighting on CHW tensors. Specular arguments are retained for compatibility but unused."""
    NdotL = torch.sum(normal * dir_light, dim=0, keepdim=True)
    diffuse = albedo * light * torch.clamp(NdotL, min=0.0)
    result = diffuse
    return result


def phong_lighting(albedo, specular, shininess, normal, dir_light, dir_pp):
    """Historical per-point diffuse/specular helper; not called by the active renderer."""
    dot_product = torch.sum(normal * dir_light, dim=1, keepdim=True)
    normal = torch.where(dot_product < 0, -normal, normal)
    NdotL = torch.sum(normal * dir_light, dim=1)
    diffuse = albedo * NdotL.unsqueeze(1)
    reflect_dir = (
        2 * torch.sum(dir_light * normal, dim=1).unsqueeze(1) * normal - dir_light
    )
    reflect_dir = reflect_dir / reflect_dir.norm(dim=1, keepdim=True)
    RdotV = torch.clamp(torch.sum(reflect_dir * dir_pp, dim=1, keepdim=True), min=0.0)
    spec = specular * RdotV**shininess
    result = diffuse + spec
    return result


def rotmat2qvec(R):
    R = R.flatten()
    (Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz) = R
    K = (
        torch.tensor(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ],
            device=R.device,
        )
        / 3.0
    )
    (eigvals, eigvecs) = torch.linalg.eigh(K)
    idx = torch.argmax(eigvals)
    qvec = eigvecs[:, idx]
    qvec = qvec[[3, 0, 1, 2]]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec
