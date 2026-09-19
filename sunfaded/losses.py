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

"""Depth losses for SunFaded."""

import torch


def silog_loss(pred, gt, mask, lam=0.85, eps=1e-06):
    m = (mask > 0).float()
    d_pred = torch.clamp(pred, min=eps)
    d_gt = torch.clamp(gt, min=eps)
    diff = (torch.log(d_pred) - torch.log(d_gt)) * m
    N = m.sum() + 1e-08
    mean = diff.sum() / N
    return diff.pow(2).sum() / N - lam * mean**2


def pearson_corrcoef_min(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Differentiable Pearson correlation preserving the original dtype, flattening, epsilon, and fewer-than-two-elements behavior."""
    x = pred.reshape(-1)
    y = target.reshape(-1)
    if x.numel() != y.numel():
        (x, y) = torch.broadcast_tensors(x, y)
        x = x.reshape(-1)
        y = y.reshape(-1)
    n = x.numel()
    if n < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    xm = x - x.mean()
    ym = y - y.mean()
    xstd = xm.norm() / n**0.5
    ystd = ym.norm() / n**0.5
    denom = (xstd * ystd).clamp_min(eps) * n
    r = xm.flatten() @ ym.flatten() / denom
    return r.to(x.dtype)
