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
from torch import nn


class LightMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64 + 12, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softplus(beta=1.0)
        )

    def forward(self, x, y):
        x = self.mlp(x)
        feature = x
        shadow = self.mlp2(torch.cat([y, feature], dim=-1))
        return (shadow[..., :3], None)
