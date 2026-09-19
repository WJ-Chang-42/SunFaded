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

"""Legacy training configuration and stage selection, without interval corrections."""

import json
import ast


def apply_legacy_freeze(optimizer, iteration, frozen_lr_cache):
    """Preserve the original freeze/restore sequence, including post-35K alternation."""
    if iteration > 30000 and frozen_lr_cache is None:
        frozen_lr_cache = {
            group["name"]: group["lr"] for group in optimizer.param_groups
        }
        for group in optimizer.param_groups:
            if group["name"] == "opacity":
                continue
            group["lr"] = 0.0
    elif iteration > 35000 and frozen_lr_cache is not None:
        for group in optimizer.param_groups:
            if group["name"] in frozen_lr_cache:
                group["lr"] = frozen_lr_cache[group["name"]]
        frozen_lr_cache = None
    return frozen_lr_cache


def read_saved_arguments(text):
    """Read the upstream Namespace format as literals, never executable Python."""
    node = ast.parse(text, mode="eval").body
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "Namespace"
        or node.args
    ):
        raise ValueError(
            "Expected a saved Namespace containing only literal keyword arguments"
        )
    result = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError("Expanded keyword arguments are not supported")
        if keyword.arg != "eval":
            result[keyword.arg] = ast.literal_eval(keyword.value)
    return result


def load_training_config(config_path):
    """Read the supplied JSON configuration; invalid files fail instead of selecting a different schedule."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        required_fields = ["iteration_configs", "special_operations", "regularization"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        print(f"Loaded training configuration: {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Invalid training configuration {config_path}: {e}") from e


def get_iteration_config(config, iteration):
    """Use the first matching half-open interval, preserving the source defaults and uncovered-iteration fallback."""
    if config is None:
        return {
            "gaussian_num": False,
            "texture_rendering": iteration > 5000,
            "uncertainty": False,
            "gaussian_update": True,
        }
    for cfg in config["iteration_configs"]:
        start = cfg["start_iteration"]
        end = cfg["end_iteration"]
        if start <= iteration and (end == -1 or iteration < end):
            return {
                "gaussian_num": cfg["gaussian_num"],
                "texture_rendering": cfg["texture_rendering"],
                "uncertainty": cfg["uncertainty"],
                "gaussian_update": cfg["gaussian_update"],
            }
    return {
        "gaussian_num": False,
        "texture_rendering": True,
        "uncertainty": False,
        "gaussian_update": True,
    }
