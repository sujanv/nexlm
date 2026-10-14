"""
Milestone 42: Model Merging Engine (SLERP & DARE / Task Arithmetic)

Post-training model merging combines capabilities of multiple specialized fine-tuned checkpoints
without additional GPU training or dataset access:

1. Spherical Linear Interpolation (SLERP):
   Interpolates weights along the spherical arc rather than linearly, preserving geometric norms:
       theta = arccos( (v1 . v2) / (|v1| * |v2|) )
       v_merged = (sin((1-t)*theta) / sin(theta)) * v1 + (sin(t*theta) / sin(theta)) * v2

2. DARE (Drop And REscale) Task Arithmetic (Yu et al., 2023):
   Extracts task delta vectors: tau = theta_ft - theta_base
   Drops random/unimportant delta parameters with probability p and rescales remaining parameters by 1/(1-p).
   Eliminates parameter interference when combining multi-task checkpoints.
"""

import copy
from typing import Dict, List, Optional
import torch
import torch.nn as nn


def slerp_tensor(v1: torch.Tensor, v2: torch.Tensor, t: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical Linear Interpolation between two weight tensors.
    """
    # Flatten tensors to 1D vectors
    orig_shape = v1.shape
    u1 = v1.reshape(-1).float()
    u2 = v2.reshape(-1).float()

    # Norms
    norm_u1 = torch.norm(u1)
    norm_u2 = torch.norm(u2)

    if norm_u1 < eps or norm_u2 < eps:
        # Fall back to linear interpolation if zero vector
        res = (1.0 - t) * u1 + t * u2
        return res.reshape(orig_shape).to(v1.dtype)

    # Normalize vectors
    u1_hat = u1 / norm_u1
    u2_hat = u2 / norm_u2

    # Dot product and angle theta
    dot = torch.clamp(torch.dot(u1_hat, u2_hat), -1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)

    # If vectors are almost collinear, use linear interpolation
    if torch.abs(theta) < eps:
        res = (1.0 - t) * u1 + t * u2
        return res.reshape(orig_shape).to(v1.dtype)

    sin_theta = torch.sin(theta)
    scale1 = torch.sin((1.0 - t) * theta) / sin_theta
    scale2 = torch.sin(t * theta) / sin_theta

    res = scale1 * u1 + scale2 * u2
    return res.reshape(orig_shape).to(v1.dtype)


def merge_models_slerp(model1: nn.Module, model2: nn.Module, t: float = 0.5) -> nn.Module:
    """
    Merges two models using Spherical Linear Interpolation (SLERP) across all matching parameters.
    """
    merged_model = copy.deepcopy(model1)
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    merged_state = {}

    for key in state1:
        if key in state2:
            p1 = state1[key]
            p2 = state2[key]
            if p1.is_floating_point():
                merged_state[key] = slerp_tensor(p1, p2, t=t)
            else:
                merged_state[key] = p1
        else:
            merged_state[key] = state1[key]

    merged_model.load_state_dict(merged_state)
    return merged_model


def dare_task_vector(task_vector: torch.Tensor, drop_rate: float = 0.2, seed: Optional[int] = None) -> torch.Tensor:
    """
    Applies Drop And REscale (DARE) to a task delta vector:
    Randomly drops drop_rate fraction of deltas and rescales retained deltas by 1 / (1 - drop_rate).
    """
    if drop_rate <= 0.0:
        return task_vector
    if drop_rate >= 1.0:
        return torch.zeros_like(task_vector)

    if seed is not None:
        torch.manual_seed(seed)

    # Bernoulli drop mask: keep with probability (1 - drop_rate)
    keep_prob = 1.0 - drop_rate
    mask = torch.bernoulli(torch.full_like(task_vector, keep_prob))
    rescaled_vector = (task_vector * mask) / keep_prob
    return rescaled_vector


def merge_models_dare(
    base_model: nn.Module,
    fine_tuned_models: List[nn.Module],
    weights: Optional[List[float]] = None,
    drop_rate: float = 0.2,
) -> nn.Module:
    """
    Merges multiple fine-tuned models onto a base model using DARE task arithmetic:
        theta_merged = theta_base + sum_i (w_i * DARE(theta_i - theta_base))
    """
    if weights is None:
        weights = [1.0 / len(fine_tuned_models)] * len(fine_tuned_models)

    merged_model = copy.deepcopy(base_model)
    base_state = base_model.state_dict()
    merged_state = {}

    for key, base_param in base_state.items():
        if not base_param.is_floating_point():
            merged_state[key] = base_param
            continue

        accumulated_delta = torch.zeros_like(base_param)
        for model_i, w_i in zip(fine_tuned_models, weights):
            ft_param = model_i.state_dict()[key]
            task_delta = ft_param - base_param
            dare_delta = dare_task_vector(task_delta, drop_rate=drop_rate)
            accumulated_delta = accumulated_delta + (w_i * dare_delta)

        merged_state[key] = base_param + accumulated_delta

    merged_model.load_state_dict(merged_state)
    return merged_model
