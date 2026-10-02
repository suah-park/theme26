"""Memory-bounded staged inference on one canonical grayscale crop.

Learned directional outputs stay raw here. Range policies and observed-slice
reinsertion belong to the workflow, and must be explicitly configured there.
"""

from __future__ import annotations

import numpy as np
import torch

from theme26.contracts import SamplingSpec
from theme26.models.saint.fdm import build_filter_distance_matrix


@torch.inference_mode()
def reconstruct_directions(
    ami: torch.nn.Module,
    sparse: np.ndarray,
    *,
    sampling: SamplingSpec,
    spacing_um_xyz: tuple[float, float, float],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Map float32 [X,Y,Z_lr] to two float32 [X,Y,R*Z_lr] CPU volumes.

Only a marginal batch is resident on the GPU. This does not call RFN or build
an autograd graph. Both directions use the same AMI weights.
"""
    if sparse.ndim != 3 or sparse.dtype != np.float32:
        raise TypeError("sparse must be float32 [X,Y,Z_lr]")
    if not np.isfinite(sparse).all() or np.any((sparse < 0) | (sparse > 1)):
        raise ValueError("sparse must contain finite grayscale values in [0,1]")
    if batch_size < 1 or sampling.offset != 0:
        raise ValueError("positive batch_size and zero sampling offset are required")
    if getattr(ami, "factor", None) != sampling.factor:
        raise ValueError("AMI factor differs from sampling")
    shape = sparse.shape[:2] + (sparse.shape[2] * sampling.factor,)
    outputs = []
    was_training = ami.training
    ami.eval()
    try:
        for axis, direction in ((0, "sagittal"), (1, "coronal")):
            output = np.empty(shape, dtype=np.float32)
            fdm = build_filter_distance_matrix(
                factor=sampling.factor, kernel_size=ami.kernel_size,
                hr_spacing_um_xyz=spacing_um_xyz, marginal_axis=direction,
                device=device,
            )
            for start in range(0, sparse.shape[axis], batch_size):
                planes = []
                for index in range(start, min(start + batch_size, sparse.shape[axis])):
                    neighbors = [max(index - 1, 0), index, min(index + 1, sparse.shape[axis] - 1)]
                    plane = sparse[neighbors, :, :] if axis == 0 else sparse[:, neighbors, :].transpose(1, 0, 2)
                    planes.append(plane)
                inputs = torch.from_numpy(np.stack(planes)).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=use_amp and device.type == "cuda"):
                    prediction = ami(inputs, fdm)
                values = prediction[:, 0].float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("AMI produced non-finite output")
                if axis == 0:
                    output[start:start + len(planes)] = values
                else:
                    output[:, start:start + len(planes)] = values.transpose(1, 0, 2)
            outputs.append(output)
    finally:
        ami.train(was_training)
    return outputs[0], outputs[1]


@torch.inference_mode()
def fuse_directions(
    rfn: torch.nn.Module,
    sagittal: np.ndarray,
    coronal: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Fuse matching float32 [X,Y,Z] volumes in axial GPU batches."""
    if sagittal.shape != coronal.shape or sagittal.ndim != 3:
        raise ValueError("directional volumes must have identical [X,Y,Z] shapes")
    if sagittal.dtype != np.float32 or coronal.dtype != np.float32:
        raise TypeError("directional volumes must be float32")
    if not np.isfinite(sagittal).all() or not np.isfinite(coronal).all():
        raise ValueError("directional volumes must be finite")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = np.empty_like(sagittal)
    was_training = rfn.training
    rfn.eval()
    try:
        for start in range(0, sagittal.shape[2], batch_size):
            stop = min(start + batch_size, sagittal.shape[2])
            inputs = np.stack((sagittal[..., start:stop], coronal[..., start:stop]), axis=0)
            inputs = torch.from_numpy(np.ascontiguousarray(inputs.transpose(3, 0, 1, 2))).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp and device.type == "cuda"):
                prediction = rfn.forward_slices(inputs)
            values = prediction[:, 0].float().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("RFN produced non-finite output")
            output[..., start:stop] = values.transpose(1, 2, 0)
    finally:
        rfn.train(was_training)
    return output
