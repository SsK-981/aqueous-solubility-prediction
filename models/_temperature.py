"""Append the two prepared temperature features to a model representation."""

import torch
from torch import Tensor


def _temperature(features: Tensor, temperature: Tensor | None, enabled: bool) -> Tensor:
    if not enabled:
        if temperature is not None:
            raise ValueError("Temperature was supplied to an unconditioned model")
        return features
    if temperature is None or temperature.shape != (features.shape[0], 2):
        raise ValueError("Temperature-enabled model requires temperature shape [B,2]")
    if not torch.isfinite(temperature).all():
        raise ValueError("Prepared temperature values must be finite")
    temperature = temperature.to(device=features.device, dtype=features.dtype)
    if features.ndim == 3:
        temperature = temperature[:, None, :].expand(-1, features.shape[1], -1)
    return torch.cat((features, temperature), dim=-1)
