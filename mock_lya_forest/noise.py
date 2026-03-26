"""Noise models for converting noiseless mock deltas into observed-like outputs."""

from __future__ import annotations

import numpy as np


def add_global_gaussian_noise(
    delta: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Add one global Gaussian noise level to the valid delta pixels."""

    if noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")

    delta = np.asarray(delta, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(delta)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    noisy_delta = delta.copy()
    if noise_sigma == 0:
        return noisy_delta

    noise = rng.normal(loc=0.0, scale=noise_sigma, size=delta.shape)
    noisy_delta[valid_mask] = noisy_delta[valid_mask] + noise[valid_mask]
    return noisy_delta


def build_global_ivar(valid_mask: np.ndarray, noise_sigma: float) -> np.ndarray:
    """Build an inverse-variance array implied by the global noise level."""

    valid_mask = np.asarray(valid_mask, dtype=bool)
    ivar = np.zeros(valid_mask.shape, dtype=np.float32)
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")
    if noise_sigma == 0:
        ivar[valid_mask] = np.inf
        return ivar

    ivar[valid_mask] = np.float32(1.0 / noise_sigma**2)
    return ivar