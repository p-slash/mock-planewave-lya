"""Plane-wave sampling utilities for the mock forest model."""

from __future__ import annotations

import numpy as np

from .models import PlaneWaveSet


def sample_wave_vectors(
    n_waves: int,
    k_min: float,
    k_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample isotropic 3D wave vectors with magnitudes drawn from a fixed range."""

    if n_waves <= 0:
        raise ValueError("n_waves must be positive")
    if k_min <= 0 or k_max <= 0:
        raise ValueError("k_min and k_max must be positive")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min")

    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_waves)
    cos_theta = rng.uniform(-1.0, 1.0, size=n_waves)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    k_mag = rng.uniform(k_min, k_max, size=n_waves)

    kx = k_mag * sin_theta * np.cos(phi)
    ky = k_mag * sin_theta * np.sin(phi)
    kz = k_mag * cos_theta
    return np.column_stack((kx, ky, kz)).astype(np.float64)


def sample_phases(n_waves: int, rng: np.random.Generator) -> np.ndarray:
    """Sample scalar phases uniformly on the interval [0, 2pi)."""

    if n_waves <= 0:
        raise ValueError("n_waves must be positive")
    return rng.uniform(0.0, 2.0 * np.pi, size=n_waves).astype(np.float64)


def build_plane_wave_set(
    n_waves: int,
    k_min: float,
    k_max: float,
    seed: int | None = None,
) -> PlaneWaveSet:
    """Build a reproducible collection of random plane-wave parameters."""

    rng = np.random.default_rng(seed)
    k_vectors = sample_wave_vectors(n_waves, k_min, k_max, rng)
    phases = sample_phases(n_waves, rng)
    return PlaneWaveSet(
        k_vectors=k_vectors,
        phases=phases,
        k_min=float(k_min),
        k_max=float(k_max),
        seed=seed,
    )