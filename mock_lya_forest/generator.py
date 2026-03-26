"""Core mock-field evaluation routines based on superposed real plane waves."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .models import PlaneWaveSet, SightlineCatalog, SightlineSpectrum
from .noise import add_global_gaussian_noise, build_global_ivar


def _valid_pixel_mask(sightline: SightlineSpectrum) -> np.ndarray:
    """Return the pixels that are valid inputs for mock generation."""

    return (
        np.isfinite(sightline.wavelength)
        & np.isfinite(sightline.cont)
        & np.isfinite(sightline.delta)
        & np.isfinite(sightline.ivar)
    )


def evaluate_plane_wave_sum(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    wave_set: PlaneWaveSet,
) -> np.ndarray:
    """Evaluate the real-valued plane-wave sum at a set of Cartesian positions."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, and z must have matching shapes")

    delta_mock = np.zeros_like(x, dtype=np.float64)
    for k_vector, phase in zip(wave_set.k_vectors, wave_set.phases, strict=True):
        phase_argument = k_vector[0] * x + k_vector[1] * y + k_vector[2] * z + phase
        delta_mock += np.cos(phase_argument)
    return delta_mock / np.sqrt(wave_set.k_vectors.shape[0])


def generate_mock_for_sightline(sightline: SightlineSpectrum, wave_set: PlaneWaveSet) -> np.ndarray:
    """Generate the noiseless mock delta field for one sightline."""

    if sightline.x is None or sightline.y is None or sightline.z is None:
        raise ValueError(f"Sightline {sightline.name!r} is missing Cartesian coordinates")
    return evaluate_plane_wave_sum(sightline.x, sightline.y, sightline.z, wave_set)


def generate_mock_catalog(
    catalog: SightlineCatalog,
    wave_set: PlaneWaveSet,
    noise_sigma: float,
    seed: int | None = None,
) -> SightlineCatalog:
    """Generate a full mock catalog and apply the requested global noise model."""

    rng = np.random.default_rng(seed)
    updated_sightlines: list[SightlineSpectrum] = []

    for sightline in catalog.sightlines:
        valid_mask = _valid_pixel_mask(sightline)
        noiseless_delta = generate_mock_for_sightline(sightline, wave_set)
        noisy_delta = add_global_gaussian_noise(noiseless_delta, noise_sigma, rng, valid_mask=valid_mask)
        output_delta = np.full(noisy_delta.shape, np.nan, dtype=np.float32)
        output_delta[valid_mask] = noisy_delta[valid_mask].astype(np.float32)
        output_ivar = build_global_ivar(valid_mask, noise_sigma)

        updated_sightlines.append(
            replace(
                sightline,
                delta=output_delta,
                ivar=output_ivar,
            )
        )

    return SightlineCatalog(
        sightlines=updated_sightlines,
        primary_header=catalog.primary_header.copy(),
        reference_redshift=catalog.reference_redshift,
        field_center_ra_deg=catalog.field_center_ra_deg,
        field_center_dec_deg=catalog.field_center_dec_deg,
    )