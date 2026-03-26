"""Dataclasses shared across FITS I/O, geometry, and mock generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.io.fits import Header


@dataclass(slots=True)
class SightlineSpectrum:
    name: str
    los_id: str
    target_id: str
    ra_deg: float
    dec_deg: float
    quasar_redshift: float
    wavelength: np.ndarray
    delta: np.ndarray
    ivar: np.ndarray
    cont: np.ndarray
    mean_snr: float | None = None
    mean_resolution: float | None = None
    pixel_redshift: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    header: Header = field(default_factory=Header)


@dataclass(slots=True)
class SightlineCatalog:
    sightlines: list[SightlineSpectrum]
    primary_header: Header = field(default_factory=Header)
    reference_redshift: float | None = None
    field_center_ra_deg: float | None = None
    field_center_dec_deg: float | None = None


@dataclass(slots=True)
class PlaneWaveSet:
    k_vectors: np.ndarray
    phases: np.ndarray
    k_min: float
    k_max: float
    seed: int | None = None