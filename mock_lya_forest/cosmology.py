"""Cosmology and wavelength-to-distance conversion utilities."""

from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM


DEFAULT_H0 = 67.74
DEFAULT_OM0 = 0.3089
DEFAULT_T_CMB0 = 2.7255
LYA_REST_WAVELENGTH_ANGSTROM = 1215.67


def build_cosmology(
    h0: float = DEFAULT_H0,
    om0: float = DEFAULT_OM0,
    tcmb0: float = DEFAULT_T_CMB0,
) -> FlatLambdaCDM:
    """Construct the fiducial flat LambdaCDM cosmology used by the pipeline."""

    return FlatLambdaCDM(H0=h0, Om0=om0, Tcmb0=tcmb0)


def wavelength_to_absorber_redshift(
    wavelength_angstrom: np.ndarray,
    lambda_lya_rest: float = LYA_REST_WAVELENGTH_ANGSTROM,
) -> np.ndarray:
    """Convert observed wavelength to Lyman-alpha absorber redshift."""

    wavelength_angstrom = np.asarray(wavelength_angstrom, dtype=np.float64)
    return wavelength_angstrom / lambda_lya_rest - 1.0


def redshift_to_comoving_distance(redshift: np.ndarray, cosmology: FlatLambdaCDM) -> np.ndarray:
    """Convert redshift to comoving distance in cMpc/h."""

    redshift = np.asarray(redshift, dtype=np.float64)
    h = cosmology.H0.value / 100.0
    return np.asarray(cosmology.comoving_distance(redshift).value * h, dtype=np.float64)


def relative_comoving_distance(
    redshift: np.ndarray,
    z_ref: float,
    cosmology: FlatLambdaCDM,
) -> np.ndarray:
    """Return comoving distance relative to a chosen reference redshift."""

    return redshift_to_comoving_distance(redshift, cosmology) - float(redshift_to_comoving_distance(np.array([z_ref]), cosmology)[0])


def reference_comoving_distance(z_ref: float, cosmology: FlatLambdaCDM) -> float:
    """Return the comoving distance of the reference redshift in cMpc/h."""

    return float(redshift_to_comoving_distance(np.array([z_ref]), cosmology)[0])