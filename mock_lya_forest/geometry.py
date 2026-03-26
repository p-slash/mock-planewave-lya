"""Coordinate transforms from sky positions and wavelengths to local Cartesian space."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from .cosmology import (
    LYA_REST_WAVELENGTH_ANGSTROM,
    reference_comoving_distance,
    relative_comoving_distance,
    wavelength_to_absorber_redshift,
)
from .models import SightlineCatalog, SightlineSpectrum


def infer_field_center(sightlines: list[SightlineSpectrum]) -> tuple[float, float]:
    """Infer a field center from the mean sky position of all sightlines."""

    if not sightlines:
        raise ValueError("Cannot infer a field center from an empty sightline list")

    ra_deg = np.asarray([sightline.ra_deg for sightline in sightlines], dtype=np.float64)
    dec_deg = np.asarray([sightline.dec_deg for sightline in sightlines], dtype=np.float64)
    return float(np.mean(ra_deg)), float(np.mean(dec_deg))


def spherical_angles_from_field_center(
    ra_deg: float,
    dec_deg: float,
    ra_center_deg: float,
    dec_center_deg: float,
) -> tuple[float, float]:
    """Return the polar angle and azimuth relative to the chosen field center."""

    origin = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg, frame="icrs")
    sightline = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    theta_rad = origin.separation(sightline).to_value(u.rad)
    phi_rad = origin.position_angle(sightline).to_value(u.rad)
    return float(theta_rad), float(phi_rad)


def spherical_to_cartesian(
    chi_cmpc_h: np.ndarray,
    theta_rad: float,
    phi_rad: float,
    chi_ref_cmpc_h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical coordinates around the field center into local Cartesian coordinates."""

    chi_cmpc_h = np.asarray(chi_cmpc_h, dtype=np.float64)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    x_coord = chi_cmpc_h * sin_theta * np.cos(phi_rad)
    y_coord = chi_cmpc_h * sin_theta * np.sin(phi_rad)
    z_coord = chi_cmpc_h * cos_theta - chi_ref_cmpc_h
    return x_coord, y_coord, z_coord


def assign_cartesian_coordinates(
    catalog: SightlineCatalog,
    cosmology: FlatLambdaCDM,
    z_ref: float,
    field_center_ra_deg: float | None = None,
    field_center_dec_deg: float | None = None,
    lambda_lya_rest: float = LYA_REST_WAVELENGTH_ANGSTROM,
) -> SightlineCatalog:
    """Attach relative Cartesian coordinates and pixel redshifts to each sightline."""

    if not catalog.sightlines:
        raise ValueError("Cannot assign coordinates to an empty sightline catalog")

    if field_center_ra_deg is None or field_center_dec_deg is None:
        inferred_ra_deg, inferred_dec_deg = infer_field_center(catalog.sightlines)
        if field_center_ra_deg is None:
            field_center_ra_deg = inferred_ra_deg
        if field_center_dec_deg is None:
            field_center_dec_deg = inferred_dec_deg

    chi_ref_cmpc_h = reference_comoving_distance(z_ref, cosmology)
    updated_sightlines: list[SightlineSpectrum] = []

    for sightline in catalog.sightlines:
        pixel_redshift = wavelength_to_absorber_redshift(sightline.wavelength, lambda_lya_rest=lambda_lya_rest)
        chi_coord = relative_comoving_distance(pixel_redshift, z_ref=0.0, cosmology=cosmology)
        theta_rad, phi_rad = spherical_angles_from_field_center(
            sightline.ra_deg,
            sightline.dec_deg,
            field_center_ra_deg,
            field_center_dec_deg,
        )
        x_coord, y_coord, z_coord = spherical_to_cartesian(chi_coord, theta_rad, phi_rad, chi_ref_cmpc_h)

        updated_sightlines.append(
            replace(
                sightline,
                pixel_redshift=np.asarray(pixel_redshift, dtype=np.float64),
                x=np.asarray(x_coord, dtype=np.float64),
                y=np.asarray(y_coord, dtype=np.float64),
                z=np.asarray(z_coord, dtype=np.float64),
            )
        )

    return SightlineCatalog(
        sightlines=updated_sightlines,
        primary_header=catalog.primary_header.copy(),
        reference_redshift=float(z_ref),
        field_center_ra_deg=float(field_center_ra_deg),
        field_center_dec_deg=float(field_center_dec_deg),
    )