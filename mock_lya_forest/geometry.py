"""Coordinate transforms from sky positions and wavelengths to local Cartesian space."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
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


def rotate_to_field_center(
    ra_deg: float,
    dec_deg: float,
    ra_center_deg: float,
    dec_center_deg: float,
) -> tuple[float, float]:
    """Rotate one sky position into a local offset frame centered on the field."""

    origin = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg, frame="icrs")
    sightline = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    offset_frame = SkyOffsetFrame(origin=origin)
    rotated = sightline.transform_to(offset_frame)
    return float(rotated.lon.to_value(u.rad)), float(rotated.lat.to_value(u.rad))


def project_to_plane_parallel(
    lon_rot_rad: float,
    lat_rot_rad: float,
    chi_ref_cmpc_h: float,
) -> tuple[float, float]:
    """Project angular offsets into transverse plane-parallel coordinates."""

    return chi_ref_cmpc_h * lon_rot_rad, chi_ref_cmpc_h * lat_rot_rad


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
        z_coord = relative_comoving_distance(pixel_redshift, z_ref=z_ref, cosmology=cosmology)
        lon_rot_rad, lat_rot_rad = rotate_to_field_center(
            sightline.ra_deg,
            sightline.dec_deg,
            field_center_ra_deg,
            field_center_dec_deg,
        )
        x_coord, y_coord = project_to_plane_parallel(lon_rot_rad, lat_rot_rad, chi_ref_cmpc_h)

        updated_sightlines.append(
            replace(
                sightline,
                pixel_redshift=np.asarray(pixel_redshift, dtype=np.float64),
                x=np.full_like(z_coord, x_coord, dtype=np.float64),
                y=np.full_like(z_coord, y_coord, dtype=np.float64),
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