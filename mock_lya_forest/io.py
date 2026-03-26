"""FITS I/O helpers for LATIS-style per-sightline delta files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from .models import SightlineCatalog, SightlineSpectrum


PRIMARY_RESERVED_KEYS = {
    "SIMPLE",
    "BITPIX",
    "NAXIS",
    "EXTEND",
}

TABLE_RESERVED_KEYS = {
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
}

TABLE_RESERVED_PREFIXES = ("TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL", "TZERO")


def _copy_primary_header(header: fits.Header) -> fits.Header:
    """Copy non-structural cards from a primary FITS header."""

    new_header = fits.Header()
    for card in header.cards:
        if card.keyword in PRIMARY_RESERVED_KEYS:
            continue
        new_header.append(card)
    return new_header


def _copy_extension_header(header: fits.Header) -> fits.Header:
    """Copy non-structural cards from a binary-table extension header."""

    new_header = fits.Header()
    for card in header.cards:
        keyword = card.keyword
        if keyword in TABLE_RESERVED_KEYS:
            continue
        if keyword.startswith(TABLE_RESERVED_PREFIXES):
            continue
        new_header.append(card)
    return new_header


def _extract_required_columns(data: fits.FITS_rec, extname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the required LATIS sightline columns from one table HDU."""

    required_columns = ("LAMBDA", "DELTA", "IVAR", "CONT")
    missing = [name for name in required_columns if name not in data.dtype.names]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Extension {extname!r} is missing required columns: {missing_list}")

    wavelength = np.asarray(data["LAMBDA"], dtype=np.float32)
    delta = np.asarray(data["DELTA"], dtype=np.float32)
    ivar = np.asarray(data["IVAR"], dtype=np.float32)
    cont = np.asarray(data["CONT"], dtype=np.float32)
    return wavelength, delta, ivar, cont


def _build_sightline(hdu: fits.BinTableHDU) -> SightlineSpectrum:
    """Convert one sightline table HDU into a SightlineSpectrum object."""

    if hdu.data is None:
        raise ValueError(f"Extension {hdu.name!r} does not contain table data")

    wavelength, delta, ivar, cont = _extract_required_columns(hdu.data, hdu.name)
    header = hdu.header.copy()

    return SightlineSpectrum(
        name=hdu.name,
        los_id=str(header.get("LOS_ID", hdu.name)),
        target_id=str(header.get("TARGETID", hdu.name)),
        ra_deg=float(header["RA"]),
        dec_deg=float(header["DEC"]),
        quasar_redshift=float(header["Z"]),
        wavelength=wavelength,
        delta=delta,
        ivar=ivar,
        cont=cont,
        mean_snr=_optional_float(header, "MEANSNR"),
        mean_resolution=_optional_float(header, "MEANRESO"),
        header=header,
    )


def _optional_float(header: fits.Header, key: str) -> float | None:
    """Read a floating-point header value when present, else return None."""

    if key not in header:
        return None
    return float(header[key])


def _build_table_hdu(sightline: SightlineSpectrum) -> fits.BinTableHDU:
    """Build a LATIS-style binary table HDU from one sightline object."""

    _validate_sightline_arrays(sightline)

    columns = [
        fits.Column(name="LAMBDA", format="E", array=np.asarray(sightline.wavelength, dtype=np.float32)),
        fits.Column(name="DELTA", format="E", array=np.asarray(sightline.delta, dtype=np.float32)),
        fits.Column(name="IVAR", format="E", array=np.asarray(sightline.ivar, dtype=np.float32)),
        fits.Column(name="CONT", format="E", array=np.asarray(sightline.cont, dtype=np.float32)),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name=sightline.name)

    for card in _copy_extension_header(sightline.header).cards:
        hdu.header.append(card)

    hdu.header["EXTNAME"] = sightline.name
    hdu.header["LOS_ID"] = sightline.los_id
    hdu.header["TARGETID"] = sightline.target_id
    hdu.header["RA"] = float(sightline.ra_deg)
    hdu.header["DEC"] = float(sightline.dec_deg)
    hdu.header["Z"] = float(sightline.quasar_redshift)
    if sightline.mean_resolution is not None:
        hdu.header["MEANRESO"] = float(sightline.mean_resolution)
    if sightline.mean_snr is not None:
        hdu.header["MEANSNR"] = float(sightline.mean_snr)

    return hdu


def _validate_sightline_arrays(sightline: SightlineSpectrum) -> None:
    """Validate that the core sightline arrays are one-dimensional and aligned."""

    arrays = {
        "wavelength": np.asarray(sightline.wavelength),
        "delta": np.asarray(sightline.delta),
        "ivar": np.asarray(sightline.ivar),
        "cont": np.asarray(sightline.cont),
    }
    lengths = {name: array.shape[0] for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Sightline {sightline.name!r} has inconsistent array lengths: {lengths}")
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"Sightline {sightline.name!r} field {name!r} must be one-dimensional")


def read_sightline_fits(path: str | Path) -> SightlineCatalog:
    """Read a multi-extension LATIS-style FITS file into a sightline catalog."""

    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"Expected at least one sightline extension in {path}")

        sightlines = [_build_sightline(hdu) for hdu in hdul[1:]]
        primary_header = hdul[0].header.copy()

    return SightlineCatalog(
        sightlines=sightlines,
        primary_header=primary_header,
    )


def write_sightline_fits(
    path: str | Path,
    catalog: SightlineCatalog,
    overwrite: bool = False,
) -> None:
    """Write a sightline catalog to a LATIS-style multi-extension FITS file."""

    path = Path(path)
    primary_hdu = fits.PrimaryHDU()
    for card in _copy_primary_header(catalog.primary_header).cards:
        primary_hdu.header.append(card)

    hdus: list[fits.hdu.base.ExtensionHDU] = [primary_hdu]
    hdus.extend(_build_table_hdu(sightline) for sightline in catalog.sightlines)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)