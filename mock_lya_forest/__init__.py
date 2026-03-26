"""Public package interface for mock Lyman-alpha forest generation."""

from .cosmology import build_cosmology
from .generator import generate_mock_catalog, generate_mock_for_sightline
from .geometry import assign_cartesian_coordinates, infer_field_center
from .io import read_sightline_fits, write_sightline_fits
from .models import PlaneWaveSet, SightlineCatalog, SightlineSpectrum
from .waves import build_plane_wave_set

__all__ = [
    "assign_cartesian_coordinates",
    "build_cosmology",
    "build_plane_wave_set",
    "generate_mock_catalog",
    "generate_mock_for_sightline",
    "infer_field_center",
    "PlaneWaveSet",
    "SightlineCatalog",
    "SightlineSpectrum",
    "read_sightline_fits",
    "write_sightline_fits",
]