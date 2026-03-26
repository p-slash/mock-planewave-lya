from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from mock_lya_forest import (
    assign_cartesian_coordinates,
    build_cosmology,
    build_plane_wave_set,
    generate_mock_catalog,
    read_sightline_fits,
    write_sightline_fits,
)
from mock_lya_forest.cli import main as cli_main


def _write_sample_input_fits(path: Path) -> None:
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header.add_comment("Synthetic test fixture for mock_lya_forest")

    def build_sightline_hdu(
        name: str,
        ra_deg: float,
        dec_deg: float,
        quasar_redshift: float,
        wavelength: np.ndarray,
        delta: np.ndarray,
        ivar: np.ndarray,
        cont: np.ndarray,
    ) -> fits.BinTableHDU:
        columns = [
            fits.Column(name="LAMBDA", format="E", array=wavelength.astype(np.float32)),
            fits.Column(name="DELTA", format="E", array=delta.astype(np.float32)),
            fits.Column(name="IVAR", format="E", array=ivar.astype(np.float32)),
            fits.Column(name="CONT", format="E", array=cont.astype(np.float32)),
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name=name)
        hdu.header["LOS_ID"] = name
        hdu.header["TARGETID"] = name
        hdu.header["RA"] = float(ra_deg)
        hdu.header["DEC"] = float(dec_deg)
        hdu.header["Z"] = float(quasar_redshift)
        hdu.header["BLINDING"] = "none"
        hdu.header["MEANRESO"] = 150.0
        hdu.header["MEANSNR"] = 2.0
        hdu.header["RSNR"] = 3.0
        hdu.header["DLTLAMBD"] = 1.0
        hdu.header["SMSCALE"] = 0
        return hdu

    wave_a = np.linspace(3800.0, 3810.0, 6)
    wave_b = np.linspace(3810.0, 3822.0, 7)
    hdus = [
        primary_hdu,
        build_sightline_hdu(
            name="D1-181266",
            ra_deg=0.640272019131642,
            dec_deg=-0.0816956142454111,
            quasar_redshift=2.372,
            wavelength=wave_a,
            delta=np.array([0.1, -0.1, 0.05, 0.0, 0.2, -0.2]),
            ivar=np.full(6, 4.0),
            cont=np.linspace(1.0, 1.5, 6),
        ),
        build_sightline_hdu(
            name="D1-181479",
            ra_deg=0.641903504651699,
            dec_deg=-0.0817059185455125,
            quasar_redshift=2.533,
            wavelength=wave_b,
            delta=np.array([0.0, 0.03, -0.02, 0.04, -0.01, 0.02, 0.01]),
            ivar=np.full(7, 5.0),
            cont=np.linspace(0.8, 1.3, 7),
        ),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestWithFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_path = Path(cls.temp_dir.name) / "sample-input.fits"
        _write_sample_input_fits(cls.data_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()


class TestSightlineFitsIO(TestWithFixture):
    def test_read_sightline_fits_loads_expected_structure(self) -> None:
        catalog = read_sightline_fits(self.data_path)

        self.assertEqual(len(catalog.sightlines), 2)
        first = catalog.sightlines[0]
        self.assertEqual(first.name, "D1-181266")
        self.assertEqual(first.los_id, "D1-181266")
        self.assertEqual(first.target_id, "D1-181266")
        self.assertEqual(len(first.wavelength), len(first.delta))
        self.assertEqual(len(first.wavelength), len(first.ivar))
        self.assertEqual(len(first.wavelength), len(first.cont))
        self.assertIn("COMMENT", catalog.primary_header)

    def test_write_sightline_fits_roundtrip_preserves_basic_metadata(self) -> None:
        catalog = read_sightline_fits(self.data_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "roundtrip.fits"
            write_sightline_fits(output_path, catalog, overwrite=True)
            roundtrip = read_sightline_fits(output_path)

        self.assertEqual(len(roundtrip.sightlines), len(catalog.sightlines))
        self.assertEqual(roundtrip.sightlines[0].name, catalog.sightlines[0].name)
        self.assertEqual(roundtrip.sightlines[0].los_id, catalog.sightlines[0].los_id)
        self.assertEqual(roundtrip.sightlines[0].target_id, catalog.sightlines[0].target_id)
        np.testing.assert_allclose(roundtrip.sightlines[0].wavelength, catalog.sightlines[0].wavelength)
        np.testing.assert_allclose(roundtrip.sightlines[0].delta, catalog.sightlines[0].delta)
        np.testing.assert_allclose(roundtrip.sightlines[0].ivar, catalog.sightlines[0].ivar)
        np.testing.assert_allclose(roundtrip.sightlines[0].cont, catalog.sightlines[0].cont)


class TestCoordinateAssignment(TestWithFixture):
    def test_assign_cartesian_coordinates_populates_relative_coordinates(self) -> None:
        catalog = read_sightline_fits(self.data_path)
        coords = assign_cartesian_coordinates(catalog, build_cosmology(), z_ref=2.5)

        self.assertIsNotNone(coords.reference_redshift)
        self.assertIsNotNone(coords.field_center_ra_deg)
        self.assertIsNotNone(coords.field_center_dec_deg)

        first = coords.sightlines[0]
        self.assertIsNotNone(first.pixel_redshift)
        self.assertIsNotNone(first.x)
        self.assertIsNotNone(first.y)
        self.assertIsNotNone(first.z)
        self.assertEqual(first.pixel_redshift.shape, first.wavelength.shape)
        self.assertEqual(first.x.shape, first.wavelength.shape)
        self.assertEqual(first.y.shape, first.wavelength.shape)
        self.assertEqual(first.z.shape, first.wavelength.shape)
        self.assertAlmostEqual(float(np.std(first.x)), 0.0)
        self.assertAlmostEqual(float(np.std(first.y)), 0.0)
        self.assertGreater(float(np.std(first.z)), 0.0)

    def test_assign_cartesian_coordinates_respects_explicit_field_center(self) -> None:
        catalog = read_sightline_fits(self.data_path)
        coords = assign_cartesian_coordinates(
            catalog,
            build_cosmology(),
            z_ref=2.5,
            field_center_ra_deg=0.64,
            field_center_dec_deg=-0.08,
        )

        self.assertAlmostEqual(coords.field_center_ra_deg, 0.64)
        self.assertAlmostEqual(coords.field_center_dec_deg, -0.08)


class TestMockGeneration(TestWithFixture):
    def test_mock_generation_is_deterministic_for_fixed_seed(self) -> None:
        catalog = read_sightline_fits(self.data_path)
        coords = assign_cartesian_coordinates(catalog, build_cosmology(), z_ref=2.5)

        wave_set_a = build_plane_wave_set(n_waves=6, k_min=0.05, k_max=0.15, seed=17)
        wave_set_b = build_plane_wave_set(n_waves=6, k_min=0.05, k_max=0.15, seed=17)
        mock_a = generate_mock_catalog(coords, wave_set_a, noise_sigma=0.2, seed=23)
        mock_b = generate_mock_catalog(coords, wave_set_b, noise_sigma=0.2, seed=23)

        np.testing.assert_allclose(mock_a.sightlines[0].delta, mock_b.sightlines[0].delta)
        np.testing.assert_allclose(mock_a.sightlines[0].ivar, mock_b.sightlines[0].ivar)

    def test_global_noise_sets_constant_ivar_for_valid_pixels(self) -> None:
        catalog = read_sightline_fits(self.data_path)
        coords = assign_cartesian_coordinates(catalog, build_cosmology(), z_ref=2.5)
        wave_set = build_plane_wave_set(n_waves=4, k_min=0.05, k_max=0.10, seed=5)
        mock_catalog = generate_mock_catalog(coords, wave_set, noise_sigma=0.25, seed=9)

        first = mock_catalog.sightlines[0]
        valid = np.isfinite(first.delta)
        self.assertTrue(np.any(valid))
        np.testing.assert_allclose(first.ivar[valid], np.float32(16.0))

    def test_zero_noise_produces_infinite_ivar_for_valid_pixels(self) -> None:
        catalog = read_sightline_fits(self.data_path)
        coords = assign_cartesian_coordinates(catalog, build_cosmology(), z_ref=2.5)
        wave_set = build_plane_wave_set(n_waves=4, k_min=0.05, k_max=0.10, seed=5)
        mock_catalog = generate_mock_catalog(coords, wave_set, noise_sigma=0.0, seed=9)

        first = mock_catalog.sightlines[0]
        valid = np.isfinite(first.delta)
        self.assertTrue(np.any(valid))
        self.assertTrue(np.all(np.isinf(first.ivar[valid])))


class TestCli(TestWithFixture):
    def test_cli_generates_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cli-output.fits"
            cli_main(
                [
                    "--input-fits",
                    str(self.data_path),
                    "--output-fits",
                    str(output_path),
                    "--z-ref",
                    "2.5",
                    "--n-waves",
                    "5",
                    "--k-min",
                    "0.05",
                    "--k-max",
                    "0.10",
                    "--noise-sigma",
                    "0.2",
                    "--seed",
                    "4",
                    "--overwrite",
                ]
            )

            self.assertTrue(output_path.exists())
            with fits.open(output_path) as hdul:
                self.assertEqual(len(hdul), 3)
                self.assertEqual(hdul[1].name, "D1-181266")
                self.assertTrue(np.all(np.isfinite(hdul[1].data["DELTA"])))
                np.testing.assert_allclose(hdul[1].data["IVAR"][:5], np.float32(25.0))
                self.assertTrue(hdul[0].header["MOCKLYA"])
                self.assertEqual(hdul[0].header["INPUTBAS"], "sample-input.fits")
                self.assertAlmostEqual(hdul[0].header["ZREF"], 2.5)
                self.assertEqual(hdul[0].header["NWAVES"], 5)
                self.assertAlmostEqual(hdul[0].header["KMIN"], 0.05)
                self.assertAlmostEqual(hdul[0].header["KMAX"], 0.10)
                self.assertAlmostEqual(hdul[0].header["NSIGMA"], 0.2)
                self.assertEqual(hdul[0].header["RNGSEED"], 4)
                self.assertIn("Generated by mock_lya_forest", "\n".join(hdul[0].header.get("HISTORY", [])))


if __name__ == "__main__":
    unittest.main()