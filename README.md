# mock-lya-forest

Small Python package for generating mock 3D Lyman-alpha forest transmission fluctuations from superpositions of real plane waves, using LATIS-style per-sightline FITS inputs and outputs.

This repository currently contains:

- `mock_lya_forest/`: the package implementation
- `tests/`: regression tests for I/O, geometry, mock generation, and CLI behavior
- `pyproject.toml`: package metadata and console-script entry point
- `LICENSE`: MIT license for the code in this repository

## Features

- Reads LATIS-style multi-extension sightline FITS files with one binary table HDU per quasar sightline
- Preserves the same output FITS structure on write
- Converts observed wavelength to absorber redshift using a fiducial cosmology
- Rotates sightlines to a field center and uses a plane-parallel local Cartesian coordinate system
- Uses relative comoving coordinates for all three axes, including the line-of-sight axis
- Generates real-valued plane-wave fluctuations with random 3D `k` vectors and scalar phases
- Adds one user-controlled global Gaussian noise level to every valid pixel
- Writes self-describing provenance metadata into the output primary FITS header

## Installation

The code was developed in a conda environment named `lya` with `numpy` and `astropy` available.

For local editable installation:

```bash
python -m pip install -e .
```

## Command-Line Usage

The package can be run directly as a module:

```bash
python -m mock_lya_forest \
  --input-fits path/to/input.fits \
  --output-fits mock-output.fits \
  --z-ref 2.5 \
  --n-waves 100 \
  --k-min 0.05 \
  --k-max 0.15 \
  --noise-sigma 0.2 \
  --seed 7 \
  --overwrite
```

Or, after installation, through the console script:

```bash
mock-lya-forest --help
```

### Main Arguments

- `--input-fits`: input multi-extension sightline FITS file
- `--output-fits`: output FITS path
- `--z-ref`: reference redshift that defines the relative line-of-sight coordinate
- `--n-waves`: number of plane waves to superimpose
- `--k-min`, `--k-max`: wave-number range in `h/cMpc`
- `--noise-sigma`: global Gaussian noise standard deviation for `DELTA`
- `--field-center-ra`, `--field-center-dec`: optional explicit field center; defaults to the mean sightline position
- `--seed`: RNG seed used for both the wave draw and the noise realization

## Output Format

The output keeps the same layout as the input example:

- Primary HDU for top-level metadata
- One binary table HDU per sightline
- Per-sightline columns: `LAMBDA`, `DELTA`, `IVAR`, `CONT`

The primary header also stores provenance keys such as:

- `MOCKLYA`
- `INPUTBAS`
- `ZREF`
- `NWAVES`
- `KMIN`, `KMAX`
- `NSIGMA`
- `RNGSEED`
- `COSMOH0`, `COSMOOM0`, `TCMB0`
- `FCRA`, `FCDEC`

## Testing

Run the regression suite with:

```bash
python -m unittest discover -s tests -v
```

Current coverage includes:

- FITS read/write round-trip behavior
- coordinate assignment and explicit field-center override
- deterministic mock generation with fixed seeds
- zero-noise and constant-`IVAR` behavior
- CLI end-to-end output generation

## Repository Notes

- The regression suite is self-contained and generates its own small FITS fixture at runtime.
- Generated outputs such as `mock-output.fits` are ignored by `.gitignore` and should not be committed.
- Editor, virtual environment, and Python cache files are also ignored.