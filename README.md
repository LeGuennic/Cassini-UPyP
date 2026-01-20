# Cassini-UPyP

Cassini-UPyP is a Python toolbox for working with Cassini/UVIS PDS3 products (.LBL/.DAT) in a research-oriented workflow.

This is not meant to be a black-box “push button, get science” package. The intent is to provide a clean, inspectable codebase that you can clone, read, and modify: if you need to tweak calibration details, masking logic, geometry choices, or binning rules for your own analysis, you should feel comfortable doing so.

An example notebook is available in example/example.ipynb.

## Design choices (read this first)

This repository relies on user-editable TOML configuration files stored in user_config/:

- user_config/env.toml
- user_config/plotting.toml

At runtime, the code reads these files relative to the repository root (detected from cassini_upyp/utils.py). Because of that, a standard non-editable installation (pip install .) is not the intended mode of use: you would end up having to edit configuration files inside site-packages, which is fragile and discourages experimentation.

Supported/recommended workflows are therefore:
- add the repository to PYTHONPATH, or
- install the repository in editable mode (pip install -e .)

## Requirements

- Python >= 3.11
- Runtime dependencies: numpy, scipy, spiceypy, tqdm
- Optional (plots / GIF export): matplotlib, pillow

## Installation / enabling imports

Option A (recommended): editable install

From the repository root:

python -m pip install -e .

With plotting extras:

python -m pip install -e ".[plot]"

Option B: add the repository to PYTHONPATH

export PYTHONPATH="/path/to/Cassini-UPyP:$PYTHONPATH"

Make sure you keep the repository layout intact (cassini_upyp/ and user_config/ must remain at the same level).

## Configuration (TOML)

Before using the package, edit user_config/env.toml to point to your local resources.

1) user_config/env.toml

This file contains local paths for resources that are not shipped with the code, typically:
- UVIS calibration files directory
- a star catalog file (stars.npy)
- SPICE kernels directory and specific kernel paths (IK, LSK)

The code expects a [paths] section and will expose its entries directly as attributes for convenience.

Minimal template (adapt to your machine):

[paths]
calibration_dir = "/path/to/calibration_files"
star_file       = "/path/to/stars.npy"

kernels_dir = "/path/to/spice_kernels"
ik_path     = "/path/to/spice_kernels/ik/cas_uvis_v07.ti"
lsk_path    = "/path/to/spice_kernels/lsk/naif0012.tls"

Notes:
- Absolute paths are recommended.
- The package does not download SPICE kernels or calibration files for you.

2) user_config/plotting.toml

This file controls plotting parameters used by geometry/plotting helpers:
- visible bodies in the field of view
- offsets
- grid definitions
- line/marker styles per object

If you do not use plotting helpers, you may not need to touch this file, but it is kept user-editable by design.

## External resources

This repository does not bundle mission resources such as:
- SPICE kernels (CK, SPK, FK, IK, LSK, etc.)
- UVIS calibration tables/files
- star catalog file used by some masking/validation utilities

You must provide those locally and set the paths in user_config/env.toml.

## Quick start

Create an observation from one or more UVIS PDS base paths (with or without extension). Each product is expected to have matching .LBL and .DAT files.

from cassini_upyp import UVIS_Observation

obs = UVIS_Observation(
    "/path/to/FUV2006_015_14_47_PRIME"
)

You can also pass multiple files:

obs = UVIS_Observation(
    "/path/to/file_1",
    "/path/to/file_2",
)

Or use a batch list file (one base path per line), using paths relative to the batch file directory or absolute paths:

obs = UVIS_Observation(batch="/path/to/batch_list.txt")

Typical processing steps depend on what you need, but the workflow generally includes:
- background estimation/removal
- radiometric calibration
- geometry (SPICE-based)

Refer to example/example.ipynb for a more complete run-through.

## Repository layout

cassini_upyp/
  uvisdata.py            Main classes (UVIS_Observation, UVIS_Bin) and I/O logic
  uvisutils.py           Spectral utilities, calibration helpers, uncertainties
  background.py          Background fit/removal helpers
  kernellib.py           SPICE/kernel handling logic
  geometry/              SPICE-based geometry and plotting helpers
  config/                Code-side defaults and instrument constants

user_config/
  env.toml               User paths (kernels, calibration files, star file)
  plotting.toml           Plotting configuration (styles, visible objects, grids)

example/
  example.ipynb          Example workflow notebook
  data/                  Small example data (if present)

## Modifying the code / contributing

If you are using this in your own research:
- you are encouraged to modify the code locally to match your scientific assumptions and workflow,
- if you fix a bug or add a generally useful feature, issues and pull requests are welcome.

This project is under active development; the API may evolve.
