# Cassini-UPyP - The UVIS Python Package
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://leguennic.github.io/Cassini-UPyP/)

The **Cassini UVIS Python Package** is a research-oriented Python toolbox for reading, processing and analysing data from the [UltraViolet Imaging Spectrograph (UVIS)](https://lasp.colorado.edu/cassini/) on board the Cassini spacecraft.

The package aims to provide tools to work with UVIS data, display informations, compute geometry, plot spectra and related products and prepare such data to further processing if needed like forward modelling.
It is not designed as a black-box tool. The code is meant to be clean, inspectable, and modifiable: if you need to tweak methods or routines for your own analysis, you should feel comfortable doing so.

An example notebook is available in [`example/example.ipynb`](https://github.com/LeGuennic/Cassini-UPyP/blob/main/example/example.ipynb).

Hereafter a short description. Refer to the [**documentation**](https://leguennic.github.io/Cassini-UPyP/) for more details on installation, configuration, and usage:

## Requirements

- Python >= 3.11
- Runtime dependencies: [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [SpiceyPy](https://pypi.org/project/spiceypy/), [tqdm](https://pypi.org/project/tqdm/)
- Optional (plots / GIF export): [Matplotlib](https://matplotlib.org/), [Pillow](https://pypi.org/project/pillow/)


## Repository layout

Those are the files and directories that are useful. The `cassini_upyp` directory contains the main code, while `user_config` contains user-editable configuration files.
The `example` folder provides a simple workflow notebook and some example data.

```
cassini_upyp/
├─ uvisdata.py            Main classes (UVIS_Observation, UVIS_Bin) and I/O logic
├─ uvisutils.py           Spectral utilities, calibration helpers, uncertainties
├─ background.py          Background noise fit/calculation helpers
├─ kernellib.py           SPICE/kernel handling logic
├─ geometry/              SPICE-based geometry and plotting routine
└─ config/                Code-side defaults and instrument constants

user_config/
├─ env.toml               User paths (kernels, calibration files, star file)
└─ plotting.toml          Plotting configuration (styles, visible objects, grids)

example/
├─ example.ipynb          Example workflow notebook
└─ data/                  Small example data
```


## Installation

Once you downloaded or cloned the repository, either run your Python script within the repository or proceed to installation.
There are two supported installation methods, either add the repository to PYTHONPATH or install it in editable mode using pip.

Make sure you keep the repository layout intact (`cassini_upyp/` and `user_config/` must remain at the same level).

## Configuration

This repository ships with user-editable TOML configuration files stored in `user_config/`:

- `user_config/env.toml`
- `user_config/plotting.toml`

After downloading or cloning the repository, you must edit `user_config/env.toml` to point to the external resources available on your machine (SPICE kernels, calibration files, star catalogue).

### Environment

The `user_config/env.toml` file defines local paths for resources that are not bundled with the code, typically:
- UVIS calibration files directory
- a star catalogue file (stars.npy)
- SPICE kernels directory and specific kernel paths (IK, LSK)

## Required external resources

This code requires a small set of external resources that are not bundled with the repository. They must be referenced from `user_config/env.toml`.

### 1) UVIS calibration files
You must provide the UVIS calibration files directory. These files are required for calibration and related flat field correction steps. Set the corresponding path in `user_config/env.toml:calibration_dir`.

### 2) SPICE kernels (geometry)
SPICE kernels are required to compute observation geometry. You can use either of the two supported approaches:

A) Full kernel tree (automatic kernel selection)
Place your locally downloaded kernels using the following subdirectory layout:
```
spice_kernels/
├─ ck/
├─ fk/
├─ ik/
├─ lsk/
├─ mk/
├─ pck/
├─ sclk/
└─ spk/
```

With this layout, the geometry code can automatically search for and load the appropriate kernels (depending on observation time, frames, and target).

B) Manual kernel list (explicit kernel loading)
If you prefer not to maintain a full kernel tree, you may explicitly provide the list of required kernels to the geometry routines, or a metakernel file. The geometry layer supports passing a manual kernel list (for example: LSK + SCLK + one or more SPK/CK/FK/IK/PCK as needed). In this mode, kernel selection is fully controlled by the user and the code will not attempt to discover kernels automatically.

In both cases, you must at minimum provide a valid LSK and SCLK, and ensure that the UVIS IK is available if instrument frame definitions are needed. The exact CK/SPK/PCK/FK requirements depend on the specific geometry you want to compute and the time span of the observation.


## Quick start

Create an observation from one or more UVIS PDS base paths (with or without extension). Each product is expected to have matching .LBL and .DAT files.

```python
from cassini_upyp import UVIS_Observation

obs = UVIS_Observation(
    "/path/to/FUV2006_015_14_47_PRIME"
)
```

You can also pass multiple files:

```python
obs = UVIS_Observation(
    "/path/to/file_1",
    "/path/to/file_2",
)
```

Or use a batch list file (one base path per line), using paths relative to the batch file directory or absolute paths:

```python
obs = UVIS_Observation(batch="/path/to/batch_list.txt")
```

Typical processing steps depend on what you need, but the workflow generally includes:
- calibration
- background noise estimation and/or subtraction
- geometry (SPICE-based)

Refer to [`example/example.ipynb`](https://github.com/LeGuennic/Cassini-UPyP/blob/main/example/example.ipynb) for a more complete run-through.



## Modifying the code

If you are using this in your own research, you are encouraged to modify the code locally to match your scientific assumptions and workflow.

Note that this project is under active development and may evolve with time.


## Notice on generative AI

AI LLM (Large Language Model) tools were used during development to help draft and edit documentation (docstrings, README, commits, ...), to suggest refactoring and optimizations for parts of the codebase, and to improve spelling and style. All changes (including this paragraph) were reviewed, modified, corrected, adapted and integrated by the author, who remains responsible for the final content and implementation.
