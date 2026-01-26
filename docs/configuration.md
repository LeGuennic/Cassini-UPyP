# Configuration

Cassini-UPyP relies on user-editable configuration files stored in the user_config/ directory.
These files define machine-specific paths and plotting preferences and must be adapted locally after cloning or downloading the repository.

## user_config/env.toml

The file user_config/env.toml defines paths to external resources required at runtime.
It is shipped with the repository as a template and must be edited by the user to match their local environment.

This file is expected to contain a [paths] section.

Typical entries include:

the directory containing UVIS calibration files

the star catalogue file used by some validation and masking utilities

the directory containing SPICE kernels

explicit paths to mandatory SPICE kernels (IK and LSK)

Minimal example (paths must be adapted):

`[paths]`
`calibration_dir = "path/to/calibration_files"`
`star_file = "path/to/stars.npy"`

`kernels_dir = "path/to/spice_kernels"`
`ik_path = "path/to/spice_kernels/ik/cas_uvis_v07.ti"`
`lsk_path = "path/to/spice_kernels/lsk/naif0012.tls"`

Absolute paths are recommended to avoid ambiguity.

## External resources

The following external resources are required and are not bundled with the code.

### UVIS calibration files

UVIS calibration files must be available locally and grouped in a single directory.
This directory must be referenced by the calibration_dir entry in env.toml.

These files are required for radiometric calibration and related correction steps.

### SPICE kernels (geometry)

SPICE kernels are required to compute observation geometry.
Two usage patterns are supported.

#### Full kernel tree (automatic selection)

Users may maintain a local SPICE kernel repository with the following directory structure:

spice_kernels/
ck/
fk/
ik/
lsk/
mk/
pck/
sclk/
spk/

When this layout is provided and kernels_dir is set accordingly, the geometry code can automatically search for and load appropriate kernels based on observation time and context.

#### Manual kernel list

Alternatively, users may explicitly provide the list of SPICE kernels to be loaded when calling geometry-related routines.

In this mode, kernel selection is fully controlled by the user, and the code will not attempt to discover kernels automatically.

At minimum, a valid LSK and SCLK must be provided.
The UVIS instrument kernel (IK) is required when instrument frame definitions are needed.
Additional SPK, CK, PCK, or FK files may be required depending on the geometry being computed.

## user_config/plotting.toml

The file user_config/plotting.toml controls parameters used by geometry and plotting helper functions.
It typically includes definitions for visible bodies, grids, offsets, and style options.

If plotting helpers are not used, this file may remain unchanged.