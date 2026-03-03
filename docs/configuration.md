# Configuration

Cassini-UPyP relies on user-editable configuration files stored in the user_config/ directory.
These files define machine-specific paths and plotting preferences and must be adapted locally after cloning or downloading the repository.


## user_config/env.toml

The file user_config/env.toml defines paths to external resources required at runtime.
It is shipped with the repository as a template and must be edited by the user to match their local environment.

- `calibration_dir`: path to the calibration files directory
- `star_file`: path to the `stars.npy` file, provided in the `resources` directory on the GitHub.
- `kernels_dir`: path to the SPICE kernels directory
- `ik_path`: path to a Instrument Kernel (IK) `.ti` file, for example `cas_uvis_v07.ti`
- `lsk_path`: path to a Leapsecond Kernel (LSK) `.tls` file, for example `naif0012.tls`
- `pck_path`: path to a Planetary Constants Kernel (PCK) `.tpc` file, for example `pck00010.tpc`

Absolute paths are recommended to avoid ambiguity.

## user_config/plotting.toml

The file user_config/plotting.toml controls parameters used by geometry and plotting helper functions, such as colours, markers and other cosmetic and style objects.

If plotting helpers are not used, this file may remain unchanged.