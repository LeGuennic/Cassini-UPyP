# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] - 2026-02-18

### Added
- `CHANGELOG.md`: initial changelog covering all versions from pre-versioning to 1.3.0
- `LOS_tangent` now returns both **tangent point** and **surface intersection** geometry — new fields: `t_lon`, `t_lat`, `t_sza`, `t_phase`, `t_ems`, `t_lt`
- `spacecraft_position` attribute on `UVIS_Observation`, replacing `sub_sc_point` — structured array with full geometry fields (`lon`, `lat`, `alt`, `sza`, `phase`, `ems`, `lt`)

### Changed
- `UV_picture`: projection axis now based on **local time** instead of longitude; x-axis orientation derived from spacecraft local time
- `UV_picture`: pixel coordinates use tangent-point fields (`t_lat`, `t_lon`, `t_lt`) instead of `lat`/`lon`
- Titan-specific longitude flip removed from `UV_picture` — now handled uniformly in `LOS_tangent`
- Radii indexing corrected: `r_e` is now the equatorial mean, `r_p` the polar radius
- `set_geometry` no longer stores the full `Geometry` object list; data is unpacked immediately into dedicated attributes — fixing huge memory usage
- Local solar time formula sign corrected (`12 + delta` instead of `12 - delta`)

### Removed
- `sub_sc_point` attribute (superseded by `spacecraft_position`)
- Longitude annotation text overlay in `UV_picture`
- `fullsave` parameter from `UVIS_Observation.save()`, geometry is not saved anymore

---

## [1.2.0] - 2026-03-03

### Added
- `UV_picture()`: new function to build and render a projected UV radiance image from pixel footprint geometry (longitude/latitude/altitude projection, contour or imshow rendering, optional annotation)
- Cyclic dimension support in automatic bin creation (e.g. longitude, local time)
- Updated example notebook

---

## [1.0.1] - 2026-02-27

### Added
- `sub_sc_point` attribute on `UVIS_Observation` storing sub-spacecraft longitude and latitude at each exposure

---

## [1.0.0] - 2026-02-18

### Added
- `CITATION.cff` and `LICENSE` files

---

## [0.1.0] - 2026-01-05 / 2026-02-16

This is the initial versioned release, covering all development from the first commit (May 2025) through the public release refactor (January 2026) up to the final fixes before v1.0.0.

### Added
- `cassini_upyp` installable package (`pyproject.toml`) with explicit dependencies (`numpy`, `scipy`, `tqdm`, `spiceypy`) and optional `plot` / `dev` extras
- `cassini_upyp/geometry/` subpackage:
  - `spice_engine.py`: SPICE-based geometry engine (`Geometer` class, `LOS_tangent`)
  - `geometry.py`: high-level `Geometry` class
  - `computational.py`: low-level geometric routines (ellipsoid intersections, projections, angles)
  - `plot.py`: geometry plotting
- `UVIS_Observation` and `UVIS_Bin` classes with lazy-import public API
- `kernellib.py`: SPICE kernel selection and metakernel builder, rewritten with `pathlib`
- `utils.py`: TOML config loader (`read_toml`, `env_config`, `plot_config`)
- Configuration converted from Python files to TOML: `config/env.toml`, `user_config/plotting.toml`
- `poisson_error()`: exact Garwood confidence intervals for Poisson counts
- `correction_factor()`: small-sample bias correction using log-gamma
- `UVIS_Bin.save()` / `UVIS_Bin.load()` methods; `UVIS_Bin.__repr__()`
- `UVIS_Bin.integrate()`: radiance integration over a wavelength range with per-bin statistics
- `bin_LOS`, `pixel_LOS`, `slit_width`, `HD` attributes on `UVIS_Bin`
- `stars.npy` star catalogue resource
- Example notebook and data files (`example/`)
- Full Sphinx documentation (`docs/`) with API reference, installation, configuration, and resources pages
- GitHub Actions workflow for automated documentation deployment
- Background estimation refactored: Monte Carlo simulation replaced by analytical gap-histogram model (`gap_histogram`)

### Changed
- Package restructured: monolithic `lib/` layout replaced by flat top-level `cassini_upyp/` modules
- Monolithic `geometry.py` split into the `geometry/` subpackage
- `UVIS_bin` renamed to `UVIS_Bin`; `__init__` signature changed to take a `shape` tuple
- `UVIS_Bin.average_bins()` renamed to `UVIS_Bin.average()`
- `list_ndarray()` refactored to take a shape tuple instead of bin boundary lists
- `.uvis` / `.uvisbin` file extensions replace `.pkl` for saved objects
- Poisson error computation replaced by exact Garwood intervals
- Phase and emission angle sign corrected in `LOS_tangent`
- `Instrument` class now takes instrument name instead of SPICE ID
- `env_config()` now returns path values as `Path` objects
- Smoothing restricted to FUV channel only
- `spice.furnsh` / `spice.unload` calls use `str()` on Path objects for SPICE compatibility
- Matplotlib imports made lazy throughout

### Fixed
- `bin_LOS` initialized with NaN values instead of zeros
- `pixel_stars_mask` index order corrected in `add_pixel_stars_from_file`
- `UVIS_Bin` name attribute assignment (hotfix)

---

## Pre-versioning — 2025-05-13 / 2025-10-24

Development before versioning was introduced. Not tagged.

### Notable work
- Initial prototype: `UVIS_main.py`, `UVIS_background.py`, `UVIS_geometry.py`, `kernellib.py`
- PDS file reading, calibration pipeline, background fitting, spectral binning, SPICE geometry
- Iterative fixes to `UVIS_bin`, background fitting, NaN handling, import paths, file suffixes
- `poisson_error()` and `correction_factor()` introduced (pre-v0.1.0)