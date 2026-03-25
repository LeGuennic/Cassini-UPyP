# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.4] - 2026-02-18
 
### Fixed
 
- **`UV_picture`**: Completely overhauled the pixel coordinate projection. The
  previous implementation projected pixels using tangent-point local time and
  latitude, which was geometrically incorrect for off-equator geometries. Pixels
  are now projected onto the plane of sky via perspective division of J2000 unit
  vectors, with axes defined by the body's north pole direction (retrieved from
  SPICE at each frame) and the observer–target vector.
 
### Added
 
- **`UVIS_Observation.set_geometry`**: Now stores two new attributes required by
  the updated `UV_picture` that also can be usefull for other analyses:
  - `used_pixels` — pixel corner pointing vectors in J2000 (`"XYZ"`) and
    equatorial (`"RADEC"`) coordinates, shape `(n_pics, n_pixels, 5, 3)`.
  - `target_vector` — observer-to-target position vector in J2000 (km), shape
    `(n_pics, 3)`.
 
- **`UVIS_Observation.check_stars`**: Added `vmax` parameter to cap the upper
  bound of the vmax slider range (useful when a small number of bright pixels
  would otherwise compress the colour scale for the rest of the observation).
 
### Changed
 
- **`UVIS_Observation.check_stars`**: Integrated radiance is now displayed in
  rayleighs (R) instead of kilorayleighs (kR) for better readability at typical
  airglow signal levels. The colorbar label and hover tooltip are updated
  accordingly.
 
- **`UVIS_Bin.__init__`**: Scalar attributes (`int`, `float`, `str`) are now
  copied from the parent `UVIS_Observation` via a generic loop over
  `vars(uvis_obs)`, replacing the previous explicit copies of `name`,
  `slit_width`, `slit_dlambda`, and `HD`. New scalar attributes added to
  `UVIS_Observation` in the future will be propagated to `UVIS_Bin`
  automatically.

## [1.3.3] — 2026-03-10

### Fixed
Sub-spacecraft altitude calculation: the `alt` field in `spacecraft_position` is now correctly returning the spacecraft's altitude above the ellipsoid, and not the subspacecraft point's altitude that was just zero.

## [1.3.2] — 2026-03-09

### Fixed

- **Calibration** (`uvisdata.py`): Division by zero and invalid values in the
  calibration array are now handled explicitly instead of raising warnings.

- **Geometry plotting** (`uvisdata.py`): Replaced `enumerate(self.geometry)`
  with index-based iteration using `get_geometry(ET_middle[i])` in both the GIF
  and per-file rendering paths, because now there is no precomputed geometry list.

### Changed

- **`check_stars()` — interactive UI overhaul** (`uvisdata.py`):
  - `color_scale` and `exp_range` parameters removed; their functionality is
    now exposed as real-time **interactive sliders** (vmin / vmax / exposure
    range) directly on the figure.
  - Added a **colormap selector** (◀ / ▶ cycle buttons) with nine presets
    (`gist_ncar`, `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `hot`,
    `coolwarm`, `turbo`). Default changed from `'gist_ncar'` to `'plasma'`.
  - Hover tooltip rewritten with **blitting** (`animated=True` +
    `copy_from_bbox` / `restore_region`) for significantly better rendering
    performance on large datasets.
  - Tooltip and click handlers refactored into small, focused helpers
    (`_build_hover_text`, `_cell_from_event`, `_blit_hover`, `_on_click`,
    `_on_hover`). Widget references kept on `self._check_stars_widgets` to
    prevent garbage collection.
  - Figure layout adjusted to `(10, 8)` with explicitly positioned axes for
    the heatmap, colorbar, selection panel, and slider strip.
  - Docstring updated to reflect the new interactive workflow and removed
    `color_scale` / `exp_range` from the parameter table.


## [1.3.1] – 2025-03-05

### Fixed
- `LOS_tangent`: intersection geometry fields (`lon`, `lat`, `sza`, `phase`,
  `ems`, `lt`) no longer return NaN for lines of sight that do not intersect
  the ellipsoid; they now correctly fall back to the tangent-point values.
- `plot`: sub-spacecraft longitude/latitude annotation now reads from
  `spacecraft_position['lon'/'lat']` instead of the removed `sub_sc_lon`/`sub_sc_lat` attributes.

### Documentation
- Added a warning in `plot` docstring about potential inaccuracy for
  observations with very distant targets.
  
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