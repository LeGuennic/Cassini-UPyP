from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from numpy.typing import ArrayLike
import tomllib

from typing import Literal, Sequence
import numpy as np

# DATA BINS UTILITIES -----
def list_ndarray(bin_boundaries: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Create a NumPy array (dtype=object) where each cell is initialized as an independant empty list.

    The shape of the array is determined by the number of bins in each dimension,
    i.e., (len(boundaries)-1, ...).

    Parameters
    ----------
    bin_boundaries : sequence of sequences of float
        Bin edges for each dimension. For example, two arrays of edges
        will produce a 2D array of shape
        (len(bin_boundaries[0]) - 1, len(bin_boundaries[1]) - 1).

    Returns
    -------
    np.ndarray
        A NumPy array of shape (len(bin_boundaries[0])-1, len(bin_boundaries[1])-1, ...)
        where each cell is an empty Python list.

    Examples
    --------
    Create a 2D array of lists for two properties:

    >>> from cassini_upyp.utils import list_ndarray
    >>> bin_edges = ([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    >>> bins = list_ndarray(bin_edges)
    >>> bins.shape
    (3, 2)

    Append data points to a given bin:

    >>> bins[0, 0].append(42.0)
    >>> bins[0, 0]
    [42.0]
    """

    shape = tuple(len(bounds) - 1 for bounds in bin_boundaries)
    bins_array = np.empty(shape, dtype=object)
    
    # Initialize each cell with an empty list.
    for index in np.ndindex(shape):
        bins_array[index] = []
    return bins_array

def find_bin_index(prop: float | ArrayLike, boundaries: Sequence[float], mode: Literal['center', 'all'] = "center") -> int | None:
    """
    Determine the bin index for a given property value (or array of values) relative to the provided boundaries.

    In 'center' mode, 'prop' is expected to be a scalar value.
    In 'all' mode, 'prop' is expected to be an array; all values must fall within the same bin.

    The binning convention is half-open: [edges[i], edges[i+1]),
    so the last edge is excluded.

    Parameters
    ----------
    prop : scalar or array-like
        The property value(s) for which to determine the bin index.
    boundaries : sequence of float
        A sorted list of bin edges.
    mode : {"center", "all"}, optional
        The mode of operation:
        - 'center': use a single representative value.
        - 'all': require that all values in the pixel fall within the same bin.
        Default is "center".

    Returns
    -------
    int or None
        The bin index if valid; otherwise, None.
    """
    
    edges = np.array(boundaries)

    if mode == 'center':
        # Check that the value is within the interval [edges[0], edges[-1])
        if prop < edges[0] or prop >= edges[-1]:
            return None
        idx = int(np.searchsorted(edges, prop, side='right') - 1)
        return idx
    elif mode == 'all':
        arr = np.asarray(prop)
        if arr.size == 0 or np.min(arr) < edges[0] or np.max(arr) >= edges[-1]:
            return None
        indices = np.searchsorted(edges, arr, side='right') - 1
        
        # All values must fall into the same bin.
        if np.all(indices == indices[0]):
            return int(indices[0])
        else:
            return None



# CONFIGURATION UTILITIES -----
def read_toml(path: str | Path) -> SimpleNamespace:
    """
    Read a TOML config file and expose top-level keys as attributes.

    The file is parsed with ``tomllib`` and converted as follows:
    - top-level scalar values become attributes of the returned object,
    - top-level lists are converted to NumPy arrays,
    - top-level dicts are kept as dicts (e.g. for matplotlib rcParams).

    Nested structures inside those dicts are left unchanged.

    Parameters
    ----------
    path : str or Path
        Path to the TOML configuration file.

    Returns
    -------
    types.SimpleNamespace
        Object whose attributes correspond to the top-level keys of the
        TOML file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    tomllib.TOMLDecodeError
        If the file is not valid TOML.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    for key, value in data.items():
        if isinstance(value, dict):
            # keep nested dicts as-is
            continue
        elif isinstance(value, list):
            data[key] = np.array(value)
        else:
            data[key] = value

    return SimpleNamespace(**data)


def _repo_root() -> Path:
    """
    Return the root directory of the Cassini_UPyP repository.

    This assumes the standard layout where this file lives in
    ``cassini_upyp/utils.py`` and the repo root is the parent directory
    of the ``cassini_upyp`` package.
    """
    # __file__ = .../cassini_upyp/utils.py
    # parents[0] = .../cassini_upyp
    # parents[1] = .../Cassini_UPyP  (repo root)
    return Path(__file__).resolve().parents[1]


def env_config() -> SimpleNamespace:
    """
    Load the env.toml configuration as a module-like object.

    The file ``user_config/env.toml`` is read from the repository root
    (as determined by ``_repo_root()``) using ``read_toml``. Top-level
    keys become attributes of the returned object. In addition, the
    ``[paths]`` section, if present, is flattened so that entries like

        [paths]
        calibration_dir = "..."

    are also accessible as ``cfg.calibration_dir`` for backward
    compatibility.

    Returns
    -------
    types.SimpleNamespace
        Configuration object with attributes corresponding to the
        top-level keys of env.toml, plus flattened entries from
        the [paths] section.
    """

    cfg_path = _repo_root() / "user_config" / "env.toml"
    cfg = read_toml(cfg_path)

    # Flatten [paths] section for backward-compatible attributes
    if hasattr(cfg, "paths") and isinstance(cfg.paths, dict):
        for k, v in cfg.paths.items():
            setattr(cfg, k, v)

    return cfg

def plot_config() -> SimpleNamespace:
    """
    Load the plotting configuration from plotting.toml.

    The file ``user_config/plotting.toml`` is read from the repository
    root (as determined by ``_repo_root()``) using ``read_toml``.
    Top-level keys become attributes of the returned object.

    Returns
    -------
    types.SimpleNamespace
        Plotting configuration object.
    """
    cfg_path = _repo_root() / "user_config" / "plotting.toml"
    return read_toml(cfg_path)