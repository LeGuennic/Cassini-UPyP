from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tomllib

from typing import List, Tuple, Literal
import numpy as np

# DATA BINS UTILITIES -----
def list_ndarray(bin_boundaries: Tuple[List[float], ...]) -> np.ndarray:
    """
    Create a NumPy array (dtype=object) where each cell is initialized as an empty list.

    The shape of the array is determined by the number of bins in each dimension,
    i.e., (len(boundaries)-1, ...).

    Parameters
    ----------
    bin_boundaries : Tuple[List[float], ...]
        A tuple of lists defining the bin edges for each property.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (len(bin_boundaries[0])-1, len(bin_boundaries[1])-1, ...)
        where each cell is an empty list.
    """

    shape = tuple(len(bounds) - 1 for bounds in bin_boundaries)
    bins_array = np.empty(shape, dtype=object)
    
    # Initialize each cell with an empty list.
    it = np.nditer(bins_array, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
    while not it.finished:
        bins_array[it.multi_index] = []
        it.iternext()
    return bins_array

def find_bin_index(prop, boundaries, mode: Literal['center', 'all']) -> int:
    """
    Determine the bin index for a given property value (or array of values) relative to the provided boundaries.

    In 'center' mode, 'prop' is expected to be a scalar value.
    In 'all' mode, 'prop' is expected to be an array; all values must fall within the same bin.

    Parameters
    ----------
    prop : scalar or array-like
        The property value(s) for which to determine the bin index.
    boundaries : List[float]
        A sorted list of bin edges.
    mode : Literal['center', 'all']
        The mode of operation:
            - 'center': use a single representative value.
            - 'all': require that all values in the pixel fall within the same bin.

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
def read_toml(path: str | Path):
    """
    Read a TOML file and return a module-like object:
    - top-level keys -> attributes
    - nested dicts remain dicts (matplotlib-friendly)
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    for key, value in data.items():
        if isinstance(value, dict):
            continue
        elif isinstance(value, list):
            data[key] = np.array(value)
        else:
            data[key] = value

    return SimpleNamespace(**data)


def _repo_root() -> Path:
    """Return the root of the Cassini_UPyP repo."""
    # __file__ = .../cassini_upyp/utils.py
    # parents[0] = .../cassini_upyp
    # parents[1] = .../Cassini_UPyP  (repo root)
    return Path(__file__).resolve().parents[1]


def env_config():
    """Return the env.toml config as a module-like object (flattened)."""
    cfg_path = _repo_root() / "user_config" / "env.toml"
    cfg = read_toml(cfg_path)

    # Flatten [paths] section for backward-compatible attributes
    if hasattr(cfg, "paths") and isinstance(cfg.paths, dict):
        for k, v in cfg.paths.items():
            setattr(cfg, k, v)

    return cfg

def plot_config():
    """Return the plot.toml config as a module-like object."""
    cfg_path = _repo_root() / "user_config" / "plotting.toml"
    return read_toml(cfg_path)