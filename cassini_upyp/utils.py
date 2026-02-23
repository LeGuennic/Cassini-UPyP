from __future__ import annotations
from types import SimpleNamespace

from pathlib import Path
import tomllib
import numpy as np



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
            setattr(cfg, k, Path(v))

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