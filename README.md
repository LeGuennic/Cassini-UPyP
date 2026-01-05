# Cassini-UPyP

Personal Python utilities for working with Cassini UVIS data (UPyP workflow).

## Installation

From source:
```bash
pip install .
```

Optional plotting support:
```bash
pip install ".[plot]"
```

## Usage

```python
from cassini_upyp import UVIS_Observation, UVIS_Bin
```

## Notes

- NAIF/SPICE kernels are required for some features but are not bundled.
- Plotting features require the optional `plot` dependencies.
- This package is under active development; the API may change.
