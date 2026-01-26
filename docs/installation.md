# Installation

Cassini-UPyP is intended to be used from a cloned repository.

Supported workflows:

editable installation using pip

adding the repository to PYTHONPATH

## Editable install (recommended)

From the repository root:

`python -m pip install -e .`

Optional plotting dependencies:

`python -m pip install -e ".[plot]"`

## PYTHONPATH-based usage

Alternatively, you may add the repository root to PYTHONPATH:

`export PYTHONPATH="/path/to/Cassini-UPyP:$PYTHONPATH"`