# Installation

Download or clone the repository from GitHub:

[https://github.com/LeGuennic/Cassini-UPyP.git](https://github.com/LeGuennic/Cassini-UPyP.git)

The main code is under the `cassini_upyp` directory, make sure to also download the `user_config` directory.
The `example` folder only provides some data and a simple script.

Once downloaded, either run your Python script within the repository or proceed to installation using one of the two methods.

## PYTHONPATH

You may add the repository root to PYTHONPATH, depending on your OS and Python installation.

For example, on linux:
`export PYTHONPATH="/path/to/Cassini-UPyP:$PYTHONPATH"`


## PIP editable install
Make sure to download the `pyproject.toml` configuration file.

From the repository root:

`python -m pip install -e .`

Optional dependencies (recommended):

`python -m pip install -e ".[plot][qt]"`
