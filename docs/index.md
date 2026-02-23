# Cassini-UPyP

The **Cassini UVIS Python Package** is a research-oriented Python toolbox for reading, processing and analysing data from the [UltraViolet Imaging Spectrograph (UVIS)](https://lasp.colorado.edu/cassini/) on board the Cassini spacecraft.

The package aims to provide tools to work with UVIS data, display informations, compute geometry, plot spectra and related products and prepare such data to further processing if needed like forward modelling.
The package is not designed as a black-box tool. The code is meant to be clean, inspectable, and modifiable: if you need to tweak methods or routines for your own analysis, you should feel comfortable doing so.

## Main classes
- [UVIS_Observation](api/uvisdata_UVIS_Observation)
- [UVIS_Bin](api/uvisdata_UVIS_Bin)

## Documentation
```{toctree}
:maxdepth: 1

installation
resources
configuration

api/index
```