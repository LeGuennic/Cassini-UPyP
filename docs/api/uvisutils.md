# Data utilities (cassini_upyp.uvisutils)

This module contains the main data utilities to load and process Cassini/UVIS PDS3 products used in [cassini_upyp.uvisdata](uvisdata.md) module.


## Uncertainty utilities
```{eval-rst}

.. autofunction:: cassini_upyp.uvisutils.poisson_error

.. autofunction:: cassini_upyp.uvisutils.correction_factor
```

## Spectrum related utilities
```{eval-rst}

.. autofunction:: cassini_upyp.uvisutils.UVIS_WL

.. autofunction:: cassini_upyp.uvisutils.integrate_spectrum

.. autofunction:: cassini_upyp.uvisutils.interpolate_nans

.. autofunction:: cassini_upyp.uvisutils.smooth_spectrum
```



## Calibration utilities and I/O routines
```{eval-rst}

.. autofunction:: cassini_upyp.uvisutils.uvis_lab_calibration

.. autofunction:: cassini_upyp.uvisutils.get_cal_time_variation

.. autofunction:: cassini_upyp.uvisutils.get_ff_time_variation

.. autofunction:: cassini_upyp.uvisutils.read_spica_ff
```

## Data binning
```{eval-rst}

.. autofunction:: cassini_upyp.uvisutils.list_ndarray

.. autofunction:: cassini_upyp.uvisutils.find_bin_index
```