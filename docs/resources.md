# External resources
The following external resources are required and most of them are not bundled within the GitHub repository.
All resources must be referenced with the corresponding entries in the `env.toml` file (see [Configuration](configuration.md)).

## Stars catalogue

The `stars.npy` file is provided on the GitHub under the `resource` directory. This binary file is a sample
of the [Pickles and Depagne (2010)](https://doi.org/10.1086/657947) catalogue. They use stellar equatorial coordinates and proper motions from the Tycho-2 catalog, providing reliable positions and temporal drift;
UV magnitudes are derived from spectral fits to multi-band photometry, allowing an estimate of each star’s brightness in the relevant filters.
The `stars.npy` file contains a subset of the full catalogue, with only stars with a magnitude smaller than 10 in the U band, and only the following fields:

- ``tyRA``: Right ascension [°].
- ``tyDE``: Declination [°].
- ``pmRA``: Proper motion in RA [mas/yr].
- ``pmDE``: Proper motion in Dec [mas/yr].

- ``Bt`` : Tycho-2 B_T magnitude.
- ``eBt``: Uncertainty on B_T.
- ``fBt``: Spectrally matched / fitted B_T magnitude.
- ``U``  : Spectrally matched / fitted U magnitude.

This dataset is stored as a NumPy structured array, and can be loaded with the {py:func}`cassini_upyp.geometry.geometry.stars_pickles` function.
The array also stores the following fields, that are not part of the original catalogue but are computed for a given observation epoch during the geometric computations:

- ``RA_cor``  : Proper-motion corrected RA at the observation epoch [°].
- ``DEC_cor`` : Proper-motion corrected Dec at the observation epoch [°].
- ``XYZ``     : Unit direction vector in the J2000 frame.

## UVIS calibration files

UVIS calibration files must be available locally and grouped in a single directory.

The required files are:
- **Laboratory calibration**: `EUV_1999_Lab_Cal.dat` and `FUV_1999_Lab_Cal.dat`

    Those contains the full-slit, low-resolution monochromatic
    extended-source sensitivity measured in the laboratory (1997,
    updated 1999), in units of (counts s⁻¹ ) / (kilorayleigh).

- **Calibration time dependance**: `uvis_calibration_trending_v01_data.sav`

- **Flat-field**: `FLATFIELD_*UV_POSTBURN.txt` and `FLATFIELD_*UV_PREBURN.txt`
    
    There should be four files, two for each UVIS channel. The two files relates to the flat field before and after the starburn event that occured on 6 June 2002.

- **Flat-field time dependance**: `*{channel}*ff_modifier*.dat`

    Those are the flat-field modifier files. The last 10 characters of the file name
    should correspond to the spacecraft time.

Calibration files are available on the [PDS website page of UVIS](https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Cassini/inst-uvis.html#analyzing) in the corresponding [directory](https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Cassini/CASSINIUVIS/UVIS-16/Cube%20Generator%20Software/UVIS_cal_and_ff_modifiers/).


## SPICE kernels (geometry)

SPICE kernels are required to compute observation geometry.
Two usage patterns are supported.

### Full kernel tree (automatic selection)

Users may maintain a local SPICE kernel repository with the following directory structure:
```
spice_kernels/
├─ ck/
├─ fk/
├─ ik/
├─ lsk/
├─ mk/
├─ pck/
├─ sclk/
└─ spk/
```

When this layout is provided and kernels_dir is set accordingly, the geometry code can automatically search for and load appropriate kernels based on observation time and context, provided that the required kernels are available.

### Manual kernel list

Alternatively, users may explicitly provide the list of SPICE kernels to be loaded when calling geometry-related routines in the form of a single metakernel (`.mk`) file.

In this mode, kernel selection is fully controlled by the user, and the code will not attempt to discover kernels automatically.