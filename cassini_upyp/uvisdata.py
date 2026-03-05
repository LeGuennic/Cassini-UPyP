from __future__ import annotations
from typing import Literal, Iterable
from numpy.typing import ArrayLike
from collections.abc import Sequence

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import spiceypy as spice
import ast
import io
import math
import pickle
import json
import warnings

from scipy.interpolate import PchipInterpolator

# MODULES
from .uvisutils import (
    # Calibration
    uvis_lab_calibration,
    get_ff_time_variation,
    get_cal_time_variation,
    read_spica_ff,

    # Spectrum
    UVIS_WL,
    interpolate_nans,
    integrate_spectrum,
    smooth_spectrum,

    # Uncertainty
    poisson_error,
    correction_factor,

    list_ndarray, find_bin_index
)

# CONFIG
from .config.pipeline_defaults import *
from .config.uvis import (
    pixel_bandpasses,
    slit_ratios,
    sctimeburn,
    slit_dlambda,
    slit_width
)
from .utils import env_config
env = env_config()

def pds_lbl(labelfile: str | Path) -> "AttrDict":
    """
    Parse a PDS3-formatted LBL (Label) file and return its contents as a nested dictionary.

    Parameters
    ----------
    labelfile : str or pathlib.Path
        Path to the PDS3 .LBL file.

    Returns
    -------
    AttrDict
        Nested dictionary-like object with attribute access. Top-level
        keys are the main label fields; nested OBJECT blocks are stored
        as sub-AttrDicts.

    Examples
    --------
    Load a label and access a top-level field and a nested object:

    >>> lbl = pds_lbl("path/to/label.lbl")
    >>> lbl.TARGET_NAME
    'TITAN'
    >>> lbl.QUBE.AXES
    3

    Notes
    -----
    - Unit strings in angle brackets (e.g. "<KM>", "<DEGREE>") are
      preserved as part of the string values.
    - Only a simple OBJECT / END_OBJECT nesting model is supported; this
      is sufficient for the UVIS labels used in this package.
    """

    label  = AttrDict({})
    obj    = label
    nested = False

    labelfile = Path(labelfile).expanduser()
    with labelfile.open("r") as f:
        for line in f:
            line = line.strip()
            if line.strip() =='END' : break

            # ^ at the begining indicates a nested object
            if line.startswith('^'):
                nested = True
            
            elif '=' in line and not line.startswith('^'):

                key, value = line.split('=', 1)
                key   = key.strip()
                value = value.strip(' "')

                # Convert UNK and N/A into None
                if 'UNK' in value or 'N/A' in value:
                    value = None
                # Convert what we can in numbers
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    try:
                        value = float(value)
                    except (ValueError, TypeError): pass
                

                if key == "OBJECT" and nested:
                    obj[value]=AttrDict({})
                    obj = obj[value]
                elif key == "END_OBJECT":
                    nested=False
                    obj=label
                else : obj[key]  = value


            elif "=" not in line and not line.startswith('^') :
                # Continuation of the previous key's value (multi-line field)
                if obj[key] == '' : obj[key] += line.strip('"')
                else : obj[key] += ' '+line.strip('"')

    return AttrDict(label)


def pds_dat(filename_dat: str | Path, data_dims: tuple[int, int, int], data_type: str | np.dtype, endian: Literal['big', 'little'] = 'big') -> np.ndarray:
    """
    Read a binary PDS (Planetary Data System) data file and return its contents as a NumPy array.

    This function reads binary data from a PDS file and reshapes it into a data cube based on the 
    specified dimensions. It handles the byte order (endianness) and data type to correctly 
    interpret the binary data.

    Parameters
    ----------
    filename_dat : str or pathlib.Path
        Path to the binary PDS data file.
    data_dims : tuple of int
        Dimensions of the data in the order (BAND, LINE, SAMPLE).

        - BAND   : Number of spectral pixels
        - LINE   : Number of spatial pixels  
        - SAMPLE : Number of exposures
        
    data_type : str or numpy.dtype
        Data type of the binary data (e.g. "float32", "int16").
    endian : {"big", "little"}, optional
        Byte order of the data. Default is "big".

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the data reshaped into a cube with dimensions 
        (SAMPLE, LINE, BAND).

    Notes
    -----
    The dimensions should be provided in the order (BAND, LINE, SAMPLE), but the 
    resulting array will have the shape (SAMPLE, LINE, BAND).

    Examples
    --------
    >>> cube = pds_dat("data.dat", (3, 64, 1024), "float32")
    >>> cube.shape
    (1024, 64, 3)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the data cannot be reshaped to the requested dimensions.
    """

    # Détermine endianess
    dtype = np.dtype(data_type).newbyteorder(endian)

    BAND, LINE, SAMPLE = data_dims

    # Read the file as a data cube
    with filename_dat.open("rb") as f:
        data = np.fromfile(f, dtype=dtype)

    # This will raise ValueError if the size does not match
    data_cube = data.reshape((SAMPLE, LINE, BAND))

    return data_cube


class AttrDict(dict):
    """
    Dictionary with attribute-style access.

    Keys can be accessed both as dictionary items and as attributes:

        d["KEY"] <-> d.KEY

    This is mainly used for convenience when working with parsed PDS
    labels, where fields like TARGET_NAME or QUBE.AXES are accessed
    as attributes.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")


class Instrument(AttrDict):
    """
    UVIS-like instrument geometry derived from SPICE kernels.

    This class queries the SPICE instrument kernel (IK) for the requested
    instrument (UVIS and specific slit) and builds a simple field-of-view
    model based on the corresponding slit.

    The pixel grid is modeled as a 1-D slit with `n_pixels` samples along
    the spatial direction. Pixel corner directions are computed in angular
    coordinates and converted to unit vectors expressed in the instrument
    frame.

    Parameters
    ----------
    instrument_name : str
        SPICE name of the instrument (e.g., a UVIS channel name resolvable
        by SPICE). This string is passed to SPICE to resolve the instrument
        ID (NAIF ID code).
    n_pixels : int, optional
        Number of spatial pixels used to sample the slit. Default is 64.

    Attributes
    ----------
    name : str
        Instrument name as provided by the user.
    ID : int
        NAIF instrument ID code resolved from `instrument_name`.
    shape : str
        FOV shape returned by SPICE (e.g., "RECTANGLE").
    frame : str
        Instrument frame name in which the FOV geometry is expressed.
    bsight : numpy.ndarray
        Boresight direction (3-vector) in the instrument frame.
    corners : numpy.ndarray
        FOV corner directions (vectors) (shape: (4, 3)) in the instrument frame.
    fov_height : float
        Total FOV height in degrees.
    fov_width : float
        Total FOV width in degrees.
    n_pixels : int
        Number of spatial pixels along the slit, given in argument.
    pixel_height : float
        Angular height of one pixel in degrees.
    pixel_width : float
        Angular width of one pixel in degrees.
    pixels_corners : numpy.ndarray
        Direction vectors for pixel center and corners in the instrument
        frame (shape: (n_pixels, 5, 3)).

        The second dimension uses the following order:
        center, bottom-left, bottom-right, upper-right, upper-left.

    Notes
    -----
    - Angular quantities `fov_height`, `fov_width`, `pixel_height`, and
      `pixel_width` are expressed in degrees.
    - This initializer loads and unloads the instrument kernel (IK). Make
      sure the kernel file path is correctly set in `env.toml` config file.

    Raises
    ------
    spiceypy.utils.exceptions.SpiceyError
        If SPICE cannot resolve the instrument name, access the kernel
        pool variables, or compute the FOV geometry.

    Examples
    --------
    Create a 64-pixel slit model from SPICE IK:

    >>> inst = Instrument("CASSINI_UVIS_FUV", n_pixels=64)
    >>> inst.pixels_corners.shape
    (64, 5, 3)
    """

    def __init__(self, instrument_name: str, n_pixels: int = 64) -> None:
        self.name = instrument_name

        # Load instrument kernel and resolve ID
        spice.furnsh(str(env.ik_path))
        self.ID = spice.bodn2c(instrument_name)
        
        # Get FOV geometry in instrument frame
        # shape: string, frame: string,
        # bsight: boresight vector, bounds: FOV corners
        self.shape, self.frame, self.bsight, self.bounds, self.corners = spice.getfov(self.ID, 4)
        self.corners = np.array(self.corners)
        self.bsight  = np.array(self.bsight)

        # Get half-angles in degrees
        fov_h_angle = spice.gdpool(f'INS{self.ID}_FOV_REF_ANGLE',   0, 1)[0] 
        fov_w_angle = spice.gdpool(f'INS{self.ID}_FOV_CROSS_ANGLE', 0, 1)[0]
        
        self.fov_height = 2 * fov_h_angle # In degrees
        self.fov_width  = 2 * fov_w_angle

        self.n_pixels = int(n_pixels)
        self.pixel_height = self.fov_height / self.n_pixels
        self.pixel_width  = self.fov_width

        spice.unload(str(env.ik_path))


        # COMPUTE PIXEL CORNERS IN INSTRUMENT FRAME
        #------------------------------------------
        # Array indices order:
        # center, bottom-left, bottom-right, upper-right, upper-left

        # Pixel corners in angles (theta, phi)
        pixels_angles = np.zeros((self.n_pixels, 5, 2))
        bc = [0, -fov_h_angle] # Bottom center point

        index = np.arange(self.n_pixels)

        # theta
        pixels_angles[:, :, 0] = np.array(
            [0, -fov_w_angle, fov_w_angle, fov_w_angle, -fov_w_angle]
        )

        # phi (vertical angle)
        pixels_angles[:, 0,   1] = self.pixel_height * (.5 + index)                # Center
        pixels_angles[:, 1:3, 1] = self.pixel_height *       index [:, np.newaxis] # Bottom
        pixels_angles[:, 3:5, 1] = self.pixel_height * ( 1 + index)[:, np.newaxis] # Top
        pixels_angles += bc # Shift all angles by bottom-center offset

        # Convert to radians
        theta = np.radians(pixels_angles[:, :, 0])
        phi   = np.radians(pixels_angles[:, :, 1])

        # Pixel corners in cartesian coordinates
        pixels_corners = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(phi),
            np.cos(theta) * np.cos(phi)
        ], axis=-1)
        
        self.pixels_corners = pixels_corners # / np.linalg.norm(pixels_corners, axis=-1, keepdims=True)


class PDSRawData:
    """
    Raw PDS (Planetary Data System) data from the Cassini Ultraviolet Imaging Spectrograph (UVIS).

    This class represents a single Cassini/UVIS observation stored in PDS3
    format. It requires a binary data file (``.DAT``) containing the raw
    detector counts and a label file (``.LBL``) providing the associated
    metadata.

    The object wraps:

    - the raw detector counts read from the ``.DAT`` file,
    - the parsed PDS label metadata from the ``.LBL`` file,
    - basic derived quantities such as the UVIS channel, slit state,
      pixel bandpass, slit ratio, and integration duration.

    Parameters
    ----------
    filename : str or pathlib.Path
        Either a base filename without extension (e.g.
        ``"FUV2006_015_14_47"``), in which case ``<name>.DAT`` and
        ``<name>.LBL`` are used, or a path to a ``.DAT`` or ``.LBL`` file.
        In the latter case, `file2` must be provided and point to the
        matching file.
    file2 : str or pathlib.Path, optional
        Second file corresponding to `filename` when an explicit
        ``.DAT`` or ``.LBL`` file is provided. Default is ``None``.
    no_extract : bool, optional
        If ``False`` (default), the raw data cube is spatially and
        spectrally cropped according to the window boundaries and
        binning parameters defined in the PDS label. If ``True``, the
        full data cube stored in the ``.DAT`` file is kept.

    Raises
    ------
    ValueError
        If the ``.DAT``/``.LBL`` pairing is invalid, if one of the files
        is missing, or if the data type defined by the
        ``CORE_ITEM_TYPE`` / ``CORE_ITEM_BYTES`` combination is not
        recognized.
    
    Examples
    --------
    Read a PDS file set by specifying the base filename without extension:

    >>> data_pds = PDSRawData('example_file')

    Read a PDS file set by specifying both the label and data files explicitly:

    >>> data_pds = PDSRawData('data_file.DAT', 'label_file.LBL')
    """

    def __init__(self, filename: str | Path, file2: str | Path = None, no_extract: bool = False) -> None:

        filename = Path(filename)
    
        file_error = "Please provide one .LBL file and one .DAT file."
        if file2 is None:
            filedat = filename.with_suffix('.DAT')
            filelbl = filename.with_suffix('.LBL')
        else:
            if filename.suffix.upper() == '.DAT':
                if Path(file2).suffix.upper() == '.LBL':
                    filedat = filename
                    filelbl = Path(file2)
                else:
                    raise ValueError(file_error)
            elif filename.suffix.upper() == '.LBL':
                if Path(file2).suffix.upper() == '.DAT':
                    filedat = Path(file2)
                    filelbl = filename
                else:
                    raise ValueError(file_error)
            else:
                raise ValueError(file_error)
        if not filedat.is_file():
            raise ValueError(f".DAT file: {filedat} does not exist")
        if not filelbl.is_file():
            raise ValueError(f".LBL file: {filelbl} does not exist")
        #____________________________________________________________



        #-----------------------
        # READING DATA AND LABEL

        self.label = pds_lbl(filelbl)
        self.qube  = AttrDict(self.label.QUBE)

        # data_dims = (BAND, LINE, SAMPLE)
        # BAND   : Number of spectral pixels
        # LINE   : Number of spatial pixels
        # SAMPLE : Number of exposures
        data_dims = ast.literal_eval(self.qube.CORE_ITEMS)

        # Binary type
        if   self.qube.CORE_ITEM_TYPE == 'IEEE_REAL'            and self.qube.CORE_ITEM_BYTES == 4 :
            data_type = np.float32
        elif self.qube.CORE_ITEM_TYPE == 'MSB_UNSIGNED_INTEGER' and self.qube.CORE_ITEM_BYTES == 2 :
            data_type = np.uint16
        else : raise ValueError("Unrecognized data type: "+str(self.qube.CORE_ITEM_TYPE))

        # Read
        self.raw_data = pds_dat(filename_dat=filedat, data_dims=data_dims, data_type=data_type)
        self.samples  = self.raw_data.shape[0]
        # data[SAMPLE, LINE, BAND]

        if not no_extract :
            x1 = self.qube.UL_CORNER_BAND
            x2 = self.qube.UL_CORNER_BAND + (self.qube.LR_CORNER_BAND-self.qube.UL_CORNER_BAND+1) // self.qube.BAND_BIN
            y1 = self.qube.UL_CORNER_LINE
            y2 = self.qube.UL_CORNER_LINE + (self.qube.LR_CORNER_LINE-self.qube.UL_CORNER_LINE+1) // self.qube.LINE_BIN
            self.raw_data = self.raw_data[:,y1:y2,x1:x2]
        #_______________________________________________

        # Misc attributes -----
        self.sctime_sec_start = float(self.label.SPACECRAFT_CLOCK_START_COUNT.split('/')[-1])

        self.channel      = 'FUV' if 'FUV' in self.label.PRODUCT_ID else 'EUV'
        self.slit         = self.label.SLIT_STATE
        self.pix_bandpass = pixel_bandpasses[self.channel]
        self.slit_ratio   = slit_ratios[self.channel][self.slit]

        self.INTEGRATION_DURATION = float(self.label.INTEGRATION_DURATION.split()[0])
pds_raw_data = PDSRawData

class UVIS_Bin:
    """
    Container holding the result of a pixel binning operation.

    A UVIS_Bin instance stores:

    - the mapping between detector pixels and bins,
    - per-bin population statistics,
    - optional bin definitions (e.g., geometric boundaries),
    - per-bin geometric line-of-sight (LOS) properties,
    - references to the original unbinned observation arrays.

    The class is agnostic to how bins are constructed: bins may result from
    automatic geometric binning or explicit manual pixel grouping.

    Parameters
    ----------
    shape : tuple of int
        Shape of the bin grid. Each element corresponds to the number of bins
        along one binning dimension. For manual binning this is typically
        (n_bins,).
    uvis_obs : UVIS_Observation
        Parent observation providing the unbinned data arrays and geometry.
    """


    def __init__(self, shape: tuple[int, ...], uvis_obs: UVIS_Observation) -> None:

        self.bins = list_ndarray(shape)
        self.bin_def = None

        self.number_per_bin = np.zeros_like(self.bins, dtype=int)

        self.pixel_LOS = np.copy(uvis_obs.pixel_LOS)
        self.bin_LOS   = np.full_like(self.bins, fill_value=np.nan, dtype=uvis_obs.pixel_LOS.dtype)

        # Unbinned data
        self.name = uvis_obs.name
        self.data = np.copy(uvis_obs.data)
        self.uncertainty_sup = np.copy(uvis_obs.uncertainty_sup)
        self.uncertainty_inf = np.copy(uvis_obs.uncertainty_inf)
        self.WL = np.copy(uvis_obs.WL)

        self.slit_width = uvis_obs.slit_width
        self.slit_dlambda = uvis_obs.slit_dlambda
        self.HD = uvis_obs.HD

        self.bin_averaged   = False
        self.bin_integrated = False

    def average(self):
        """
        Compute per-bin mean spectra and uncertainties.

        For each bin, this method computes [1]:

        - an unweighted mean spectrum, sample standard deviation (ddof=1),
          and corrected standard error of the mean,
        - a weighted mean spectrum using 1/σ² weights, where σ is the
          symmetric uncertainty (0.5*(sup+inf)),
        - propagated upper and lower uncertainties for the weighted mean
          for each wavelength.

        Empty bins are left as NaN. Results are stored as attributes and
        ``self.bin_averaged`` is set to True.
        
        Returns
        -------
        None

        Attributes
        ----------
        `bin_mean_spectrum` : ndarray
            Unweighted mean spectrum per bin.
        `bin_stddev_spectrum` : ndarray
            Unweighted standard deviation per bin.
        `bin_stderr_spectrum` : ndarray
            Unweighted standard error per bin (with small-sample correction).
        `bin_wmean_spectrum` : ndarray
            Weighted mean spectrum per bin.
        `bin_u_sup_spectrum` : ndarray
            Propagated upper uncertainty of weighted mean per bin.
        `bin_u_inf_spectrum` : ndarray
            Propagated lower uncertainty of weighted mean per bin.

        References
        ----------
        [1] Le Guennic et al. (2026)
        """

        bin_shape = self.bins.shape
        out_shape = bin_shape + self.WL.shape

        self.bin_mean_spectrum   = np.full(out_shape, np.nan, dtype=float)
        self.bin_stddev_spectrum = np.full(out_shape, np.nan, dtype=float)
        self.bin_stderr_spectrum = np.full(out_shape, np.nan, dtype=float)
        self.bin_wmean_spectrum  = np.full(out_shape, np.nan, dtype=float)
        self.bin_u_sup_spectrum  = np.full(out_shape, np.nan, dtype=float)
        self.bin_u_inf_spectrum  = np.full(out_shape, np.nan, dtype=float)

        for idx in np.ndindex(bin_shape):
            pairs = self.bins[idx]
            if not pairs: continue # Empty bin

            # 1) Gather data and uncertainties for all pixels in the bin
            stacked_data = np.array([self.data[i, j, :]            for (i, j) in pairs])            
            stacked_sup  = np.array([self.uncertainty_sup[i, j, :] for (i, j) in pairs]) 
            stacked_inf  = np.array([self.uncertainty_inf[i, j, :] for (i, j) in pairs])

            # 2) Unweighted mean & std deviation & standard error
            N = len(pairs)
            
            self.bin_mean_spectrum[idx]   = stacked_data.mean(axis=0)
            if N>1 :
                self.bin_stddev_spectrum[idx] = stacked_data.std(axis=0, ddof=1)

                correction = correction_factor(N)
                self.bin_stderr_spectrum[idx] = correction * self.bin_stddev_spectrum[idx] / np.sqrt(N)
            else :
                self.bin_stddev_spectrum[idx] = np.zeros_like(self.WL)
                self.bin_stderr_spectrum[idx] = np.zeros_like(self.WL)

            # 3) Weighted mean using combined uncertainty = (sup + inf)/2
            combined_unc = 0.5 * (stacked_sup + stacked_inf)
            weights      = 1.0 / np.square(combined_unc)

            w_sum        = weights.sum(axis=0)
            w_data_sum   = (stacked_data * weights).sum(axis=0)

            # Weighted mean (avoid division by zero where w_sum==0)
            self.bin_wmean_spectrum[idx] = np.divide(
                w_data_sum, w_sum,
                out=np.full(self.WL.shape, np.nan),
                where=(w_sum>0)
            )

            # 4) Propagate separate upper/lower uncertainties
            inv_sup_sq = 1.0 / np.square(stacked_sup)
            inv_inf_sq = 1.0 / np.square(stacked_inf)

            sum_inv_sup_sq = inv_sup_sq.sum(axis=0)
            sum_inv_inf_sq = inv_inf_sq.sum(axis=0)

            # sqrt(1 / sum(1/unc^2))
            self.bin_u_sup_spectrum[idx] = np.sqrt(
                np.divide(1.0, sum_inv_sup_sq,
                          out=np.full(self.WL.shape, np.nan), where=(sum_inv_sup_sq>0))
                )
            self.bin_u_inf_spectrum[idx] = np.sqrt(
                np.divide(1.0, sum_inv_inf_sq,
                          out=np.full(self.WL.shape, np.nan), where=(sum_inv_inf_sq>0))
                )
            
        self.bin_averaged = True

    def integrate(self, wl_range: tuple[float, float] | None = None, uncertainty: bool = True, method: Literal['simpson', 'trapezoid', 'trapz'] = 'simpson') -> None:
        """
        Integrate spectra over a wavelength band and compute per-bin integrated products.

        This method performs three integration tasks:

        - Integrates the raw pixel-level spectra and uncertainties [1].
        - Averages these integrated values within each bin.
        - If bins have already been averaged (via :meth:`average`), also integrates
          the bin-averaged spectra.

        Parameters
        ----------
        wl_range : tuple of float, optional
            Integration bounds (min_wl, max_wl).
        uncertainty : bool, optional
            If True (default), uncertainty spectra are propagated under the
            integral (see :func:`integrate_spectrum`).
        method : {"simpson", "trapezoid", "trapz"}, optional
            Numerical integration method passed to :func:`integrate_spectrum`.
            `trapz` is an alias for `trapezoid`.
            Default is "simpson".

        Returns
        -------
        None

        Attributes
        -----
        integrated_data : ndarray
            Integrated spectrum for each pixel.
        integrated_uncertainty_sup : ndarray
            Integrated upper uncertainty for each pixel.
        integrated_uncertainty_inf : ndarray
            Integrated lower uncertainty for each pixel.
        binned_integrated_data : ndarray
            Mean of integrated spectra of each pixel per bin.
        binned_integrated_uncertainty_sup : ndarray
            Mean of integrated upper uncertainty of each pixel per bin.
        binned_integrated_uncertainty_inf : ndarray
            Mean of integrated lower uncertainty of each pixel per bin.
        bin_stddev : ndarray
            Standard deviation of integrated values per bin.
        bin_stderr : ndarray
            Standard error of integrated values per bin.
        integrated_avrg_data : ndarray
            Integration of unweighted bin-averaged spectra.
        integrated_avrg_data_w : ndarray
            Integration of weighted bin-averaged spectra.
        integrated_avrg_stddev : ndarray
            Integration of bin standard deviation spectrum.
        integrated_avrg_stderr : ndarray
            Integration of bin standard error spectrum.
        integrated_avrg_uncertainty_sup : ndarray
            Integration of bin upper uncertainty spectra.
        integrated_avrg_uncertainty_inf : ndarray
            Integration of bin lower uncertainty spectra.

        References
        ----------
        [1] Le Guennic et al. (2026)
        """

        # Integrate spectra
        self.integrated_data            = integrate_spectrum(self.WL, self.data,            method=method, wl_range=wl_range, axis=-1)
        self.integrated_uncertainty_inf = integrate_spectrum(self.WL, self.uncertainty_inf, method=method, wl_range=wl_range, axis=-1, uncertainty=uncertainty)
        self.integrated_uncertainty_sup = integrate_spectrum(self.WL, self.uncertainty_sup, method=method, wl_range=wl_range, axis=-1, uncertainty=uncertainty)


        # Average integrated arrays
        self.binned_integrated_data            = np.full(self.bins.shape, np.nan, dtype=float)
        self.binned_integrated_uncertainty_sup = np.full(self.bins.shape, np.nan, dtype=float)
        self.binned_integrated_uncertainty_inf = np.full(self.bins.shape, np.nan, dtype=float)
        self.bin_stddev                        = np.full(self.bins.shape, np.nan, dtype=float)
        self.bin_stderr                        = np.full(self.bins.shape, np.nan, dtype=float)

        for idx in np.ndindex(self.bins.shape):
            
            pairs = self.bins[idx]
            if not pairs: continue

            N = len(pairs)

            self.binned_integrated_data[idx] = np.mean([self.integrated_data[i, j] for (i, j) in pairs])

            if N>1:
                correction = correction_factor(N)

                self.bin_stddev[idx]             = np.std ([self.integrated_data[i, j] for (i, j) in pairs], ddof=1)
                self.bin_stderr[idx]             = correction * self.bin_stddev[idx] / np.sqrt(N)

            else:
                self.bin_stddev[idx]             = 0
                self.bin_stderr[idx]             = 0

            self.binned_integrated_uncertainty_sup[idx] = np.sqrt(1/np.sum([1/self.integrated_uncertainty_sup[i, j]**2 for (i, j) in pairs]))
            self.binned_integrated_uncertainty_inf[idx] = np.sqrt(1/np.sum([1/self.integrated_uncertainty_inf[i, j]**2 for (i, j) in pairs]))

        self.bin_integrated = True


        # Integrate the already bin-averaged spectra
        if self.bin_averaged:
            self.integrated_avrg_data            = integrate_spectrum(self.WL, self.bin_mean_spectrum,   wl_range=wl_range, axis=-1, method=method)
            self.integrated_avrg_data_w          = integrate_spectrum(self.WL, self.bin_wmean_spectrum,  wl_range=wl_range, axis=-1, method=method)

            self.integrated_avrg_stddev          = integrate_spectrum(self.WL, self.bin_stddev_spectrum, wl_range=wl_range, axis=-1, method=method, uncertainty=uncertainty)
            self.integrated_avrg_stderr          = integrate_spectrum(self.WL, self.bin_stderr_spectrum, wl_range=wl_range, axis=-1, method=method, uncertainty=uncertainty)
            self.integrated_avrg_uncertainty_sup = integrate_spectrum(self.WL, self.bin_u_sup_spectrum,  wl_range=wl_range, axis=-1, method=method, uncertainty=uncertainty)
            self.integrated_avrg_uncertainty_inf = integrate_spectrum(self.WL, self.bin_u_inf_spectrum,  wl_range=wl_range, axis=-1, method=method, uncertainty=uncertainty)

    def plot_bin(self, show: bool = True):
        """
        Display a table visualization of the number of pixels per bin.

        Creates a matplotlib table showing bin centers as row/column labels
        and the count of pixels in each bin as cell values. Empty bins are
        displayed as blank cells.

        Parameters
        ----------
        show : bool, optional
            If True (default), call plt.show(). The figure and axes are still
            returned.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The created axes.

        Raises
        ------
        ValueError
            If the bin grid is not 2D.

        Notes
        -----
        - Only works for 2D bin grids.
        - Bin centers are computed as midpoints of bin edges.
        """
            
        import matplotlib.pyplot as plt

        if self.bins.ndim != 2:
            raise ValueError("plot_bin() only supports 2D bin grids.")
        
        number_per_bin = np.where(self.number_per_bin == 0, "", self.number_per_bin.astype(str))
        number_per_bin = number_per_bin.T
        number_per_bin = np.flip(number_per_bin, axis=0)
        
        col_val    = list(self.bin_def.values())[0]
        col_labels = [str((col_val[i+1]+col_val[i])/2) for i in range(self.bins.shape[0])]

        row_val    = list(self.bin_def.values())[1]
        row_labels = [str((row_val[i+1]+row_val[i])/2) for i in range(self.bins.shape[1])][::-1]

        fig, ax = plt.subplots()
        ax.set_axis_off()


        table = ax.table(
            cellText=number_per_bin,
            loc='center',
            cellLoc='center',
            colLabels=col_labels,
            rowLabels=row_labels
        )

        plt.tight_layout()
        if show: plt.show()
        return fig, ax

    def save(self, filepath: str | Path, overwrite: bool = False) -> str:
        """
        Save the UVIS_Bin object to disk.

        The file extension is set to ".uvisbin" if not provided.

        Parameters
        ----------
        filepath : str or Path
            Output path. If no ".uvisbin" suffix is provided, it is appended.
        overwrite : bool, optional
            If True, existing files are overwritten without prompting.
            If False (default), the user is asked for confirmation.

        Returns
        -------
        str or Path
            The original `filepath` argument.

        Notes
        -----
        The method uses the `pickle` module.
        """

        p = Path(filepath)
        if p.suffix.lower() != '.uvisbin':
            p = p.with_suffix('.uvisbin')

        print(f"Saving UVIS observation bin object {p.stem}...", end='', flush=True)

        if p.exists() and not overwrite:
            response = input(f"\nFile '{p.absolute()}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response not in ('y', 'yes', 'o', '1', 'oui'):
                print("Save cancelled.")
                return

        with p.open('wb') as f:
            pickle.dump(self, f)
        
        print(' Done')

        return filepath
    
    @classmethod
    def load(cls, filepath: str | Path) -> UVIS_Bin:
        """
        Load a UVIS_Bin object from a pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to the pickle file containing the serialized object.

        Returns
        -------
        UVIS_Bin
            The deserialized UVIS_Bin object.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        pickle.UnpicklingError
            If the file cannot be unpickled or is corrupted.
        """

        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        """
        Return a string representation of the UVIS_Bin object.

        Returns
        -------
        str
            Summary of bin configuration and state.
        """

        info          = f"<UVIS_bin object>\n"
        info         += f"  Observation: {self.name}\n"
        info         += f"  Bin shape  : {self.bins.shape}\n"
        info         += f"  Bin attributes:\n"

        if self.bin_def is not None:
            for key, val in self.bin_def.items():
                info += f"    - {key}: {len(val)-1} bins ({val[0]} to {val[-1]})\n"
        return info

class UVIS_Observation:
    """
    Cassini/UVIS observation assembled from one or more PDS3 products.

    This class loads UVIS PDS3 data/label pairs (``.DAT``/``.LBL``),
    concatenates exposures in time order, and provides utilities for:

    - calibration (counts/s -> radiance),
    - background noise estimation and subtraction,
    - geometry computation from SPICE kernels,
    - stellar contamination identification,
    - pixel binning based on geometry or manual selection.

    See [1] for details on UVIS data processing.

    Parameters
    ----------
    *files : str or pathlib.Path or iterable of (str or pathlib.Path)
        One or more PDS base paths (with or without extension). Each entry can be:
        If a single iterable is passed (e.g. a list of paths), it is unpacked.

        - If exactly two paths are provided and they form a single .LBL/.DAT pair,
          they are interpreted as one explicit product.
        - Otherwise, each input is interpreted as a product base path: the extension
          (if any) is ignored and the corresponding .LBL and .DAT files are expected
          to share the same base name.

        Strings are treated as paths, not as iterables of characters.

    batch : str or pathlib.Path, optional
        Path to a text file listing PDS products, one per line.

        Each line may be an absolute path or a path relative to the location of
        the batch file. Relative paths are interpreted with respect to the
        directory containing the batch file.

        If ``batch`` is provided, it overrides ``*files``. The same pairing rules
        apply as for ``*files``: if the batch contains exactly two entries forming
        a .LBL/.DAT pair, they are treated as a single explicit product;
        otherwise, each entry is interpreted as a product base path.

    target : str, optional
        Main target name used for georeference computations. Stored uppercased. Default is ``None``.

    prime_instrument : {"PRIME", "UVIS", "CIRS", "VIMS", "ISS"}, optional
        Prime instrument tag stored in metadata and used for naming.
        If `"UVIS"` is provided, it is normalized
        internally to `"PRIME"`.
    ID : int, optional
        Extra identifier used when multiple observations share the same base name.
        Default is 0 (no identifier).

    name : str, optional
        If provided, overrides the auto-generated observation name.
        The default name is constructed as:
        ``CHANNEL_YEAR_DOY_INSTRUMENT(_ID)``
    sort : bool, optional
        If True (default), input files (and resulting exposures) are ordered by
        spacecraft clock start time.

    Attributes
    ----------
    name : str
        Observation name (auto-generated or user-provided).
    target : str or None
        Target name (uppercased) or None.
    YEAR : int
        Observation year (from first exposure).
    DOY : int
        Day of year of the observation (from first exposure).
    channel : {"FUV", "EUV"}
        UVIS channel inferred from the first product ID.
    prime : str
        Prime instrument tag used in metadata/naming.
    n_pics : int
        Number of exposures (time samples).
    n_pixels : int
        Number of spatial pixels.
    n_wl : int
        Number of spectral pixels.

    slit : str
        Slit state (from first product). Can be "OCCULTATION", "LOW_RESOLUTION", or "HIGH_RESOLUTION".
    slit_width : float
        Slit width [µm].
    slit_dlambda : float
        Slit width image on the spectral dimension detector [Å].
        Defines the point spread function (PSF).
    
    WL : numpy.ndarray
        Wavelength grid for the detector [Å]. Shape: (n_wl,).

    counts : numpy.ndarray
        Raw detector counts. Shape: (n_pics, n_pixels, n_wl).
    cps : numpy.ndarray
        Counts per second (counts / expo_time). Same shape as :attr:`counts`.
    cps_bg_removed : numpy.ndarray
        Background-corrected counts per second. Same shape as :attr:`counts`.

    data : numpy.ndarray
        Calibrated radiance array [kR]. Same shape as :attr:`counts`). Populated by :meth:`calibrate()`.
    uncertainty_sup : numpy.ndarray or None
        Upper radiance uncertainty array. Populated by :meth:`calibrate()` / :meth:`get_radiance_uncertainty()`.
    uncertainty_inf : numpy.ndarray or None
        Lower radiance uncertainty array. Populated by :meth:`calibrate()` / :meth:`get_radiance_uncertainty()`.

    calibration : numpy.ndarray
        Calibration multiplier per exposure [kR/counts]. Same shape as :attr:`counts`.
    calibration_error : numpy.ndarray
        Calibration uncertainty array.
    geometry : list[Geometry] or None
        Per-exposure :class:`Geometry` objects. Populated by :meth:`set_geometry()`.
    pixel_LOS : numpy.ndarray or None
        Structured array of line-of-sight geometry per pixel and exposure.
        Fields: ``"lon"`` [°], ``"lat"`` [°], ``"alt"`` [km],
        ``"sza"`` [°], ``"phase"`` [°], ``"ems"`` [°], ``"lt"`` [h].
        Shape: (n_pics, n_pixels, 5) for each field.
        The last dimension corresponds to the pixel center,
        bottom-left, bottom-right, upper-right, and upper-left corners, respectively.
    spacecraft_position : numpy.ndarray or None
        Sub-spacecraft point properties in planetocentric coordinates (shape: (n_pics,)
        Fields: ``"lon"`` [°], ``"lat"`` [°], ``"alt"`` [km],
        ``"sza"`` [°], ``"phase"`` [°], ``"ems"`` [°], ``"lt"`` [h].
        Altitude is the spacecraft altitude.
    HD : float or None
        Mean heliocentric distance during the observation. Populated by :meth:`set_geometry()`.

    pixel_stars_mask : numpy.ndarray
        Boolean mask of star-contaminated pixels (shape: (n_pics, n_pixels)).
    pixel_corrupted : numpy.ndarray
        Boolean mask of pixels showing partial data (shape: (n_pics, n_pixels)).
    evil_pixels : numpy.ndarray
        Boolean mask of known bad pixels on the detector. Shape: (64, 1024).
    evil_pixels_binned : numpy.ndarray
        Boolean mask of known bad pixels after binning. Shape: (n_pixels, n_wl).

    background_level : float
        Background level in counts/s per spectral pixel (before binning factors).
    background_error : float
        Uncertainty on `background_level`.

    ET_start : numpy.ndarray
        Ephemeris times at each exposure start (seconds past J2000).
    ET_middle : numpy.ndarray
        Ephemeris times at each exposure mid-point.
    ET_stop : numpy.ndarray
        Ephemeris times at each exposure end.
    UTC_start : list[str]
        UTC strings for exposure start times.
    UTC_middle : list[str]
        UTC strings for exposure mid-point times.
    UTC_stop : list[str]
        UTC strings for exposure end times.
    time_exposition: numpy.ndarray
        Array of times during the each exposure (s).
        Shape : (n_pics, PicsPerExposure).
        PicsPerExposure is defined in the pipeline defaults file. Default is 60.
        Useful for getting geometric information at finer time resolution than the exposure cadence.

    is_calibrated : bool
        True if `data` has been populated by `calibrate()`.
    calibration_set : bool
        True if per-exposure calibration arrays are set.
    is_bkg_removed : bool
        True if background correction has been applied to `cps_bg_removed`.
    is_smoothed : bool
        True if smoothing has been applied to calibrated spectra.

    instrument : Instrument
        Cassini/UVIS :class:`Instrument` object providing instrument properties
        and SPICE instrument kernel management.
    pds_data : list of PDSRawData
        List of loaded PDS products forming the observation.


    Notes
    -----
    - All PDS products are saved into :attr:`self.pds_data` as :class:`PDSRawData` objects.
    - Geometry is not computed during initialization. Call :meth:`set_geometry()` to populate
      :attr:`self.geometry` and :attr:`self.pixel_LOS`.
    - This initializer loads and unloads SPICE kernels (LSK, and IK indirectly via Instrument). Ensure
      the kernel paths are correctly configured in your environment configuration.

    See Also
    --------
    :meth:`set_geometry` : Compute SPICE-based geometry for each exposure.
    :meth:`set_background` : Estimate and/or set detector background level.
    :meth:`calibrate` : Apply radiometric calibration and populate calibrated arrays.
    :meth:`bin_pixels` : Bin pixels geometrically or manually into a :class:`UVIS_Bin` container.

    References
    ----------
    [1] Le Guennic et al. (2026)
    """

    def __init__(
            self,
            *files: str | Path | Iterable[str | Path],
            batch: str | Path | None = None,
            target: str = None,
            prime_instrument: Literal["PRIME", "UVIS", "CIRS", "VIMS", "ISS"] = None,
            ID: int = 0,
            name: str | None = None,
            sort: bool = True,
    ):
        """
        Build an observation from UVIS PDS files.
        """
        
        # READING DATA
        #________________________
        if batch is not None :
            # Read batch .txt file of PDS files
            batch = Path(batch)
            with batch.open('r') as f :
                files = [Path(line.strip()) for line in f if line.strip()]

            if batch.is_absolute():
                batch_parent = batch.parent
                files = [f if f.is_absolute() else batch_parent / f for f in files]


        else :
            if len(files) == 1 and hasattr(files[0], '__iter__') and not isinstance(files[0], (str, bytes, Path)):
                files = files[0]
        files = [Path(f) for f in files]

        if ( # Only one pair of .LBL/.DAT files provided
            len(files) == 2
            and {files[0].suffix.upper(), files[1].suffix.upper()} == {".LBL", ".DAT"}
        ):
            self.pds_data  = [PDSRawData(files[0], files[1])]
            self.raw_files = [str(files[0]), str(files[1])]
        else:
            files = [str(f.with_suffix('')) for f in files]
            files = list(dict.fromkeys(files)) # Remove duplicates while preserving order

            if sort :
                # Sort files by spacecraft clock start time
                files_with_sctime = []
                for f in files :
                    pds    = PDSRawData(f)
                    sctime = float(pds.label.SPACECRAFT_CLOCK_START_COUNT.split('/')[-1])
                    files_with_sctime.append( (f, sctime) )
                files_with_sctime.sort(key=lambda x: x[1])
                files = [f[0] for f in files_with_sctime]
            self.pds_data  = [PDSRawData(f) for f in files]
            self.raw_files = [str(f)        for f in files]
        #________________________



        # MAIN DATA
        #______________________________

        self.counts    = np.concatenate(
            [e.raw_data for e in self.pds_data], axis=0               # Raw counts on detector
            )
        self.data      = np.zeros_like(self.counts, dtype=float)      # Calibrated data (kR)

        self.expo_time      = self.pds_data[0].INTEGRATION_DURATION   # Exposition duration (s)
        self.cps            = self.counts/self.expo_time              # Counts per second
        self.cps_bg_removed = self.cps.copy()                         # Background-corrected counts per second


        

        # Uncertainty on detector counts
        self.uncertainty_sup = None
        self.uncertainty_inf = None

        # Calibration arrays
        self.calibration       = np.zeros_like(self.counts, dtype=float) # Calibration factor
        self.calibration_error = np.zeros_like(self.counts, dtype=float) # Calibration factor uncertainty
        

        # MAIN METADATA
        #______________________________
        
        nameid = self.pds_data[0].label.PRODUCT_ID

        # Date
        self.YEAR     = int(nameid[3:7])
        self.DOY      = int(nameid[8:11])           # Day of year at the beginning of observation
        self.prime    = prime_instrument            # UVIS (PRIME), CIRS, VIMS or ISS
        if prime_instrument=='UVIS' : self.prime='PRIME'
        self.is_prime = prime_instrument=='PRIME'
        if prime_instrument is None :
            self.is_prime = None
            self.prime=''

        # Detector properties
        self.n_pics   = self.counts.shape[0]   # Number of exposures
        self.n_pixels = self.counts.shape[1]   # Number of spatial  pixels
        self.n_wl     = self.counts.shape[2]   # Number of spectral pixels

        # UVIS binning
        self.spat_bin   = self.pds_data[0].qube.LINE_BIN
        self.spec_bin   = self.pds_data[0].qube.BAND_BIN
        self.spat_start = self.pds_data[0].qube.UL_CORNER_LINE
        self.spat_stop  = self.pds_data[0].qube.LR_CORNER_LINE
        self.spec_start = self.pds_data[0].qube.UL_CORNER_BAND
        self.spec_stop  = self.pds_data[0].qube.LR_CORNER_BAND

        # Channel and slit
        self.channel      = 'FUV' if 'FUV' in nameid else 'EUV'
        self.pix_bandpass = pixel_bandpasses[self.channel]
        self.slit         = self.pds_data[0].label.SLIT_STATE
        self.slit_ratio   = slit_ratios [self.channel][self.slit]
        self.slit_width   = slit_width  [self.channel][self.slit]
        self.slit_dlambda = slit_dlambda[self.channel][self.slit]

        self.evil_pixels        = None  # Mask: True when evil pixel
        self.evil_pixels_binned = None

        

        # Name
        self.ID = ID
        if ID>0 : self.IDstr = '_'+str(ID) # Identifier for multiple observation during one DOY
        else    : self.IDstr = ''

        if name is None :
            if self.prime!='' :
                # Observation name format : CHANNEL_YEAR_DOY_PRIME(_ID)
                self.name = str(self.channel)+'_'+str(self.YEAR)+'_'+nameid[8:11]+'_'+self.prime+self.IDstr
            else:
                # Observation name format : CHANNEL_YEAR_DOY(_ID) self.name = str(self.channel)+'_'+str(self.YEAR)+'_'+nameid[8:11]+self.IDstr
                self.name = str(self.channel)+'_'+str(self.YEAR)+'_'+nameid[8:11]+self.IDstr
        else : self.name=name
        

        # Wavelength range
        self.WL = UVIS_WL(self.channel, bin=self.spec_bin)




        # GEOMETRY
        #__________________
        self.geometry         = None  # List of geometry objects
        self.pixel_stars      = []    # List of star-contaminated pixel coordinates
        self.pixel_stars_mask = np.zeros((self.n_pics, self.n_pixels), dtype=bool)  # Pixel with star contamination
        self.pixel_corrupted  = np.zeros((self.n_pics, self.n_pixels), dtype=bool)  # Pixel with transmission loss
        self.markers = {}

        self.pixel_LOS = None
        self.pixel_star_geometry = None

        self.HD = None # Mean heliocentric distance of Cassini during the observation


        # MISCELLANEOUS
        #______________________________

        # Instrument
        match self.slit :
            case 'LOW_RESOLUTION'  : self.slit_ID = 'LO'
            case 'HIGH_RESOLUTION' : self.slit_ID = 'HI'
            case 'OCCLTATION'      : self.slit_ID = 'OCC'
        self.instrument_name = 'CASSINI_UVIS_'+self.channel+'_' + self.slit_ID
        self.instrument = Instrument(self.instrument_name, 64)


        # Observation
        self.target = target.upper()

        self.background_level = None
        self.background_error = None
        self.n_bg_pixels = None
        self.max_gap = None

        # Instance status
        self.is_calibrated   = False
        self.calibration_set = False
        self.is_bkg_removed  = False
        self.is_smoothed     = False




        # TIMES
        #______________________________

        spice.furnsh(str(env.lsk_path))

        # Spacecraft clock start for each LBL file
        self.sctime_sec_start = np.array( [float(e.label.SPACECRAFT_CLOCK_START_COUNT.split('/')[-1]) for e in self.pds_data] )-self.expo_time

        # Times for each exposure
        samples   = np.array([e.samples for e in self.pds_data])
        pds_ET_start = np.array( [spice.str2et(utc) for utc in [e.label.START_TIME for e in self.pds_data]] )
        pds_ET_start-=self.expo_time # Label start time is given at the END of the first sample (see UVIS User Guide)


        # Start, middle and stop ET for each exposure
        self.ET_start  = np.concatenate(
            [et +  np.arange(s)      * self.expo_time   for et, s in zip(pds_ET_start, samples)] )
        self.ET_middle = np.concatenate(
            [et + (np.arange(s)+0.5) * self.expo_time   for et, s in zip(pds_ET_start, samples)] )
        self.ET_stop   = np.concatenate(
            [et + (np.arange(s)+  1) * self.expo_time   for et, s in zip(pds_ET_start, samples)] )
        
        # Start and stop UTC for each exposure
        self.UTC_start  = [spice.et2utc(et, "ISOD", 3) for et in self.ET_start ]
        self.UTC_middle = [spice.et2utc(et, "ISOD", 3) for et in self.ET_middle]
        self.UTC_stop   = [spice.et2utc(et, "ISOD", 3) for et in self.ET_stop  ]

        del samples, pds_ET_start
        spice.unload(str(env.lsk_path))

        # Sub-exposure times
        self.times_exposition = np.array([np.linspace(self.ET_start[i], self.ET_stop[i], PicsPerExposure, endpoint=True) for i in range(self.n_pics)])


    def integrate_radiance(self, wl_range: tuple[float, float] = None, method: Literal['simpson', 'trapezoid', 'trapz'] = 'simpson') :
        """
        Integrate the calibrated radiance over a specified wavelength range.

        Parameters
        ----------
        wl_range : tuple of float, optional
            Wavelength range (min, max) for integration.
            If None (default), integrates over the full wavelength range of the detector.
        method : {'simpson', 'trapezoid'}, optional
            Integration method to use. Default is 'simpson'.
            See :func:`cassini_upyp.uvisutils.integrate_spectrum` for details.

        Returns
        -------
        numpy.ndarray
            Integrated radiance values per exposure and spatial pixel.
        """


        if self.is_calibrated :
            signal = self.data.copy()
        else :
            if not self.calibration_set : self.set_calibration()
            signal = np.array([interpolate_nans(self.cps[i] * self.calibration[i]) for i in range(self.n_pics)])

        # Integrate spectrum
        return integrate_spectrum(self.WL, signal, wl_range=wl_range, axis=2, method=method)
    
    def get_radiance_uncertainty(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-pixel radiance uncertainties (upper/lower).

        Uses Garwood Poisson bounds on detector counts, then propagates
        count and calibration uncertainties to radiance. If a background level is set,
        adds its contribution in quadrature.

        Uses :func:`cassini_upyp.uvisutils.poisson_error` to compute count uncertainties.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (uncertainty_sup, uncertainty_inf), same shape as ``self.counts``.

        Raises
        ------
        RuntimeError
            If calibration has not been set prior to calling this method.
        """

        if not self.calibration_set:
            raise RuntimeError("Calibration must be set before computing radiance uncertainty (run set_calibration()).")


        counts_position = self.counts>0

        if self.background_level is None:
            bg_radiance_err = 0
        elif self.background_level == 0:
            bg_radiance_err = 0
        else:
            bg_radiance_err = self.background_level * self.calibration * np.sqrt(
                (self.background_error/self.background_level)**2 + (self.calibration_error/self.calibration)**2
            )

        # Scores error interval
        count_sup_err = poisson_error(self.counts, bound='sup') - self.counts
        count_inf_err = poisson_error(self.counts, bound='inf') - self.counts

        uncertainty_sup = np.zeros_like(self.counts, dtype=float)
        uncertainty_inf = np.zeros_like(self.counts, dtype=float)

        # Upper bound for zero counts
        uncertainty_sup[~counts_position] = self.calibration[~counts_position] / self.expo_time

        uncertainty_sup[ counts_position] = (
            self.cps_bg_removed[counts_position] * self.calibration[counts_position] *
            np.sqrt(
                (count_sup_err[counts_position]          / self.counts[counts_position])**2 +
                (self.calibration_error[counts_position] / self.calibration[counts_position])**2
            )
        )

        uncertainty_inf[ counts_position] = (
            self.cps_bg_removed[counts_position] * self.calibration[counts_position] *
            np.sqrt(
                (count_inf_err[counts_position] / self.counts[counts_position])**2 +
                (self.calibration_error[counts_position] / self.calibration[counts_position])**2
            )
        )

        uncertainty_sup = np.sqrt(uncertainty_sup**2 + bg_radiance_err**2)
        uncertainty_inf = np.sqrt(uncertainty_inf**2 + bg_radiance_err**2)

        return uncertainty_sup, uncertainty_inf

    def smooth(self, force: bool = False, smoothing_kernel: ArrayLike = smoothing_kernel) -> None:
        """
        Smooth the calibrated FUV data and uncertainties using 1D convolution.
        Uses :func:`cassini_upyp.uvisutils.smooth_spectrum` for convolution.

        Parameters
        ----------
        force : bool, optional
            If True, forces smoothing even if data is already smoothed. Default is False.
        smoothing_kernel : array-like, optional
            1D array defining the smoothing kernel.

        Notes
        -----
        Smoothing is applied only on valid (non-NaN) data points.
        """

        if self.channel!='FUV' :
            return
        if self.is_smoothed and not force :
            print('Spectral data is already smoothed')
            return

        valid = ~np.isnan(self.data) # Convolution will extend NaN domains
        self.data[valid]            =         smooth_spectrum(self.data[valid],               smoothing_kernel    )
        self.uncertainty_sup[valid] = np.sqrt(smooth_spectrum(self.uncertainty_sup[valid]**2, smoothing_kernel**2))
        self.uncertainty_inf[valid] = np.sqrt(smooth_spectrum(self.uncertainty_inf[valid]**2, smoothing_kernel**2))

        self.is_smoothed = True


    # -------- CALIBRATION
    def get_calibration(self, sctime: float, interp: Literal['linear', 'pchip'] = 'pchip', flat_field: bool = True) -> dict[str, np.ndarray]:
        """
        Retrieve the calibration multiplier (inverse sensitivity) of the Cassini UVIS instrument.

        The method incorporates several calibration steps:

            - Laboratory calibration data is adjusted for slit width.
            - Time variation is accounted for using spacecraft time.
            - Flat-field corrections are applied if `flat_field` is `True`.
            - Binning is performed according to the spatial and spectral binning factors.

        Parameters
        ----------
        sctime : float
            Spacecraft time of the observation, used to apply time-dependent calibration
            modifiers and to select the appropriate flat-field epoch.
        interp : {'linear', 'pchip'}, optional
            Interpolation method used to map the lab calibration to the full detector range.
            See :func:`uvisutils.interpolate_nans` for details.

            Default is 'pchip'.
        flat_field : bool, optional
            If True, apply flat-field correction (including its time variation). If False,
            the flat-field is effectively disabled.
            Default is `True`.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing:

            - 'calibration'       : numpy.ndarray
                The calibration multiplier, binned and shaped to match the raw data dimensions.
            - 'calibration_error' : numpy.ndarray
                The calibration error array.

        Notes
        -----

        - For the FUV channel, pixels known as 'evil' pixels with anomalous behavior are handled,
          and corresponding elements in the arrays are set to NaN.
        """

        # -- LABORATORY CALIBRATION
        lab_cal=uvis_lab_calibration(self.channel)

        wavelength_lab        = lab_cal['WAVELENGTH']
        sensitivity_lab       = lab_cal['SENSITIVITY']
        sensitivity_lab_error = lab_cal['SENSITIVITY_ERROR']

        sensitivity_lab       /= self.slit_ratio
        sensitivity_lab_error /= self.slit_ratio

        # Interpolate lab calibration wavelength
        WL = UVIS_WL(self.channel)
        match interp:
            case 'pchip':
                sensitivity       = PchipInterpolator(wavelength_lab,sensitivity_lab)(WL)
                sensitivity_error = PchipInterpolator(wavelength_lab,sensitivity_lab_error)(WL)
            case 'linear':
                sensitivity       = np.interp(WL, wavelength_lab,sensitivity_lab)
                sensitivity_error = np.interp(WL, wavelength_lab,sensitivity_lab_error)
            case _: raise ValueError('Incorrect interpolation method')

        
        # -- TIME VARIATION
        specmod = get_cal_time_variation(self.channel, sctime)

        # Apply bandpass if continuous input spectrum selected
        # Sensitivity units now (counts/second) / (kilorayleigh) / (angstrom)
        sensitivity       *= specmod*self.pix_bandpass
        sensitivity_error *= specmod*self.pix_bandpass

        # 2D Sensitivity
        # Divide the array by number of illuminated rows
        n_spat_pix = self.pds_data[0].qube.LR_CORNER_LINE+1-self.pds_data[0].qube.UL_CORNER_LINE
        sensitivity       = np.tile(sensitivity,       (64,1)) /60 # TODO : EUV different value?
        sensitivity_error = np.tile(sensitivity_error, (64,1)) /60



        # -- FLAT FIELD
        if sctime < sctimeburn :
            ff = read_spica_ff(Path(env.calibration_dir) / f'FLATFIELD_{self.channel}_PREBURN.txt')
        else :
            ff = read_spica_ff(Path(env.calibration_dir) / f'FLATFIELD_{self.channel}_POSTBURN.txt')
        


        if self.channel == 'FUV' :
            # Adjust sensitivity to account for elimination
            # of evil pixels in original calibration
            sensitivity       /= 0.91
            sensitivity_error /= 0.91

            # Adjust flat field normalization to account
            # for asymmetry in histogram distribution
            ff *= 1.05
            
            # Row 2 and row 61 in the FUV flat-field corrector
            # appear erroneous. For now eliminate the corrector by setting to 1
            ff[2,:]  = 1
            ff[61,:] = 1
        
        # Apply flat-field
        if not flat_field : ff[~np.isnan(ff)] = 1
        sensitivity       /= ff
        sensitivity_error /= ff

        # Flatfield modifier
        arrmod = get_ff_time_variation(self.channel, sctime)
        if not flat_field : arrmod[~np.isnan(arrmod)] = 1
        sensitivity       *= arrmod
        sensitivity_error *= arrmod

        self.evil_pixels = np.isnan(sensitivity)



        # -- BINNING
        SPA_UL  = self.pds_data[0].qube.UL_CORNER_LINE
        SPA_LR  = self.pds_data[0].qube.LR_CORNER_LINE
        SPE_UL  = self.pds_data[0].qube.UL_CORNER_BAND
        SPE_LR  = self.pds_data[0].qube.LR_CORNER_BAND
        SPATIAL_BIN = self.pds_data[0].qube.LINE_BIN
        SPECTRA_BIN = self.pds_data[0].qube.BAND_BIN

        # Extract the illuminated window
        WL_win       = WL[SPE_UL:SPE_LR + 1]
        sens_win     = sensitivity       [SPA_UL: SPA_LR+ 1, SPE_UL:SPE_LR + 1]
        sens_err_win = sensitivity_error [SPA_UL: SPA_LR+ 1, SPE_UL:SPE_LR + 1]
        
        width  = WL_win.shape[0]
        height = sens_win.shape[0]
        
        # Adjust sizes to be multiples of bin sizes
        width_trim  = math.ceil(width  / SPECTRA_BIN) * SPECTRA_BIN
        height_trim = math.ceil(height / SPATIAL_BIN) * SPATIAL_BIN

        # Create indices for binning
        spe_indices = np.arange(0, width_trim,  SPECTRA_BIN)
        spa_indices = np.arange(0, height_trim, SPATIAL_BIN)

        # Bin WL_win
        WL = np.add.reduceat(WL_win, spe_indices) / SPECTRA_BIN

        # Bin sensitivity and sensitivity_error
        sensitivity       = np.add.reduceat(
                            np.add.reduceat(sens_win, spa_indices, axis=0),
                                                      spe_indices, axis=1)

        sensitivity_error = np.add.reduceat(
                            np.add.reduceat(sens_err_win, spa_indices, axis=0),
                                                          spe_indices, axis=1
        ) / np.sqrt(SPATIAL_BIN * SPECTRA_BIN)

        self.evil_pixels_binned = np.isnan(sensitivity)
        # -- FINAL CALIBRATION
        return {'calibration'       : 1/sensitivity,
                'calibration_error' : sensitivity_error/(sensitivity**2)}

    def set_calibration(self, **kwargs) :
        """
        Set calibration factors for all exposures.

        This method computes and assigns calibration factors and uncertainties for each exposure
        using the get_calibration method.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to get_calibration.
        """
        
        for i in range(self.n_pics) :
            cal = self.get_calibration(self.sctime_sec_start[0] + self.ET_start[i] - self.ET_start[0]
                                       , **kwargs)
            
            self.calibration[i]       = cal['calibration']
            self.calibration_error[i] = cal['calibration_error']
        self.calibration_set = True

    def calibrate(self, wl_interp: Literal["pchip","linear"] = 'pchip', nan_interp: Literal["linear","pchip"] = 'linear', flat_field: bool = True, smooth: bool = True) :
        """
        Calibrate counts/s to radiance and populate `data` and uncertainty arrays.

        Parameters
        ----------
        wl_interp : {"pchip","linear"}
            Interpolation used when building calibration sensitivity vs wavelength.
        nan_interp : {"linear","pchip"}
            Interpolation used to fill NaNs in calibrated spectra.
        flat_field : bool
            Apply flat-field and its time-dependent modifier.
        smooth : bool
            Apply spectral smoothing.
        """

        print('Applying calibration...', end='')
        
        if not self.calibration_set : self.set_calibration(flat_field=flat_field, interp=wl_interp)

        # Put evil pixels ncertainties
        self.uncertainty_sup, self.uncertainty_inf = self.get_radiance_uncertainty()

        self.uncertainty_sup[:,self.evil_pixels_binned] = np.nan
        self.uncertainty_inf[:,self.evil_pixels_binned] = np.nan

        for i in range(self.n_pics) :
            self.data[i]            = interpolate_nans(self.cps_bg_removed[i] * self.calibration[i], method=nan_interp)
            self.uncertainty_sup[i] = interpolate_nans(self.uncertainty_sup[i], method=nan_interp)
            self.uncertainty_inf[i] = interpolate_nans(self.uncertainty_inf[i], method=nan_interp)
        
        if smooth : self.smooth(force=True)
        self.is_calibrated = True
        print(' Done')


    # -------- GEOMETRY
    def get_geometry(self, ET:float, **kwargs) -> 'Geometry':
        """
        Compute the geometry for a given ephemeris time.

        Parameters
        ----------
        ET: float
            Ephemeris time for which to compute the geometry.
        **kwargs
            Additional keyword arguments for the Geometry class.

        Returns
        -------
        Geometry
            A Geometry object computed for the given time.
        """
        from .geometry import Geometry

        return Geometry( ET, u=self, **kwargs)

    def set_geometry(self, et_range: ArrayLike = None, **kwargs) :
        """
        Compute and set geometry for a range of exposures.

        Parameters
        ----------
        et_range : array_like, optional
            Array of ephemeris times for which to compute geometry. If None, the middle time of each exposure is used.
        **kwargs
            Additional keyword arguments passed to the geometry computation.

        Notes
        -----
        This method updates the geometry attribute and computes the mean heliocentric distance (HD)
        and line-of-sight pixel data.
        """
        
        self.spacecraft_position = []
        HD        = []
        pixel_LOS = []
        pixel_star = []
        n_used_pixels = []
        if et_range is None : et_range = self.ET_middle

        for i in tqdm(range(len(et_range)), desc="Computing geometry", file=sys.stdout):#, ncols=100) :
            et = et_range[i]
            g = self.get_geometry(et, **kwargs)
            HD.append(g.HD)
            self.spacecraft_position.append(g.spacecraft_position)
            pixel_LOS.append(g.used_pixels_LOS)
            pixel_star.append(g.pixel_stars)

            n_used_pixels.append(g.n_used_pixels)
        

        self.HD = np.mean(HD)
        self.spacecraft_position = np.array(self.spacecraft_position)
        self.pixel_LOS = np.array(pixel_LOS)
        

        dtype = np.dtype([
            ('MAG',     float),
            ('is_UV',   bool ),
            ('number',  int  ),
            ('on_disk', bool  )
        ])

        n_pixels = n_used_pixels[0]
        self.pixel_star_geometry = [
            (
                pixel_star[i][j]["MAG"],
                pixel_star[i][j]["is_UV"],
                pixel_star[i][j]["number"],
                pixel_star[i][j]["on_disk"]
            )
            for i in range(self.n_pics)
            for j in range(n_used_pixels[i])
        ]

        self.pixel_star_geometry = np.array(
            self.pixel_star_geometry, dtype=dtype
            ).reshape(self.n_pics, n_pixels)
        
    def plot_all_geometry(self, folder: str | Path, out_format: Literal["png","gif"] = 'png', duration: float = 1/60, **kwargs) :
        """
        Plot geometry for all exposures and save the results.

        Parameters
        ----------
        folder : str or Path
            Directory in which to save the plots or GIF.
        out_format : str, optional
            Output format ('png' for individual images or 'gif' for animation). Default is 'png'.
        duration : float, optional
            Duration per frame for GIF animation (in seconds). Default is 1/60.
        """

        import matplotlib.pyplot as plt
        from PIL import Image

        folder = Path(folder)
        if not folder.exists() :
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Creating directory: {folder}")


        if out_format=='gif':
            # Case: GIF – assemble all images in memory
            frames = []
            for i, obj in tqdm(enumerate(self.geometry), total=len(self.geometry), desc="Rendering geometry animation"):
                buf = io.BytesIO()
                obj.plot(save=True, savename=buf, **kwargs)
                buf.seek(0)

                im = Image.open(buf).convert("RGBA")
                frames.append(im)
                buf.close()

            gif_filename = folder / f"{self.name}.gif"

            frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration,
            disposal=2           # important for transparency
            )
            print(f"GIF created : {gif_filename}")
        else :
            # Standard case: save each plot as an individual file
            for i, obj in tqdm(enumerate(self.geometry), total=len(self.geometry), desc="Rendering geometry plots"):
                filename = folder / f"geometry_{i}.{out_format}"
                obj.plot(save=True, savename=str(filename), show=False, **kwargs)
            
            plt.close('all')


    # -------- UV PICTURE

    def UV_picture(self, wl_range: tuple[float, float] | None = None, **kwargs):
        """
        Build and plot a projected UV radiance map for this observation.

        This is a convenience wrapper around :func:`cassini_upyp.geometry.UV_picture.UV_picture`.
        See that function for the full parameter list and plotting options.

        Parameters
        ----------
        wl_range : tuple[float, float] or None, optional
            Wavelength interval passed to the UV map builder.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`cassini_upyp.geometry.UV_picture.UV_picture`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the UV map.
        matplotlib.axes.Axes
            Axes containing the UV map.
        """

        from .geometry.UV_picture import UV_picture
        return UV_picture(self, wl_range=wl_range, **kwargs)


    # -------- STARS IDENTIFICATION
    def add_pixel_stars_from_file(self, file: str | Path):
        """
        Add star contamination data from a file.

        Parameters
        ----------
        file : str
            Path to the file containing star pixel information.
            
        Notes
        -----
        The file should have a header followed by lines with two values representing pixel indices.
        """

        with open(file, 'r') as f :
            stars_pixels = f.readlines()[1:]

        stars_pixels = [tuple(e.split()) for e in stars_pixels]

        for i,j in stars_pixels :
            if not self.pixel_stars_mask[int(j), int(i)] :
                self.pixel_stars.append((int(i), int(j)))
                self.pixel_stars_mask[int(j), int(i)] = True

    def plot_radiance_evolution(
            self,
            output_path: str | Path = None,
            ylim: tuple[float, float] = (0.01,20),
            yscale: Literal['log', 'linear'] = 'log',
            wl_range: tuple[float, float] = (1600,1900),
            method: Literal['simpson', 'trapz', 'trapezoid'] = 'trapezoid'
        ):
        """
        Plot the evolution of integrated radiance for each pixel over exposures.

        Parameters
        ----------
        output_path : str | Path, optional
            File path to save the PDF containing the plots. Default is 'signal_time_variation.pdf'.
        ylim : tuple[float, float], optional
            Y-axis limits for the plots. Default is (0.01, 20).
        yscale : str, optional
            Scale for the y-axis ('log' or 'linear'). Default is 'log'.
        wl_range : tuple[float, float], optional
            Wavelength range for integration. Default is (1600, 1900).
        method : {'simpson', 'trapz'}, optional
            Integration method to use. Default is 'simpson'.
        """

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        integrated_radiance = self.integrate_radiance(wl_range=wl_range, method=method)

        if output_path is None : output_path = 'signal_time_variation.pdf'

        x_values = np.arange(self.n_pics)
        star_mask_global = (self.pixel_star_geometry['number'] > 0)

        with PdfPages(output_path) as pdf:
            for row in range(self.n_pixels):
                fig, ax = plt.subplots(figsize=(0.22*self.n_pics if self.n_pics>10 else 2.2, 4))
                
                
                y_values = integrated_radiance[:, row]
                
                ax.step(x_values, y_values, where='mid')
                ax.set_title(f"Pixel {row}")
                ax.set_xlabel("Exposure index")
                ax.set_ylabel("Integrated radiance (kR)")

                ax.set_ylim(*ylim)
                ax.set_yscale(yscale)

                ax.grid()

                star_cols = np.where(star_mask_global[:, row])[0]
                
                for col in star_cols:
                    is_uv = self.pixel_star_geometry['is_UV'][col, row]
                    mag_val = self.pixel_star_geometry['MAG'][col, row]
                    color = 'purple' if is_uv else 'darkgoldenrod'
                    
                    ax.axvline(x=col, color=color, linestyle='--', alpha=0.7)

                    ax.plot(col, y_values[col], marker='o', color=color)

                    ax.text(
                        col+0.5, y_values[col]* 10**0.1 if yscale=='log' else y_values[col] + 0.1,
                        f"{mag_val:.2f}",
                        color=color,
                        ha='center', va='bottom',
                        fontsize=8, rotation=270,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
                    )

                fig.tight_layout()
                
                pdf.savefig(fig)
                plt.close(fig)


    def check_stars(
            self,
            cmap: str = 'gist_ncar',
            color_scale: tuple[float, float] = (0,14),
            wl_range: tuple[float, float]    = (1600,1900),
            method: Literal['simpson', 'trapz', 'trapezoid'] = 'trapezoid',
            exp_range: tuple[int, int] = None
        ):
        """
        Create a heatmap of integrated radiance similar to a chronophotography and highlight pixels affected by stars.
        The heatmap is interactive for the user to identify pixels as contaminated (by stars, other objects, or
        any pixel to remove from analysis).

        Parameters
        ----------
        cmap : str, optional
            Colormap to use for the heatmap. Default is 'gist_ncar'.
        color_scale : tuple of float, optional
            Color scale limits. Default is (0, 14).
        wl_range : tuple of float, optional
            Wavelength range for integration. Default is (1600, 1900) in angströms.
        method : {'simpson', 'trapz', 'trapezoid'}, optional
            Integration method to use. Default is 'simpson'.
        exp_range : tuple of int, optional
            Range of exposures to display (start, end). If None, all exposures are shown.

        See Also
        --------
        :func:`cassini_upyp.uvisdata.UVIS_Observation.integrate_radiance` : Compute integrated radiance used in this plot.
        :func:`cassini_upyp.geometry.UV_picture.UV_picture` : Create a UV image of the observation.
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.patches import Rectangle
        from matplotlib import pyplot as plt

        integrated_radiance = self.integrate_radiance(wl_range=wl_range, method=method)
        if exp_range is None:
            exp0, exp1 = 0, self.n_pics
        else:
            exp0, exp1 = exp_range
            exp0 = 0 if exp0 is None else int(exp0)
            exp1 = self.n_pics if exp1 is None else int(exp1)
            exp0 = max(0, min(self.n_pics, exp0))
            exp1 = max(0, min(self.n_pics, exp1))
            if exp1 <= exp0:
                raise ValueError(f"Invalid exp_range={exp_range}. Expected (start, end) with 0 <= start < end <= {self.n_pics}.")

        n_exp = exp1 - exp0
        integrated_radiance = integrated_radiance[exp0:exp1, :]
        X, Y = np.arange(n_exp + 1) - 0.5, np.arange(self.n_pixels + 1) - 0.5

        fig     = plt.figure(figsize=(8, 6))
        ax      = fig.add_axes([0.1, 0.1, 0.6, 0.8])
        ax_text = fig.add_axes([0.75, 0.1, 0.2, 0.8])
        ax_text.axis('off')


        if color_scale is None :
            mesh = ax.pcolormesh(X,Y, integrated_radiance.T, edgecolors='k', linewidth=0.5, cmap=cmap)
        else :
            mesh = ax.pcolormesh(X,Y, integrated_radiance.T, edgecolors='k', linewidth=0.5, cmap=cmap, vmin=color_scale[0], vmax=color_scale[1])

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.1, pad=0.15)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Integrated radiance (kR)", rotation=270, labelpad=15)

        # Ticks definition
        if n_exp < 10:
            xticks = np.arange(n_exp)
        else:
            xticks = np.arange(0, n_exp, step=2)
        
        yticks = np.arange(self.n_pixels)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks + exp0)
        ax.set_yticks(yticks)
        ax.set_aspect('equal')
        ax.set_xlabel("Exposure Number")
        ax.set_ylabel("Pixel Number")
        ax.set_zorder(2)
        cax.set_zorder(1)


        # ADD GEOMETRY INFO
        if self.pixel_LOS is not None :
            
            # DISK
            mask_ij = np.all(self.pixel_LOS[exp0:exp1, :]['alt'] < 0, axis=2)
            for xuv, yuv in np.argwhere(mask_ij):
                ax.plot([xuv], [yuv], color="#000000", ls='', marker='o', markersize=2)
            
            if np.any(self.pixel_star_geometry[exp0:exp1, :]['number']>0) :
                index  = np.where((self.pixel_star_geometry[exp0:exp1, :]['number']>0)*(~self.pixel_star_geometry[exp0:exp1, :]['is_UV']))
                result = list(zip(index[0], index[1]))

                for xuv,yuv in result :

                    rect = Rectangle((xuv - 0.5, yuv - 0.5), 1, 1, 
                     edgecolor='yellow', linewidth=1, facecolor='none')
                    ax.add_patch(rect)
        
        # INTERACTION HANDLES
        for (row, col) in self.pixel_stars:
            if not (exp0 <= col < exp1):
                continue
            col_local = col - exp0
            x_center = (X[col_local] + X[col_local+1]) / 2
            y_center = (Y[row] + Y[row+1]) / 2
            line, = ax.plot(x_center, y_center, marker='x', color='black', markersize=8, mew=2)
            self.markers[(row, col)] = line
        text_handle = ax_text.text(0, 1, "Selected pixels:\n", va='top', fontsize=10,family='monospace')



        def update_text():
            header = f"{'Pixel':>6} {'Exposure':>10}\n"
            if self.pixel_stars:
                lines = "\n".join(f"{int(row):>6} {int(col):>10}" for row, col in self.pixel_stars)
                text_str = "Selected pixels:\n"  + header + lines
            else:
                text_str = "Selected pixels:\n"+ header
            
            text_handle.set_text(text_str)
            plt.draw()

        def add_marker(row, col):
            if not (exp0 <= col < exp1):
                return
            col_local = col - exp0
            x_center = (X[col_local] + X[col_local+1]) / 2
            y_center = (Y[row] + Y[row+1]) / 2
            line, = ax.plot(x_center, y_center, marker='x', color='black', markersize=8, mew=2)
            self.markers[(row, col)] = line

        def remove_marker(row, col):
            if (row, col) in self.markers:
                self.markers[(row, col)].remove()
                del self.markers[(row, col)]

        def on_click(event):
            if event.inaxes == ax:
                x_click = event.xdata
                y_click = event.ydata

                col_local = np.searchsorted(X, x_click) - 1
                row = np.searchsorted(Y, y_click) - 1
                if not (0 <= row < self.n_pixels and 0 <= col_local < n_exp):
                    return
                col = col_local + exp0

                if (row, col) in self.pixel_stars:
                    self.pixel_stars.remove((row, col))
                    self.pixel_stars_mask[col, row] = False
                    remove_marker(row, col)
                else:
                    self.pixel_stars.append((row, col))
                    self.pixel_stars_mask[col, row] = True
                    add_marker(row, col)

                update_text()

        hover_annotation = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontfamily="monospace"
        )
        hover_annotation.set_visible(False)

        def on_hover(event):
            # Vérifie si la souris est dans la zone de l’axe principal
            if event.inaxes == ax:
                x_hover = event.xdata
                y_hover = event.ydata

                # Trouver l’indice du pixel survolé
                col_local = np.searchsorted(X, x_hover) - 1
                row = np.searchsorted(Y, y_hover) - 1

                # Vérifier qu’on est bien dans la grille
                if 0 <= row < self.n_pixels and 0 <= col_local < n_exp:
                    col = col_local + exp0
                    # Exemple: récupérer la valeur du signal
                    pixel_value = integrated_radiance[col_local, row]

                    # Construire la chaîne de texte à afficher
                    if self.pixel_LOS is None:
                        text = (
                            f"Pixel: {row},  Exposure: {col}\n"
                            f"Signal: {pixel_value:.2f} kR"
                        )
                    else:
                        alt_center, alt_min, alt_max=self.pixel_LOS[col,row,0]['alt'], min(self.pixel_LOS[col,row,:]['alt']), max(self.pixel_LOS[col,row,:]['alt'])
                        text = (
                            f"Pixel      : {row},  Exposure: {col}\n"
                            f"Signal     : {pixel_value:.2f} kR\n"
                            f"Altitude   : {alt_center:.0f} km ({alt_min:.0f} km <-> {alt_max:.0f} km)\n"
                            f"Local time : {self.pixel_LOS[col,row,0]['lt']:.1f}\n"
                            f"SZA        : {self.pixel_LOS[col,row,0]['sza']:.1f}°\n"
                            f"Latitude   : {self.pixel_LOS[col,row,0]['lat']:.1f}°\n"
                        )

                    # Mise à jour de l’annotation
                    hover_annotation.xy = (col_local, row)
                    hover_annotation.set_text(text)
                    hover_annotation.set_visible(True)
                    # Redessiner la figure
                    plt.draw()
                else:
                    hover_annotation.set_visible(False)
                    plt.draw()
            else:
                # Si la souris est hors de l'axe, on cache l’annotation
                hover_annotation.set_visible(False)
                plt.draw()


        # Connect callbacks
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        fig.canvas.mpl_connect('button_press_event', on_click)
        update_text()
        plt.show()
    
    def save_stars(self, filepath: str | Path) -> str:
        """
        Save the list of stellar-contaminated pixels to a text file.

        Parameters
        ----------
        filepath : str or Path
            Output path for the star pixel list.

        Returns
        -------
        str or Path
            The original `filepath` argument.
        """

        p = Path(filepath)

        print(f"Saving UVIS observation star pixel list {p.stem}...", end='', flush=True)

        with p.open('w') as f:
            f.write("# exposure | pixel\n")
            for (i, j) in self.pixel_stars:
                f.write(f"  {i:<2}         {j:<2}\n")
        
        print(' Done')

        return filepath
    
        

    # -------- BACKGROUND NOISE
    def get_background(
            self,
            alt_limit: float,
            mode: Literal['average', 'simulate'] = 'simulate',
            wl_range: tuple[float, float] = (1400,1850),
        ) -> tuple[float, float]:
        """
        Compute the background noise level and its uncertainty from the raw detector counts.

        This method selects background pixels based on altitude criteria from the geometry (using self.pixel_LOS)
        and calculates the average counts per second over a specified wavelength range. It supports two modes:
        - 'average': simply average the counts of background pixels.
        - 'simulate': perform a simulation by fitting a histogram of gaps in the counts, optionally in parallel.

        Parameters
        ----------
        mode : {'average', 'simulate'}, optional
            The method to compute the background. Default is 'simulate'.
        alt_limit : float
            Minimum altitude limit to consider a pixel as background.
        wl_range : tuple of int, optional
            Wavelength range (min, max) to consider for background calculation. Default is (1600, 1900).
        parallel : bool, optional
            Whether to perform the simulation in parallel processing. Default is True.

        Returns
        -------
        tuple of float
            A tuple (cps, cps_err) representing the background counts per second and its uncertainty.

        Raises
        ------
        ValueError
            If the specified wavelength range is outside the detector range or if geometry is not initialized.
        """

        from .background import histogram, gaps, max_gap, bg_fit

        if not (self.WL[0]<wl_range[0]<self.WL[-1] and self.WL[0]<wl_range[1]<self.WL[-1]) :
            raise ValueError(f'Please select a wl_range within the {self.channel} channel.')

        if self.pixel_LOS is None:
            raise ValueError("The observation geometry must be initialized before determining the background.")

        
        MinPixAlt = np.min(self.pixel_LOS['alt'], axis=2)
        self.n_bg_pixels = np.sum(MinPixAlt>alt_limit)

        if self.n_bg_pixels==0 :
            print(f'No background pixels available above {alt_limit} km, please manually set a background level.')
            return np.nan, np.nan



        # Basic averaging method
        bg_pixels = self.counts[(MinPixAlt>alt_limit) * ~self.pixel_corrupted * ~self.pixel_stars_mask, :]
        bg_pixels = bg_pixels[:, (self.WL>=wl_range[0]) * (self.WL<=wl_range[1])]
        bg_pixels = bg_pixels[np.any(bg_pixels, axis=1)] # Filter total transmission losses
        counts_per_pixel = np.sum(bg_pixels, axis=1)


        S = bg_pixels.shape[0]                                       # N samples
        L = np.sum((self.WL>=wl_range[0]) * (self.WL<=wl_range[1]))  # N spectral pixels
        
        #         total number of counts      / # of spectral pixels                    / exposure time
        cps     = np.mean(counts_per_pixel)  /  (L   * self.spat_bin*self.spec_bin)    /  self.expo_time
        N_tot   = np.sum(counts_per_pixel)
        cps_err = np.sqrt(N_tot)           /    (L*S * self.spat_bin*self.spec_bin)  /    self.expo_time

        wl_index = (
            np.where(self.WL>=wl_range[0])[0][ 0],
            np.where(self.WL<=wl_range[1])[0][-1]+1
        )

        # Flag corrupted pixels (transmission losses)
        self.max_gap = max_gap(cps, self.expo_time, n_wl = self.n_wl, SPECTRAL_BIN=self.spec_bin, SPATIAL_BIN = self.spat_bin)
        if self.max_gap <10 and self.spat_bin>1 : self.max_gap = 100
        for i_pic in range(self.counts.shape[0]) :
            for i_spat in range(self.counts.shape[1]) :

                # Total losses
                if not np.any(self.counts[i_pic,i_spat, :]) :
                    self.pixel_corrupted[i_pic,i_spat] = True

                # Lyman-alpha losses
                if self.channel=='FUV':
                    if not np.any(self.counts[i_pic,i_spat, 122//self.spec_bin:137//self.spec_bin]) :
                        self.pixel_corrupted[i_pic,i_spat] = True

                # Maximum gap on detector
                obs_gaps = gaps(self.counts[i_pic,i_spat, :])
                if obs_gaps is not None and max(obs_gaps) > self.max_gap :
                    self.pixel_corrupted[i_pic,i_spat] = True
                



        


        if mode=='simulate' :
            bg_pixels = self.counts[(MinPixAlt>alt_limit) * ~self.pixel_corrupted * ~self.pixel_stars_mask, :]
            bg_pixels = bg_pixels[:, (self.WL>=wl_range[0]) * (self.WL<=wl_range[1])]

            S = bg_pixels.shape[0]                                       # N samples
            L = np.sum((self.WL>=wl_range[0]) * (self.WL<=wl_range[1]))  # N spectral pixels

            # Build observations histogram
            H = np.zeros(L+1)
            for pixel in bg_pixels :
                H += histogram(
                    gaps(pixel), max_value=L
                    )
            H /= S

            # Perform fits
            N = bg_fit(H, detector_shape=L)

            cps = N / (L * self.spec_bin * self.spat_bin * self.expo_time)

            q = (1 - 1/L)**N            
            sigma_q = np.sqrt(q*(1-q)/(L*S))
            sigma_N = np.abs(1/(q*np.log(1-1/L))) * sigma_q
            cps_err = sigma_N / (L * self.spec_bin * self.spat_bin * self.expo_time)
        
            if cps > 1e-3 : 
                warnings.warn(
                    f"Background level is unusually high: {cps:.2e} counts/s.",
                    RuntimeWarning
                )
        return cps, cps_err


    def set_background(self, bg: float = None, bg_uncertainty: float = None, **kwargs) :
        """
        Set (or estimate) the detector background level and update background-corrected CPS.

        If `bg` and `bg_uncertainty` are not provided, estimates them with `get_background(**kwargs)`.
        Then updates `self.cps_bg_removed`.

        Parameters
        ----------
        bg : float, optional
            Background level in counts/s per spectral pixel (before applying UVIS binning).
        bg_uncertainty : float, optional
            1-sigma uncertainty on `bg`, same units as `bg`.
        **kwargs
            Passed to `get_background()` when estimating the background.

        Notes
        -----
        `self.cps_bg_removed` is computed as:
            cps_bg_removed = cps - bg * SPATIAL_BIN * SPECTRAL_BIN

        If the observation is already calibrated, this method refreshes the calibrated
        data to reflect the updated background.
        """

        if bg is None and bg_uncertainty is None :
            self.background_level, self.background_error = self.get_background(**kwargs)
        else :
            if bg             is not None : self.background_level = bg
            if bg_uncertainty is not None : self.background_error = bg_uncertainty
        self.cps_bg_removed = self.cps-self.background_level*self.spat_bin*self.spec_bin
        if self.is_calibrated : self.calibrate()
        self.is_bkg_removed = True



    # -------- BINNING
    def bin_pixels(
        self,
        pixels: Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]] | None = None,
        keys: tuple[str, ...] = ('lat', 'alt','lt'),
        bin_boundaries: tuple[Sequence[float], ...] = (
            default_lat_bins,
            default_alt_bins,
            [0,12,24]
        ),
        mode: Literal['center', 'all'] = 'center',
        modulo: float | None = None,
    ) -> 'UVIS_Bin':
        """
        Bin detector pixels either automatically using geometric criteria or manually
        using explicit pixel indices.

        Two mutually exclusive binning modes are supported:

        1) Automatic (geometric) binning
        Pixels are assigned to bins based on geometric quantities stored in
        ``pixel_LOS`` (e.g. latitude, altitude, local time), using user-defined
        bin boundaries.

        2) Manual binning
        Pixels are grouped explicitly by providing their detector indices
        ``(i_pic, i_pix)``. In this mode, no geometric selection is performed.

        Parameters
        ----------
        pixels : sequence or None, optional
            Manual pixel selection. If None, automatic geometric binning is used.

            Accepted formats are:

            - ``[(i, j), (i, j), ...]`` : a single bin containing all listed pixels.
            - ``[[(i, j), ...], [(i, j), ...], ...]`` : multiple bins, each defined by its own list of pixels.

            Indices are interpreted as ``(exposure_index, pixel_index)``.

        keys : tuple of str, optional
            Geometric quantities used for automatic binning. Each key must correspond
            to a field in ``pixel_LOS``.

            Accepted values are:

            - ``'lon'``   : longitude of the tangent point of the line of sight
            - ``'lat'``   : latitude
            - ``'alt'``   : altitude
            - ``'sza'``   : solar zenith angle
            - ``'phase'`` : phase angle
            - ``'ems'``   : emission angle
            - ``'lt'``    : local time

        bin_boundaries : tuple of sequences, optional
            Bin boundaries for each geometric key. The number of boundary arrays
            must match the number of keys.
            Bin boudaries must be monotonically increasing and cover the range of values in the data.
            Modular quantities (e.g. longitude, local time) are handled with wrap-around logic.

        mode : {'center', 'all'}, optional
            Selection mode for geometric binning:

            - ``'center'`` : use the central LOS value of each pixel.
            - ``'all'``    : require all LOS samples of the pixel to fall within
                            the bin boundaries.

        modulo : float, optional
            Modulo value for wrap-around of geometric quantities (e.g. 360 for longitude, 24 for local time).
            Required if any of the keys are modular.

        Returns
        -------
        UVIS_Bin
            A ``UVIS_Bin`` instance containing the binned pixel indices and,
            if geometry is available, the mean geometric properties per bin.

        Raises
        ------
        RuntimeError
            If automatic binning is requested but ``pixel_LOS`` is not set.
        ValueError
            If input formats are invalid or inconsistent.

        Examples
        --------
        Automatic geometric binning by altitude and latitude:

            bins = uvis_obs.bin_pixels(
                keys=('alt', 'lat'),
                bin_boundaries=(
                    np.arange(500, 1000, 50),   # altitude bins (km)
                    np.linspace(-90, 90, 19)    # latitude bins (deg)
                ),
                mode='center'
            )
        Automatic geometric binning by longitude:
            bins = uvis_obs.bin_pixels(
                keys=('lon',),
                bin_boundaries=(
                    [350, 10, 30], # Two bins: [350-360] U [0-10], and [10-30] degrees
                ),
                modulo = 360
                )
        
        Manual binning of explicitly selected pixels into a single bin:

            bins = uvis_obs.bin_pixels(
                pixels=[(20, 39), (21, 40), (22, 41)]
            )

        Manual binning into multiple independent bins:

            bins = uvis_obs.bin_pixels(
                pixels=[
                    [(20, 39), (21, 40)],
                    [(10, 5), (10, 6), (10, 7)]
                ]
            )
        """
        
        print('Creating bins...', end='', flush=True)

        # GEOMETRIC MODE
        if pixels is None:
            # Error checks
            if self.pixel_LOS is None:
                raise RuntimeError("pixel_LOS is not set. Call set_geometry() first.")
            if len(keys) != len(bin_boundaries):
                raise ValueError("The number of 'keys' must match the number of 'bin_boundaries' sets.")
            if mode not in ("center", "all"):
                raise ValueError("mode must be 'center' or 'all'")
            available = set(self.pixel_LOS.dtype.names)
            missing = [k for k in keys if k not in available]
            if missing:
                raise ValueError(f"Unknown LOS keys: {missing}. Available: {sorted(available)}")


            # Initialize bins
            shape = tuple(len(bounds) - 1 for bounds in bin_boundaries)
            bins = UVIS_Bin(shape, self)
            bins.bin_def = {key:bin_boundary for key,bin_boundary in zip(keys, bin_boundaries)}

            for i_pic in range(self.n_pics):
                for i_pix in range(self.n_pixels):
                    if i_pix==59 : continue
                    
                    if self.pixel_corrupted[i_pic,i_pix] or self.pixel_stars_mask[i_pic,i_pix] : continue

                    if   mode == 'center':
                        pixel_properties = [self.pixel_LOS[key][i_pic, i_pix, 0] for key in keys]
                    elif mode == 'all':
                        pixel_properties = [self.pixel_LOS[key][i_pic, i_pix, :] for key in keys]

                    bin_indices = []
                    valid = True
                    
                    # Determine the bin index for each property.
                    for dim_idx, prop in enumerate(pixel_properties):
                        idx = find_bin_index(prop, bin_boundaries[dim_idx], mode)
                        if idx is None:
                            valid = False
                            break
                        bin_indices.append(idx)

                    # If the pixel is valid in all dimensions, add it to the corresponding bin.
                    if valid:
                        # Use tuple indexing to access the cell in the NumPy array.
                        bins.bins[tuple(bin_indices)].append((i_pic, i_pix))
                        bins.number_per_bin[tuple(bin_indices)] += 1

        #MANUAL MODE
        else:
            if len(pixels) == 0:
                raise ValueError("Pixel list is empty.")
            # Determine the number of dimensions
            first_bin = np.asarray(pixels[0])

            if first_bin.ndim == 1 and first_bin.shape == (2,):
                # single bin: [(i,j), (i,j), ...]
                groups = [np.asarray(pixels, dtype=int)]

            elif first_bin.ndim == 2 and first_bin.shape[1] == 2:
                # multiple bins: [[(i,j), ...], [(i,j), ...], ...]
                groups = []
                for k, grp in enumerate(pixels):
                    arr = np.asarray(grp, dtype=int)
                    if arr.ndim != 2 or arr.shape[1] != 2:
                        raise ValueError(f"Bin {k} has shape {arr.shape}, expected (n_i, 2)")
                    groups.append(arr)

            else:
                raise ValueError(
                    "Invalid pixels list format. Expected pixels of indices (i, j):\n"
                    "- [(i,j), ...] for a single bin\n"
                    "- [[(i,j), ...], ...] for multiple bins"
                )
            
            bins = UVIS_Bin((len(groups),), self)
            for b, group in enumerate(groups):
                for i_pic, i_pix in group:
                    if not (0 <= i_pic < self.n_pics and 0 <= i_pix < self.n_pixels):
                        raise IndexError(f"Invalid pixel index {(i_pic, i_pix)}")
                bins.bins[b].extend(map(tuple, group))
                bins.number_per_bin[b] += len(group)

        # MEAN PIXEL GEOMETRIC PROPERTIES
        if self.pixel_LOS is not None:
            for idx in np.ndindex(bins.bins.shape):
                pairs = bins.bins[idx]
                if not pairs:  # Empty bin
                    continue
                for key in self.pixel_LOS.dtype.names:
                    bins.bin_LOS[idx][key] = np.mean([self.pixel_LOS[i, j,:][key] for (i, j) in pairs])

        print(' Done')
        return bins


        

    # -------- SAVE MANAGMENT
    def save(self, filepath: str = None, overwrite: bool = False):
        """
        Saves the current UVIS_Observation instance to a pickle (.pkl) file.
        The object is save without self.geometry attribute unless keyword fullsave
        is set.

        Parameters
        ----------
        filepath  : str, optional
            Path of the output file. Defaults to "<self.name>.pkl".
        overwrite : bool, optional
            If True, overwrites an existing file without asking.
            Defaults to False.
        Returns
        -------
        str
            The final filepath of the saved pickle file.

        Raises
        ------
        PermissionError
            If the file cannot be written due to permission issues.
        OSError
            For other I/O-related errors.
        """

        if filepath is None: filepath = f"{self.name}.uvis"


        p = Path(filepath)
        if p.suffix.lower() != '.uvis':
            p = p.with_suffix('.uvis')

        print(f"Saving UVIS observation object {p.stem}...", end='', flush=True)

        if p.exists() and not overwrite:
            response = input(f"File '{p.absolute()}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response not in ('y', 'yes', 'o', '1', 'oui'):
                print("Save cancelled.")
                return

        with p.open('wb') as f:
            pickle.dump(self, f)
        
        print(' Done')
        return filepath



    def save_JSON(self, filepath:str=None, overwrite=False):
        """
        Saves main attributes of the current UVIS observations to a JSON file.

        Parameters
        ----------
        filepath : str, optional
            The desired filepath for the JSON file. If not provided,
            defaults to "<self.name>.json".
        overwrite : bool, optional
            If True, overwrites the file if it already exists.
            Defaults to False.

        Returns
        -------
        str
            The final filepath of the saved JSON file.

        Raises
        ------
        PermissionError
            If the file cannot be written due to permission issues.
        OSError
            For other I/O related errors.
        """

        if filepath is None:
            filepath = f"{self.name}.json"
        
        p = Path(filepath)
        if p.suffix.lower() != '.json':
            p = p.with_suffix('.json')
        
        if p.exists() and not overwrite:
            response = input(f"File '{p}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response not in ('y', 'yes', 'o', '1', 'oui'):
                print("Save cancelled.")
                return

        data = {
            "CHANNEL"          : self.channel,
            "YEAR"             : self.YEAR,
            "DOY"              : self.DOY,
            "LEAD_INSTRUMENT"  : self.prime,
            "SLIT"             : self.slit,
            "FRAMES"           : self.n_pics,
            "SPATIAL_BIN"      : self.spat_bin,
            "SPECTRAL_BIN"     : self.spec_bin,
            "INTEGRATION_TIME" : int(self.expo_time),
            "BACKGROUND"       : {},
            "GEOMETRY"         : {}
        }

        if self.background_level is not None: data['BACKGROUND']['LEVEL']       = self.background_level
        if self.background_error is not None: data['BACKGROUND']['UNCERTAINTY'] = self.background_error
        if self.HD is not None: data['GEOMETRY']['SOLAR_DISTANCE'] = round(self.HD,3)
        if self.pixel_LOS is not None:
            data["GEOMETRY"]['PHASE_ANGLE'] = round(self.pixel_LOS['phase'].mean(), 2)
        

        p.write_text(
            json.dumps(data, indent=4, allow_nan=True), encoding='utf-8'
            )

        return filepath

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_JSON(cls, filepath: str):
        """
        Loads the specified JSON file and returns its content as a dictionary.

        Parameters
        ----------
        filepath : str
            The path to the JSON file to load.

        Returns
        -------
        dict
            A dictionary containing the data from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        PermissionError
            If the file cannot be opened due to permission issues.
        OSError
            For other I/O related errors.
        """

        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"JSON file '{filepath}' does not exist.")

        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)

        return data
    