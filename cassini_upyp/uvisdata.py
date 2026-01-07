from __future__ import annotations
from typing import Literal

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
from .background import bg_fit, do_bg_fit
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
    correction_factor
)

# CONFIG
from .config.pipeline_defaults import *
from .config.uvis import (
    pixel_bandpasses,
    slit_ratios,
    sctimeburn,
    slit_dlambda
)
from .utils import env_config, list_ndarray, find_bin_index
env = env_config()

def pds_lbl(labelfile:str) :
    """
    Parse a PDS3-formatted LBL (Label) file and return its contents as a nested dictionary.

    This function reads a PDS3 LBL file line by line, extracting key-value pairs and storing them in a dictionary.
    Multi-line values are concatenated into single strings, and numerical values with units (e.g., "<KM>", "<DEGREE>")
    are stored as strings for easy post-processing. The function also handles nested objects like QUBE structures
    by organizing them into sub-dictionaries for structured data access.

    Parameters
    ----------
    file_path : str
        The path to the LBL file to be parsed.

    Returns
    -------
    dict
        A nested dictionary containing the parsed content of the LBL file. Top-level keys represent main attributes,
        and sub-dictionaries are created for nested objects (e.g., "QUBE").

    Examples
    --------
    >>> lbl_data = read_lbl("path/to/your_file.lbl")
    >>> print(lbl_data["TARGET_NAME"])
    TITAN
    >>> print(lbl_data["QUBE"]["AXES"])
    3

    Notes
    -----
    - This parser is tailored for PDS3 LBL files and may not handle other formats like PDS4 or complex binary data.
    - Fields with multi-line values are concatenated; unit strings (e.g., "<KM>") remain as part of the values.
    - Single numerical values are automatically converted to float or integer
    - For complex objects like `QUBE`, the function creates sub-dictionaries with attributes accessible as nested keys.
    """

    label  = AttrDict({})
    obj    = label
    nested = False

    with open(labelfile, 'r') as f :
        for line in f :
            line = line.strip()
            if line.strip() =='END' : break

            # ^ at the begining indicates a nested object
            if line.startswith('^') :
                nested = True
            
            elif '=' in line and not line.startswith('^') :

                key, value = line.split('=', 1)
                key   = key.strip()
                value = value.strip(' "')

                # Convert UNK and N/A into None
                if 'UNK' in value or 'N/A' in value :
                    value = None
                # Convert what we can in numbers
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    try:
                        value = float(value)
                    except (ValueError, TypeError): pass
                

                if key == "OBJECT" and nested :
                    obj[value]=AttrDict({})
                    obj = obj[value]
                elif key == "END_OBJECT" :
                    nested=False
                    obj=label
                else : obj[key]  = value


            elif "=" not in line and not line.startswith('^') :
                if obj[key] == '' : obj[key] += line.strip('"')
                else : obj[key] += ' '+line.strip('"')
    return AttrDict(label)

def read_binary_pds(filename_dat, data_dims, data_type, endian='big'):
    """
    Read a binary PDS (Planetary Data System) data file and return its contents as a NumPy array.

    This function reads binary data from a PDS file and reshapes it into a data cube based on the specified dimensions.
    It handles the byte order (endianness) and data type to correctly interpret the binary data.

    Parameters
    ----------
    filename_dat : str
        The path to the binary PDS data file.
    data_dims    : tuple of ints
        A tuple containing the dimensions of the data in the order (BAND, LINE, SAMPLE).
    data_type    : str or numpy.dtype
        The data type of the binary data (e.g., 'float32', 'int16').
    endian       : str, optional
        The byte order of the data. Can be 'big' or 'little'. Default is 'big'.

    Returns
    -------
    numpy.ndarray or None
        A NumPy array containing the data reshaped into a cube with dimensions (SAMPLE, LINE, BAND).
        Returns `None` if there is an error in reading or reshaping the data.

    Notes
    -----
    - The function reads the binary data from the specified file, interprets it using the given data type and endianness,
      and reshapes it according to the provided dimensions.
    - The dimensions should be provided in the order (BAND, LINE, SAMPLE), but the resulting array will have the shape
      (SAMPLE, LINE, BAND).
    - If an error occurs during file reading or data reshaping, the function prints an error message and returns `None`.

    Examples
    --------
    Read a binary PDS file with specified dimensions and data type:

    >>> data_cube = read_binary_pds('data.dat', (3, 64, 1024), 'float32')
    >>> if data_cube is not None:
    ...     print(data_cube.shape)
    (1024, 64, 3)
    """

    # Détermine endianess
    dtype = np.dtype(data_type).newbyteorder(endian)

    BAND, LINE, SAMPLE = data_dims

    # Read the file as a data cube
    try:
        with open(filename_dat, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
            # Reshape the data according to the specified dimensions
            data_cube = data.reshape((SAMPLE, LINE, BAND))
    except Exception as e:
        print(f"Error reading : {e}")
        return None

    return data_cube


class AttrDict(dict):
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


class instrument(AttrDict):
    def __init__(self, instrument_name, n_pixels):
        self.name = instrument_name


        spice.furnsh(env.ik_path)
        self.ID = spice.bodn2c(instrument_name)
        
        

        # Get parameters in degree
        self.shape, self.frame, self.bsight, self.bounds, self.corners = spice.getfov(self.ID, 4)
        self.corners = np.array(self.corners)
        self.bsight  = np.array(self.bsight)


        fov_h_angle = spice.gdpool(f'INS{self.ID}_FOV_REF_ANGLE',   0, 1)[0] 
        fov_w_angle = spice.gdpool(f'INS{self.ID}_FOV_CROSS_ANGLE', 0, 1)[0]
        
        self.fov_height = 2 * fov_h_angle # In degrees
        self.fov_width  = 2 * fov_w_angle

        self.pixel_height = self.fov_height / n_pixels
        self.pixel_width  = self.fov_width

        spice.unload(env.ik_path)
        # COMPUTE PIXEL CORNERS IN INSTRUMENT FRAME
        #------------------------------------------
        # Array indices : center, b_l, b_r, u_r, u_l

        # Pixel corners in angles (theta, phi)
        pixels_angles = np.zeros((n_pixels,5,2))
        bc = [0, -fov_h_angle] # Bottom center point

        index = np.arange(64)

        # theta
        pixels_angles[:, :, 0] = np.array([0, -fov_w_angle, fov_w_angle, fov_w_angle, -fov_w_angle])
        # phi
        pixels_angles[:, 0,   1] = self.pixel_height * (.5 + index)            # Center
        pixels_angles[:, 1:3, 1] = self.pixel_height *       index [:, np.newaxis] # Bottom
        pixels_angles[:, 3:5, 1] = self.pixel_height * ( 1 + index)[:, np.newaxis] # Top
        pixels_angles += bc

        # Pixel corners in cartesian coordinates
        theta = pixels_angles[:, :, 0]*np.pi/180
        phi   = pixels_angles[:, :, 1]*np.pi/180

        pixels_corners = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(phi),
            np.cos(theta) * np.cos(phi)
        ], axis=-1)
        
        # Normalise vectors
        self.pixels_corners = pixels_corners / 1#np.linalg.norm(pixels_corners, axis=-1, keepdims=True)


class pds_raw_data:
    """
    A class to represent and process raw PDS (Planetary Data System) data from the Cassini UVIS instrument.

    This class stores raw data and metadata extracted from PDS files (.DAT and .LBL) and provides methods for
    calibrating the data according to the instrument's specifications and calibration files.

    Can be initialized from DAT and LBL file with read_pds() function.

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw counts data loaded from the PDS .DAT file.
    calibrated_data : numpy.ndarray
        UVIS spectra
    
    {attribute} : str or int or float
        Metadata from the .LBL file
    qube : pds_qube_lbl
        An object containing the QUBE metadata extracted from the .LBL file.
    channel : str
        The UVIS channel ('FUV' or 'EUV') determined from the NAME attribute in the label.
    pix_bandpass : numpy.ndarray
        The pixel bandpass values for the specified channel.
    slit_ratio : float
        The slit width ratio for the specified channel and slit state.
    sctime_sec_start : float
        The spacecraft clock time (in seconds) at the start of data acquisition.
    SPA_UL, SPA_LR, SPE_UL, SPE_LR : int
        Aliases for the sensor pixels' corner positions, as specified in the label.

    Methods
    -------
    get_calibration(interp='linear', flat_field=True)
        Retrieve the calibration multiplier for the raw data based on the instrument's sensitivity and time variation.

    Notes
    -----
    - The class automatically extracts and computes several attributes from the provided label data.
    - The `get_calibration` method calculates the calibration factors considering lab calibration, time variation,
      flat-field corrections, and binning.
    - FUV channel data may contain 'evil' pixels with anomalous behavior; these are handled during calibration.

    Examples
    --------
    Initialize a `pds_raw_data` object and retrieve calibration data:

    >>> data_pds = pds_raw_data(data, label)
    >>> calibration = data_pds.get_calibration()

    Initialize a `pds_raw_data` object from binary data and label file:
    >>> data_pds = read_pds('FUV2006_015_14_47')
    """

    def __init__(self, filename:str=None, file2:str=None, no_extract=False) :
        """
        Read PDS (Planetary Data System) raw files from the Cassini UVIS instrument.

        This function reads raw data files from the Cassini Ultraviolet Imaging Spectrograph (UVIS).
        It requires two files: a binary data file (.DAT) containing raw counts from the detector,
        and a label text file (.LBL) containing metadata about the observation.

        Parameters
        ----------
        filename : str
            The path to the label (.LBL) or binary (.DAT) file to read, with or without file extension.
            If given without an extension, both files are assumed to have the same name with their respective extensions.

        file2 : str, optional
            The path to the second file (either .DAT or .LBL) if `filename` is given with an extension (.DAT or .LBL).
            This allows specifying both files explicitly when they have different names or locations.

        no_extract : bool, optional
            If `True`, the data is not extracted based on window boundaries specified in the label.
            Default is `False`.

        Returns
        -------
        pds_raw_data
            An object containing the raw data and metadata, with attributes and methods for further processing.

        Raises
        ------
        ValueError
            If the required files are not provided, cannot be found, or if the data type is unrecognized.

        Notes
        -----
        - The function pairs the .DAT and .LBL files automatically if only one is specified without an extension.
        - The data is read and stored as a NumPy array, with dimensions adjusted according to the label information.
        - If `no_extract` is `False`, the data is extracted based on the window boundaries and binning specified in the label.

        Examples
        --------
        Read a PDS file set by specifying the base filename without extension:

        >>> data_pds = pds_raw_data('example_file')

        Read a PDS file set by specifying both the label and data files explicitly:

        >>> data_pds = pds_raw_data('data_file.DAT', 'label_file.LBL')

        """

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
        # SAMPLE : Number of frames
        data_dims = ast.literal_eval(self.qube.CORE_ITEMS)

        # Binary type
        if   self.qube.CORE_ITEM_TYPE == 'IEEE_REAL'            and self.qube.CORE_ITEM_BYTES == 4 :
            data_type = np.float32
        elif self.qube.CORE_ITEM_TYPE == 'MSB_UNSIGNED_INTEGER' and self.qube.CORE_ITEM_BYTES == 2 :
            data_type = np.uint16
        else : raise ValueError("Unrecognized data type: "+str(self.qube.CORE_ITEM_TYPE))

        # Read
        self.raw_data = read_binary_pds(filename_dat=filedat, data_dims=data_dims, data_type=data_type)
        self.samples  = self.raw_data.shape[0]
        # data[SAMPLE, LINE, BAND]

        if not no_extract :
            x1 = self.qube.UL_CORNER_BAND
            x2 = self.qube.UL_CORNER_BAND + (self.qube.LR_CORNER_BAND-self.qube.UL_CORNER_BAND+1) // self.qube.BAND_BIN
            y1 = self.qube.UL_CORNER_LINE
            y2 = self.qube.UL_CORNER_LINE + (self.qube.LR_CORNER_LINE-self.qube.UL_CORNER_LINE+1) // self.qube.LINE_BIN
            self.raw_data = self.raw_data[:,y1:y2,x1:x2]
        #_______________________________________________

        # OTHER ATTRIBUTES
        # Data from files
        

        self.calibrated_data = np.copy(self.raw_data)

        self.calibration       = None
        self.calibration_error = None
        self.is_calibrated = False


        # Compute other data
        self.sctime_sec_start = float(self.label.SPACECRAFT_CLOCK_START_COUNT.split('/')[-1])

        self.channel      = 'FUV' if 'FUV' in self.label.PRODUCT_ID else 'EUV'
        self.slit         = self.label.SLIT_STATE
        self.pix_bandpass = pixel_bandpasses[self.channel]
        self.slit_ratio   = slit_ratios[self.channel][self.slit]

        self.INTEGRATION_DURATION = float(self.label.INTEGRATION_DURATION.split()[0])


class UVIS_Bin:
    def __init__(self, bin_attributes, bin_boundaries, uvis_obs):
        self.bins = list_ndarray(bin_boundaries)
        self.bin_def = {key:bin_boundary for key,bin_boundary in zip(bin_attributes, bin_boundaries)}

        self.number_per_bin = np.zeros_like(self.bins, dtype=int)

        self.HD = uvis_obs.HD
        self.pixel_LOS = np.copy(uvis_obs.pixel_LOS)
        self.bin_LOS   = np.full_like(self.bins, fill_value=np.nan, dtype=uvis_obs.pixel_LOS.dtype)

        # Unbinned data
        self.name = uvis_obs.name
        self.data = uvis_obs.data
        self.uncertainty_sup = uvis_obs.uncertainty_sup
        self.uncertainty_inf = uvis_obs.uncertainty_inf
        self.WL = uvis_obs.WL

        self.slit_width = uvis_obs.slit_width

        self.bin_averaged   = False
        self.bin_integrated = False


    def average_bins(self):
        """
        Compute unweighted and weighted average spectra, along with associated uncertainties, for each bin.
        For each bin in `self.bins` (a NumPy array of lists of (x, y) pixel indices), this method:
          1. Computes the unweighted mean, standard deviation, and standard error of the spectra from `self.data[x, y, :]`.
          2. Computes the weighted mean spectrum using the combined uncertainty (average of upper and lower uncertainties)
             as weights, where the weight for each pixel is `1.0 / combined_unc^2`.
          3. Propagates upper and lower uncertainties for the weighted mean using the formula:
             `sqrt(1.0 / sum(1.0 / unc^2))` for both upper (`sup`) and lower (`inf`) uncertainties.
        Output arrays are initialized to NaN and only filled for bins containing at least one (x, y) pair.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Results are stored as attributes:
            - `self.bin_mean_spectrum`: Unweighted mean spectrum per bin.
            - `self.bin_stddev_spectrum`: Unweighted standard deviation per bin.
            - `self.bin_stderr_spectrum`: Unweighted standard error per bin (with small-sample correction).
            - `self.bin_wmean_spectrum`: Weighted mean spectrum per bin.
            - `self.bin_u_sup_spectrum`: Propagated upper uncertainty of weighted mean per bin.
            - `self.bin_u_inf_spectrum`: Propagated lower uncertainty of weighted mean per bin.
        - `self.bins.shape` is the shape of the bin grid (e.g., (d1, d2, ..., dn)).
        - `self.WL.shape` is the shape of the spectral dimension (e.g., (nwl,)).
        - All output arrays have shape `self.bins.shape + self.WL.shape`.
        - If a bin is empty, the corresponding output entries remain NaN.
        - Sets `self.bin_averaged = True` upon completion.
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
            if not pairs:  # Empty bin
                continue

            # 1) Gather data and uncertainties for all pixels in the bin
            stacked_data = np.array([self.data[i, j, :]            for (i, j) in pairs])            
            stacked_sup  = np.array([self.uncertainty_sup[i, j, :] for (i, j) in pairs]) 
            stacked_inf  = np.array([self.uncertainty_inf[i, j, :] for (i, j) in pairs])

            # 2) Unweighted mean & std deviation & standard error
            N = len(pairs)
            
            self.bin_mean_spectrum[idx]   = stacked_data.mean(axis=0)
            if N>1 :
                self.bin_stddev_spectrum[idx] = stacked_data.std (axis=0, ddof=1)

                correction = correction_factor(N)
                self.bin_stderr_spectrum[idx] = correction * self.bin_stddev_spectrum[idx] / np.sqrt(N)
            else :
                self.bin_stddev_spectrum[idx] = np.zeros_like(self.WL)
                self.bin_stderr_spectrum[idx] = np.full_like(self.WL, fill_value=np.nan, dtype=float)




            # 3) Weighted mean using combined uncertainty = (sup + inf)/2
            combined_unc = 0.5 * (stacked_sup + stacked_inf)
            weights      = 1.0 / np.square(combined_unc)

            w_sum        = weights.sum(axis=0)
            w_data_sum   = (stacked_data * weights).sum(axis=0)

            # Weighted mean (avoid division by zero where w_sum==0)
            self.bin_wmean_spectrum[idx] = np.divide(w_data_sum, w_sum,
                                                  out=np.full(self.WL.shape, np.nan), where=(w_sum>0))

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

    def integrate(self, wl_range=(1600,1900), uncertainty=True, method='simpson'):
        
        # Integrate spectra
        self.integrated_data = integrate_spectrum(self.WL, self.data, wl_range=wl_range, axis=-1)
        self.integrated_uncertainty_inf = integrate_spectrum(self.WL, self.uncertainty_inf, wl_range=wl_range, axis=-1, uncertainty=uncertainty)
        self.integrated_uncertainty_sup = integrate_spectrum(self.WL, self.uncertainty_sup, wl_range=wl_range, axis=-1, uncertainty=uncertainty)


        # Average integrated arrays
        self.binned_integrated_data            = np.full(self.bins.shape, np.nan, dtype=float)
        self.binned_integrated_uncertainty_sup = np.full(self.bins.shape, np.nan, dtype=float)
        self.binned_integrated_uncertainty_inf = np.full(self.bins.shape, np.nan, dtype=float)
        self.bin_stddev                        = np.full(self.bins.shape, np.nan, dtype=float)
        self.bin_stderr                        = np.full(self.bins.shape, np.nan, dtype=float)

        for idx in np.ndindex(self.bins.shape):
            
            pairs = self.bins[idx]
            if not pairs:  continue

            N = len(pairs)

            self.binned_integrated_data[idx] = np.mean([self.integrated_data[i, j] for (i, j) in pairs])

            if N>1:
                correction = correction_factor(N)

                self.bin_stddev[idx]             = np.std ([self.integrated_data[i, j] for (i, j) in pairs], ddof=1)
                self.bin_stderr[idx]             = correction * self.bin_stddev[idx] / np.sqrt(N)

            else:
                self.bin_stddev[idx]             = 0
                self.bin_stderr[idx]             = np.nan

            self.binned_integrated_uncertainty_sup[idx] = np.sqrt(1/np.sum([1/self.integrated_uncertainty_sup[i, j]**2 for (i, j) in pairs]))
            self.binned_integrated_uncertainty_inf[idx] = np.sqrt(1/np.sum([1/self.integrated_uncertainty_inf[i, j]**2 for (i, j) in pairs]))

        self.bin_integrated = True


        # Integrate the already bin-averaged spectra
        if self.bin_averaged:
            self.integrated_avrg_data            = integrate_spectrum(self.WL, self.bin_mean_spectrum,  wl_range=wl_range, axis=-1, method=method)
            self.integrated_avrg_data_w          = integrate_spectrum(self.WL, self.bin_wmean_spectrum, wl_range=wl_range, axis=-1, method=method)

            self.integrated_avrg_stddev          = integrate_spectrum(self.WL, self.bin_stddev_spectrum, wl_range=wl_range, axis=-1, uncertainty=uncertainty, method=method)
            self.integrated_avrg_stderr          = integrate_spectrum(self.WL, self.bin_stderr_spectrum, wl_range=wl_range, axis=-1, uncertainty=uncertainty, method=method)
            self.integrated_avrg_uncertainty_sup = integrate_spectrum(self.WL, self.bin_u_sup_spectrum,  wl_range=wl_range, axis=-1, uncertainty=uncertainty, method=method)
            self.integrated_avrg_uncertainty_inf = integrate_spectrum(self.WL, self.bin_u_inf_spectrum,  wl_range=wl_range, axis=-1, uncertainty=uncertainty, method=method)


    def plot_bin(self):
        import matplotlib.pyplot as plt

        number_per_bin = np.where(self.number_per_bin == 0, "", self.number_per_bin.astype(str))
        number_per_bin = number_per_bin.T
        number_per_bin = np.flip(number_per_bin, axis=0)
        
        col_val    = list(self.bin_def.values())[0]
        col_labels = [str((col_val[i+1]+col_val[i])/2) for i in range(self.bins.shape[0])]

        row_val    = list(self.bin_def.values())[1]
        row_labels = [str((row_val[i+1]+row_val[i])/2) for i in range(self.bins.shape[1])][::-1]

        fig, ax = plt.subplots()
        ax.set_axis_off()  # On masque les axes


        # Création du tableau au centre
        table = ax.table(
            cellText=number_per_bin,
            loc='center',
            cellLoc='center',  # Aligne le texte dans chaque cellule au centre
            colLabels=col_labels,
            rowLabels=row_labels
        )

        # Ajustement automatique de la figure
        plt.tight_layout()
        plt.show()



    def __repr__(self):
        info = f"<UVIS_bin object>\n"
        info += f"  Observation: {self.name}\n"
        info += f"  Bin shape  : {self.bins.shape}\n"
        info += f"  Bin attributes:\n"
        for key, val in self.bin_def.items():
            info += f"    - {key}: {len(val)-1} bins ({val[0]} to {val[-1]})\n"
        info += f"  Bin averaged? {'Yes' if hasattr(self, 'bin_mean_data') else 'No'}"
        return info
    
    def save(self, filepath: str | Path, overwrite: bool = False):

        p = Path(filepath)
        if p.suffix.lower() != '.uvisbin':
            p = p.with_suffix('.uvisbin')

        print(f"Saving UVIS observation bin object {p.stem}...", end='', flush=True)

        if p.exists() and not overwrite:
            response = input(f"File '{p.absolute()}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response not in ('y', 'yes', 'o', '1', 'oui'):
                print("Save cancelled.")
                return

        with p.open('wb') as f:
            pickle.dump(self, f)
        
        print(' Done')

        return filepath
    

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class UVIS_Observation:
    """
    Class representing a UVIS observation from the Cassini spacecraft.

    This class aggregates and processes raw PDS (Planetary Data System) data from the Cassini UVIS instrument.
    It handles reading, calibration, background correction, geometry computation, and star contamination identification.
    """

    def __init__(self, *args, batch:str=None, prime_instrument='PRIME', ID=0, target='Titan', name=None):
        """
        Initialize a UVIS_Observation object from PDS files.

        Parameters
        ----------
        *args : list
            List of file paths (without extensions) or an iterable of such paths.
        batch : str, optional
            Path to a batch file listing the base names of the PDS files.
        prime_instrument : str, optional
            Identifier for the primary instrument 'CIRS', 'VIMS' or 'ISS. Defaults to 'PRIME'.
        ID : int, optional
            Identifier for multiple observations within one day. Defaults to 0.
        target : str, optional
            Observation target name. Defaults to 'Titan'.
        name : str, optional
            Custom observation name. If None, a default name is generated.

        Notes
        -----
        This initializer reads the raw PDS files, concatenates the counts,
        and initializes metadata including exposure times, calibration arrays,
        geometry information, and instrument details.
        """
        

        # READING DATA
        #________________________
        if batch is not None :
            batch_path = Path(batch)
            batch_dir  = batch_path.parent
            with batch_path.open('r') as f :
                args = [line.strip() for line in f if line.strip()]

            # For each line, if the path is relative, join with batch_dir and remove extension.
            args = [str((batch_dir / line).with_suffix('')) if not Path(line).is_absolute() else str(Path(line).with_suffix('')) for line in args]
            self.pds_data  = [pds_raw_data(f)    for f in args]
            self.raw_files = [  str(f)           for f in args]

        else :
            # Read batch of PDS files
            if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], (str, bytes)):
                args = args[0]
            
            self.pds_data  = [pds_raw_data(f)    for f in args] # Generate pds_raw_data structure for each file in the batch
            self.raw_files = [  str(f)           for f in args]
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
        self.DOY      = int(nameid[8:11])           # Day of year at the begining of observation
        self.prime    = prime_instrument            # UVIS (PRIME), CIRS, VIMS or ISS
        if prime_instrument=='UVIS' : self.prime='PRIME'
        self.is_prime = prime_instrument=='PRIME'

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

        self.channel      = 'FUV' if 'FUV' in nameid else 'EUV'
        self.pix_bandpass = pixel_bandpasses[self.channel]
        self.slit         = self.pds_data[0].label.SLIT_STATE
        self.slit_ratio   = slit_ratios[self.channel][self.slit]
        self.slit_width   = slit_dlambda[self.channel][self.slit]

        self.evil_pixels        = None  # Mask: True when evil pixel
        self.evil_pixels_binned = None

        

        # Name
        if ID>0 : self.IDstr = '_'+str(ID) # Identifier for multiple observation during one DOY
        else    : self.IDstr = ''

        if name is None :
            # Observation name format : CHANNEL_YEAR_DOY_PRIME(_ID)
            self.name = str(self.channel)+'_'+str(self.YEAR)+'_'+nameid[8:11]+'_'+self.prime+self.IDstr
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
        self.instrument = instrument(self.instrument_name, 64)


        # Observation
        self.target = target.upper()

        self.background_level = 0.
        self.background_error = 0.
        self.n_bg_pixels = None
        self.max_gap = None

        # Instance status
        self.is_calibrated   = False
        self.calibration_set = False
        self.is_bkg_removed  = False
        self.is_smoothed     = False




        # TIMES
        #______________________________

        spice.furnsh(env.lsk_path)

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
        spice.unload(env.lsk_path)

        # Sub-exposure times
        self.times_exposition = np.array([np.linspace(self.ET_start[i], self.ET_stop[i], PicsPerExposure, endpoint=True) for i in range(self.n_pics)])

        self.__dict__['_init_done'] = True

    def integrate_radiance(self, wl_range=(1600,1900), method='simpson') :
        """
        Integrate the calibrated radiance over a specified wavelength range.

        Parameters
        ----------
        wl_range : tuple of float, optional
            Wavelength range (min, max) for integration. Default is (1600, 1900).
        method : {'simpson', 'trapezoid'}, optional
            Integration method to use. Default is 'simpson'.

        Returns
        -------
        numpy.ndarray
            Integrated radiance values for each exposure.
        """


        if self.is_calibrated :
            signal = self.data.copy()
        else :
            if not self.calibration_set : self.set_calibration()
            signal = np.array([interpolate_nans(self.cps[i] * self.calibration[i]) for i in range(self.n_pics)])

        # Integrate spectrum
        return integrate_spectrum(self.WL, signal, wl_range=wl_range, axis=2, method=method)
    
    def get_radiance_uncertainty(self) :
        """
        Compute the uncertainty of the calibrated radiance. Uncertainty interval
        is base on the Scores confidence interval (Barker, 2008).

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing (uncertainty_sup, uncertainty_inf) arrays.
        """

        counts_position = self.cps>0

        if self.background_level == 0:
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

    def smooth(self, force=False):
        """
        Smooth the calibrated data and uncertainties using 1D convolution.

        Notes
        -----
        Smoothing is applied only on valid (non-NaN) data points.
        """
        if self.channel!='FUV' :
            print('Smoothing is only available for FUV channel')
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
    def get_calibration(self, sctime, interp='pchip', flat_field=True) :
        """
        Retrieve the calibration multiplier (inverse sensitivity) of the Cassini UVIS instrument.

        This method calculates the calibration factors for the raw data based on several parameters,
        including the UVIS channel (EUV or FUV), slit width, data type, window boundaries, binning,
        and the time of observation.

        Parameters
        ----------
        interp : str, optional
            The interpolation method used to map the lab calibration to the full detector range.
            Options are:
            - 'linear' : Linear interpolation.
            - 'pchip'  : Piecewise Cubic Hermite Interpolating Polynomial.
            Default is 'linear'.

        flat_field : bool, optional
            Whether to apply the flat-field correction to the raw data.
            Default is `True`.

        Returns
        -------
        dict
            A dictionary containing:
            - 'calibration'       : numpy.ndarray
                The calibration factor, binned and reshaped to match the raw data dimensions.
            - 'calibration_error' : numpy.ndarray
                The calibration error array.

        Notes
        -----
        - The method incorporates several calibration steps:
            - Laboratory calibration data is adjusted for slit width.
            - Time variation is accounted for using spacecraft time.
            - Flat-field corrections are applied if `flat_field` is `True`.
            - Binning is performed according to the spatial and spectral binning factors.
        - For the FUV channel, pixels known as 'evil' pixels with anomalous behavior are handled,
          and corresponding elements in the arrays are set to NaN.

        Examples
        --------
        Compute the calibration factors with linear interpolation and flat-field correction:

        >>> calibration = data_pds.get_calibration()
        >>> cal_factor  = calibration['calibration']
        >>> cal_error   = calibration['calibration_error']

        Compute the calibration factors without flat-field correction:

        >>> calibration = data_pds.get_calibration(flat_field=False)
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

    def calibrate(self, wl_interp='pchip', nan_interp='linear', flat_field=True, smooth=True, extrapolate=False) :
        """
        Calibrate the raw data to obtain radiance and sets class attributes.

        Parameters
        ----------
        interp : {'pchip', 'linear'}, optional
            Interpolation method for processing the data. Default is 'pchip'.
        flat_field : bool, optional
            Whether to apply flat-field corrections. Default is True.
        smooth : bool, optional
            Whether to smooth the calibrated data. Default is True.
        extrapolate : bool, optional
            Whether to perform extrapolation during interpolation. Default is False.
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
    def get_geometry(self, ET:float, **kwargs) :
        """
        Compute the geometry for a given ephemeris time.

        Parameters
        ----------
        ET : float
            Ephemeris time for which to compute the geometry.
        **kwargs
            Additional keyword arguments for the geometry class.

        Returns
        -------
        geometry
            A geometry object computed for the given time.
        """
        
        from .geometry.geometry import geometry

        return geometry( ET, u=self, **kwargs)

    def set_geometry(self, et_range=None, **kwargs) :
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

        
        self.geometry = []
        if et_range is None : et_range = self.ET_middle

        for i in tqdm(range(len(et_range)), desc="Computing geometry", file=sys.stdout):#, ncols=100) :
            et = et_range[i]
            self.geometry.append(self.get_geometry(et, **kwargs))

        self.HD = np.mean([g.HD for g in self.geometry])
        self.pixel_LOS = np.array([
            self.geometry[i].used_pixels_LOS for i in range(len(self.geometry))
            ])
        

        dtype = np.dtype([
            ('MAG',     float),
            ('is_UV',   bool ),
            ('number',  int  ),
            ('on_disk', bool  )
        ])

        n_pixels = self.geometry[0].n_used_pixels

        self.pixel_star_geometry = [
            (
                self.geometry[i].pixel_stars[j]["MAG"],
                self.geometry[i].pixel_stars[j]["is_UV"],
                self.geometry[i].pixel_stars[j]["number"],
                self.geometry[i].pixel_stars[j]["on_disk"]
            )
            for i in range(self.n_pics)
            for j in range(self.geometry[i].n_used_pixels)
        ]

        self.pixel_star_geometry = np.array(
            self.pixel_star_geometry, dtype=dtype
            ).reshape(self.n_pics, n_pixels)
        
    def plot_all_geometry(self, folder, out_format='png', duration=1/60):
        import matplotlib.pyplot as plt
        from PIL import Image
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

        folder = Path(folder)
        if not folder.exists() :
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Creating directory: {folder}")


        if out_format=='gif':
            # Case: GIF – assemble all images in memory
            frames = []
            for i, obj in tqdm(enumerate(self.geometry), total=len(self.geometry), desc="Rendering geometry animation"):
                buf = io.BytesIO()
                obj.plot(save=True, savename=buf)
                buf.seek(0)

                im = Image.open(buf).convert("RGBA")
                frames.append(im)
                buf.close()


                # frame = imageio.imread(buf)
                # frames.append(frame)
                # buf.close()

            gif_filename = folder / f"{self.name}.gif"

            frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration,
            disposal=2           # important for transparency
            )
            # imageio.mimsave(str(gif_filename), frames, duration=duration, loop=0)
            print(f"GIF created : {gif_filename}")
        else :
            # Standard case: save each plot as an individual file
            for i, obj in tqdm(enumerate(self.geometry), total=len(self.geometry), desc="Rendering geometry plots"):
                filename = folder / f"geometry_{i}.{out_format}"
                obj.plot(save=True, savename=str(filename), show=False)
            
            plt.close('all')


    # -------- STARS IDENTIFICATION
    def add_pixel_stars_from_file(self, file:str|Path):
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

    
    def plot_radiance_evolution(self, output_path=None, ylim=(0.01,20), yscale='log', wl_range=(1600,1900), method='trapezoid') :
        import matplotlib.pyplot as plt
        """
        Plot the evolution of integrated radiance for each pixel over exposures.

        Parameters
        ----------
        output_path : str, optional
            File path to save the PDF containing the plots. Default is 'signal_time_variation.pdf'.
        ylim : tuple of float, optional
            Y-axis limits for the plots. Default is (0.01, 20).
        yscale : str, optional
            Scale for the y-axis ('log' or 'linear'). Default is 'log'.
        wl_range : tuple of float, optional
            Wavelength range for integration. Default is (1600, 1900).
        method : {'simpson', 'trapz'}, optional
            Integration method to use. Default is 'simpson'.
        """

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


    def check_stars(self,  cmap='gist_ncar', color_scale=(0,14), wl_range=(1600,1900), method='trapezoid'):
        """
        Create a heatmap of integrated radiance and highlight pixels affected by stars.
        The heatmap is interactive for the user to identify pixels as contaminated by a star.

        Parameters
        ----------
        cmap : str, optional
            Colormap to use for the heatmap. Default is 'gist_ncar'.
        color_scale : tuple of float, optional
            Color scale limits. Default is (0, 14).
        wl_range : tuple of float, optional
            Wavelength range for integration. Default is (1600, 1900) in angströms.
        method : {'simpson', 'trapezoid'}, optional
            Integration method to use. Default is 'simpson'.

        Notes
        -----
        The heatmap displays integrated radiance with annotations and rectangles indicating star-contaminated pixels.
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.patches import Rectangle
        from matplotlib import pyplot as plt

        integrated_radiance = self.integrate_radiance(wl_range=wl_range, method=method)

        X,Y = np.arange(self.n_pics+1)-0.5, np.arange(self.n_pixels+1)-0.5

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
        xticks = np.arange(self.n_pics)
        yticks = np.arange(self.n_pixels)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_aspect('equal')
        ax.set_zorder(2)
        cax.set_zorder(1)


        # ADD GEOMETRY INFO
        if self.geometry is not None :
            
            if np.any(self.pixel_star_geometry[:]['on_disk']) :
                index = np.where(self.pixel_star_geometry[:]['on_disk'])
                result = list(zip(index[0], index[1]))

                for xuv,yuv in result :

                    rect = Rectangle((xuv - 0.5, yuv - 0.5), 1, 1, 
                    edgecolor='green', linewidth=1, facecolor='none')
                    ax.add_patch(rect)
            
            if np.any(self.pixel_star_geometry[:]['number']>0) :
                index  = np.where((self.pixel_star_geometry[:]['number']>0)*(~self.pixel_star_geometry[:]['is_UV']))
                result = list(zip(index[0], index[1]))

                for xuv,yuv in result :

                    rect = Rectangle((xuv - 0.5, yuv - 0.5), 1, 1, 
                     edgecolor='yellow', linewidth=1, facecolor='none')
                    ax.add_patch(rect)


                if np.any(self.pixel_star_geometry[:]['is_UV']) :
                    index_uv = np.where(self.pixel_star_geometry[:]['is_UV'])
                    result = list(zip(index_uv[0], index_uv[1]))

                    for xuv,yuv in result :

                        rect = Rectangle((xuv - 0.5, yuv - 0.5), 1, 1, 
                        edgecolor='purple', linewidth=1, facecolor='none')
                        ax.add_patch(rect)

        
        # INTERACTION HANDLES
        for (row, col) in self.pixel_stars:
            x_center = (X[col] + X[col+1]) / 2
            y_center = (Y[row] + Y[row+1]) / 2
            line, = ax.plot(x_center, y_center, marker='x', color='black', markersize=8, mew=2)
            self.markers[(row, col)] = line
        text_handle = ax_text.text(0, 1, "Selected pixels:\n", va='top', fontsize=10,family='monospace')
        # if self.markers :
        #     for (row, col) in self.markers:
        #         x_center = (X[col] + X[col+1]) / 2
        #         y_center = (Y[row] + Y[row+1]) / 2
        #         line, = ax.plot(x_center, y_center, 'x', color='red', markersize=8, mew=2)
        #         self.markers[(row, col)] = line


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
            x_center = (X[col] + X[col+1]) / 2
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

                col = np.searchsorted(X, x_click) - 1
                row = np.searchsorted(Y, y_click) - 1

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
                col = np.searchsorted(X, x_hover) - 1
                row = np.searchsorted(Y, y_hover) - 1

                # Vérifier qu’on est bien dans la grille
                if 0 <= row < self.n_pixels and 0 <= col < self.n_pics:
                    # Exemple: récupérer la valeur du signal
                    pixel_value = integrated_radiance[col, row]

                    # Construire la chaîne de texte à afficher
                    if self.geometry is None:
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
                            f"SZA        : {self.pixel_LOS[col,row,0]['sza']:.1f}°"
                        )
                    

                    # On peut aussi incorporer d'autres infos (e.g. magnitude, etc.)
                    # si on a accès à self.geometry ou un tableau de métadonnées.

                    # Mise à jour de l’annotation
                    hover_annotation.xy = (col, row)
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
        

    # -------- BACKGROUND NOISE
    def get_background(self, mode:Literal['average', 'simulate']='simulate', alt_limit=2000, wl_range=(1600,1900), n_fits=20, parallel=True):
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
        alt_limit : int, optional
            Minimum altitude limit to consider a pixel as background. Default is 2000.
        wl_range : tuple of int, optional
            Wavelength range (min, max) to consider for background calculation. Default is (1600, 1900).
        n_fits : int, optional
            Number of fits to perform in simulation mode. Default is 20.
        parallel : bool, optional
            Whether to perform the simulation in parallel. Default is True.

        Returns
        -------
        tuple of float
            A tuple (cps, cps_err) representing the background counts per second and its uncertainty.

        Raises
        ------
        ValueError
            If the specified wavelength range is outside the detector range or if geometry is not initialized.
        """

        from .background import histogram, gaps, max_gap

        if not (self.WL[0]<wl_range[0]<self.WL[-1] and self.WL[0]<wl_range[1]<self.WL[-1]) :
            raise ValueError(f'Please select a wl_range within the {self.channel} channel.')

        if self.geometry is None:
            raise ValueError("The observation geometry must be initialized before determining the background.")

        
        MinPixAlt = np.min(self.pixel_LOS['alt'], axis=2)

        loop = True
        while loop :
            if not np.any(MinPixAlt>alt_limit) :
                if   alt_limit>2000 : alt_limit = 2000
                elif alt_limit>1500 : alt_limit = 1500
                else :
                    print('No background pixels, please manually set a background level.')
                    return np.nan, np.nan
            else : loop=False


        # Basic averaging method
        self.n_bg_pixels = np.sum(MinPixAlt>alt_limit)
        bg_pixels = self.counts[(MinPixAlt>alt_limit) * ~self.pixel_corrupted * ~self.pixel_stars_mask, :]
        bg_pixels = bg_pixels[:, (self.WL>=wl_range[0]) * (self.WL<=wl_range[1])]
        bg_pixels = bg_pixels[np.any(bg_pixels, axis=1)] # Filter total transmission losses
        counts_per_pixel = np.sum(bg_pixels, axis=1)


        #         total number of counts      / # of spectral pixels                   / exposition time
        cps     = np.mean(counts_per_pixel)  /  (bg_pixels.shape[1] *self.spat_bin*self.spec_bin)   /  self.expo_time
        cps_err = np.std(counts_per_pixel)  /   (bg_pixels.shape[1] *self.spat_bin*self.spec_bin)  /   self.expo_time

        wl_index = (
            np.where(self.WL>=wl_range[0])[0][ 0],
            np.where(self.WL<=wl_range[1])[0][-1]+1
        )

        # Flag corrupted pixels (transmission losses)
        self.max_gap = max_gap(cps, self.expo_time,
                               SPATIAL_BIN=self.spat_bin, SPE_UL=self.spec_start, SPE_LR=self.spec_stop, BIN=self.spec_bin)
        if self.max_gap <10 and self.spat_bin>1 : self.max_gap = 100
        for i_pic in range(self.counts.shape[0]) :
            for i_spat in range(self.counts.shape[1]) :

                # Total losses
                if not np.any(self.counts[i_pic,i_spat, :]) :
                    self.pixel_corrupted[i_pic,i_spat] = True

                # Lyman-alpha losses
                if not np.any(self.counts[i_pic,i_spat, 122//self.spec_bin:137//self.spec_bin]) :
                    self.pixel_corrupted[i_pic,i_spat] = True

                # Simulate maximum gap on detector
                if max(gaps(self.counts[i_pic,i_spat, :])) > self.max_gap :
                    self.pixel_corrupted[i_pic,i_spat] = True
                



        


        if mode=='simulate' :

            bg_pixels = self.counts[(MinPixAlt>alt_limit) * ~self.pixel_corrupted * ~self.pixel_stars_mask, :]
            bg_pixels = bg_pixels[:, (self.WL>=wl_range[0]) * (self.WL<=wl_range[1])]


            # Build observations histogram
            H = np.zeros(bg_pixels.shape[1]+1)
            for pixel in bg_pixels :
                H += histogram(
                    gaps(pixel), max_value=bg_pixels.shape[1]
                    )
            H /= bg_pixels.shape[0]

            

            # Perform simulation and fits
            if parallel :
                print('Retrieving background...', end='')
                from multiprocessing import Pool, cpu_count



                tasks = [
                    (
                        i_fit,
                        H,
                        bg_pixels.shape[0],
                        1024,
                        self.expo_time,
                        self.spec_start,
                        self.spec_stop,
                        self.spec_bin,
                        self.spat_bin,
                        wl_index
                    )
                    for i_fit in range(n_fits)
                ]

                with Pool(processes=cpu_count()) as pool:
                    bg_fits = pool.starmap(do_bg_fit, tasks)
                
                print(" Done")

            else :
                bg_fits = []

                sys.stdout.write(f"Performing background fits: 0%")
                sys.stdout.flush()
                    
                
                for i_fit in range(n_fits) :
                    bg_fits.append(
                        bg_fit(
                        H, bg_pixels.shape[0], 1024, exposition=self.expo_time,
                        SPE_UL=self.spec_start,    SPE_LR=self.spec_stop,
                        SPECTRA_BIN=self.spec_bin, SPATIAL_BIN=self.spat_bin, wl_index=wl_index
                        )
                    )
                    progress = 100*(i_fit+1)//n_fits
                    sys.stdout.write(f"\rPerforming background fits: {progress}%")
                    sys.stdout.flush()
                print('')
            

            cps     = np.mean(bg_fits)
            cps_err = np.std (bg_fits)
            if cps > 1e-3 : 
                warnings.warn(
                    f"Background level is unusually high: {cps:.2e} counts/s. Probably due to a wrong fit in the simulation (local minimum), use 'average' mode instead.",
                    RuntimeWarning
                )
        return cps, cps_err


    def set_background(self, bg=None, bg_uncertainty=None, **kwargs) :
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
        keys,
        bin_boundaries= (
            default_lat_bins,
            default_alt_bins,
            [0,12,24]
        ),
        mode: Literal['center', 'all'] = 'center'
    ):
        """
        Bin each pixel into a multidimensional bin structure (self.bins) based on provided boundaries.

        In 'center' mode, a single representative value per property (e.g., the pixel center)
        is used and must satisfy boundaries[0] <= value < boundaries[-1].
        In 'all' mode, all values of the pixel must fall into the same bin; otherwise, the pixel is ignored.

        Each cell in the resulting array (which is a list) accumulates (i_pic, i_pix) tuples corresponding
        to valid pixels.

        Parameters
        ----------
        keys : Tuple[str, ...]
            Names of the pixel properties (e.g., ('lat', 'alt')) available in self.pixel_LOS.
        bin_boundaries : Tuple[List[float], ...]
            A tuple of lists defining the bin edges for each property. Bins are defined as [edge_i, edge_i+1),
            except for the last bin which includes its upper edge.
        mode : Literal['center', 'all']
            The binning mode:
                - 'center': use a single representative value.
                - 'all': require that all values in the pixel fall within the same bin.

        Raises
        ------
        ValueError
            If the number of keys does not match the number of bin_boundaries sets.
        """

        print('Creating bins...', end='', flush=True)

        if len(keys) != len(bin_boundaries):
            raise ValueError("The number of 'keys' must match the number of 'bin_boundaries' sets.")
        

        bins = UVIS_Bin(keys,bin_boundaries, self)

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


        # MEAN PIXEL GEOMETRIC PROPERTIES
        for idx in np.ndindex(bins.bins.shape):
            pairs = bins.bins[idx]
            if not pairs:  # Empty bin
                continue
            # print(list(bins.bin_LOS.dtype.names))
            for key in self.pixel_LOS.dtype.names:

                # print(key)
                # print([self.pixel_LOS[i, j,:][key] for (i, j) in pairs])

                # print(np.mean([self.pixel_LOS[i, j,:][key] for (i, j) in pairs]))
                # print('eeee')
                # print(bins.bin_LOS[idx][key])
                
                bins.bin_LOS[idx][key] = np.mean([self.pixel_LOS[i, j,:][key] for (i, j) in pairs])
                # print(bins.bin_LOS[idx][key])


        print(' Done')
        return bins


        

    # -------- SAVE MANAGMENT
    def save(self, filepath: str = None, overwrite: bool = False, fullsave=True):
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
        fullsave  : bool, optional
            If False, self.geometry attribute is removed from the saved
            If True, all the class instance is saved, taking much
            more time.

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
        
        if not fullsave :
            tmp = self.geometry
            del self.geometry


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

        if not fullsave: self.geometry = tmp
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
    
    def __setattr__(self, key, value):
        # Si l'initialisation est terminée et que l'attribut n'existe pas déjà, on lève une erreur
        if self.__dict__.get('_init_done', False) and not hasattr(self, key):
            raise AttributeError(f"UVIS_Observation has no attribute '{key}'")
        # Sinon, on affecte normalement
        object.__setattr__(self, key, value)

    def add_attribute(self, key, value):
        """Ajoute un nouvel attribut même après l'initialisation."""
        object.__setattr__(self, key, value)
