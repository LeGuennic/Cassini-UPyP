# Python
from __future__ import annotations

from typing import Literal
from numpy.typing import ArrayLike
from pathlib import Path

# Computational
from scipy.integrate   import simpson
from scipy.interpolate import PchipInterpolator
from scipy.io          import readsav
from scipy.ndimage     import convolve1d
import numpy as np

from .utils import env_config
env = env_config()


# UNCERTAINTY
def poisson_error(x: ArrayLike, bound: Literal['inf', 'sup'], sigma: float = 1.0):
    """
    Compute Garwood confidence limits for a Poisson count.

    This function returns *one* bound (lower or upper) of the central
    two-sided confidence interval for an observed Poisson count `x`,
    using the classical Garwood construction obtained by inverting
    the chi-square CDF.

    Parameters
    ----------
    x : int or array-like of int
        Observed Poisson count(s), must be >= 0.
    bound : {'inf', 'sup'}
        Which bound to return:
        - 'inf' : lower confidence limit L.
        - 'sup' : upper confidence limit U.
        Aliases accepted: 'lower' -> 'inf', 'upper' -> 'sup'.
    sigma : float, optional
        Number of Gaussian sigmas corresponding to the *central* confidence
        level (two-sided). For example:
        - sigma = 1.0  -> CL ≈ 68.27%
        - sigma = 1.96 -> CL ≈ 95%
        Default is 1.0.

    Returns
    -------
    float or numpy.ndarray
        The requested bound with the same shape as `x`. Returns a Python
        float if `x` is scalar.

    Notes
    -----
    Let CL be the central confidence level and alpha = 1 - CL, with
    CL = Φ(sigma) - Φ(-sigma), where Φ is the standard normal CDF.
    The Garwood limits are

        L = 0.5 * chi2.ppf(alpha/2, 2*x)        for x > 0,   and L = 0 if x = 0
        U = 0.5 * chi2.ppf(1 - alpha/2, 2*(x+1))

    These intervals have (at least) the nominal coverage.

    References
    ----------
    - F. Garwood (1936). "Fiducial Limits for the Poisson Distribution."
      Biometrika, 28(3/4), 437–442.
    - G. Casella & R. L. Berger (2002). *Statistical Inference*, 2nd ed.
    """
    
    from scipy.stats import chi2, norm

    # Normalize and validate bound
    b = bound.lower().strip()
    if b in {"lower"}:
        b = "inf"
    elif b in {"upper"}:
        b = "sup"
    if b not in {"inf", "sup"}:
        raise ValueError("Invalid bound type. Use 'sup' or 'inf' (or 'upper'/'lower').")

    # Convert and validate x
    x_arr = np.asarray(x)

    if np.any(x_arr < 0):
        raise ValueError("`x` must be >= 0.")

    # Confidence level from sigma
    cl = norm.cdf(sigma) - norm.cdf(-sigma)
    alpha = 1 - cl

    # Compute bounds (vectorized)
    if b == "sup":
        res = 0.5 * chi2.ppf(1 - alpha / 2, 2*(x_arr+1))
    else:  # 'inf'
        lower_raw = 0.5 * chi2.ppf(alpha / 2.0, 2 * x_arr)
        res = np.where(x_arr == 0, 0.0, lower_raw)

    # Return scalar if scalar input
    if np.isscalar(x) or np.ndim(x_arr) == 0:
        return float(np.asarray(res).item())
    return np.asarray(res, dtype=float)

def correction_factor(N:int, log=True) -> float:
    from scipy.special import gammaln, gamma


    N = np.asarray(N, dtype=float)

    if log:
        log_ratio = gammaln((N - 1)/2) - gammaln(N/2)
        return np.exp(log_ratio) * np.sqrt((N - 1)/2)
    else:
        return ( gamma((N - 1) / 2) /
                 gamma(N / 2)        )  * np.sqrt((N - 1) / 2)


# SPECTRUM
def UVIS_WL(channel, bin=1) :
    """
    Calculate the wavelength array for the specified UVIS channel.

    Parameters
    ----------
    channel : str
        The UVIS channel for which to calculate wavelengths.
        - 'FUV': Far     Ultraviolet channel.
        - 'EUV': Extreme Ultraviolet channel.

    bin : int, optional
        The binning factor. Default is 1 (no binning). If `bin` > 1, the wavelength array
        is averaged over bins of size `bin`.

    Returns
    -------
    numpy.ndarray
        The array of wavelengths corresponding to the UVIS channel and binning factor.

    Raises
    ------
    ValueError
        If the specified channel is not supported.
    """

    match channel:
        case 'FUV' :
            D=1.e7/1066
            alpha=3.46465e-5+(9.22+.032)*np.pi/180

            beta = (np.arange(1024)-511.5)*.025*.99815/300
            beta = np.atan(beta)+(.032*np.pi/180)+3.46465e-5

            l = D*(np.sin(alpha)+np.sin(beta))

            if bin == 1 : return l
            

            wl=np.zeros(1024//bin)
            for k in range(0,1024//bin) : wl[k]=sum(l[k*bin:(k+1)*bin] )/bin
            return wl
        
        case 'EUV' :
            D=1.e7/1371
            alpha=8.03*np.pi/180 + .00032 - .0000565

            beta = (np.arange(1024)-511.5)*.025*.9987/300
            beta = np.atan(beta)-(1.19*np.pi/180)+.00032-.0000565

            l = D*(np.sin(alpha)+np.sin(beta))

            if bin == 1 : return l
            

            wl=np.zeros(1024//bin)
            for k in range(0,1024//bin) : wl[k]=sum(l[k*bin:(k+1)*bin] )/bin
            return wl
        case _:
            raise ValueError(f"Channel error, unknown UVIS channel : {channel}")

def integrate_spectrum(wl, s, wl_range=None, method='simpson', axis=0, uncertainty=False):
    """
    Integrate the array `s` over the wavelength axis `wl`.
    
    Parameters
    ----------
    wl : 1D array
        Wavelength array.
    s : array
        Spectral data (if uncertainty=False) or uncertainty array (if uncertainty=True).
        The dimension along `axis` must match the length of `wl`.
    wl_range : tuple or None, optional
        Tuple (min_wl, max_wl) defining the integration bounds in wavelength.
        If None, integration is done over the full range of `wl`.
    method : {'simpson', 'trapz'}, optional
        Integration method. 'simpson' uses scipy.integrate.simpson,
        'trapz' uses numpy.trapz.
    axis : int, optional
        The axis along which to integrate. Must match the dimension of `wl`.
    uncertainty : bool, optional
        If True, `s` is interpreted as the uncertainty array. The output
        is then the integrated uncertainty (quadratic sum under the integral).
    
    Returns
    -------
    float or array
        Integrated value (or array of integrated values if there are extra dimensions).
        - If uncertainty=False, it is the integral of the spectral data.
        - If uncertainty=True, it is the total uncertainty computed by
          sqrt( integral of s^2 ).
    """
    
    wl = np.asarray(wl)
    s  = np.asarray(s)

    if wl.size != s.shape[axis]:
        raise ValueError("The wavelength and spectrum arrays must have the same shape.")
    

    # 1. Sélection de la plage de longueurs d'onde si wl_range est spécifié
    if wl_range is not None:
        mask   = (wl >= wl_range[0]) * (wl <= wl_range[1])
        wl_sub = wl[mask]
        s_sub  = np.take(s, np.where(mask)[0], axis=axis)
    else:
        wl_sub = wl
        s_sub  = s

    # 2. En fonction de la méthode, on effectue l'intégration
    if method == 'simpson':
        if len(wl_sub) < 3:
            raise ValueError("Simpson's method requires at least 3 points for integration.")
        if (wl_sub.size - 1) % 2 != 0:
            # Remove last point
            wl_sub = wl_sub[:-1]
            s_sub  = np.take(s_sub, np.arange(s_sub.shape[axis] - 1), axis=axis) 
            
            # warnings.warn("Odd number of segments for Simpson's method, last point removed.", RuntimeWarning)

        
        if not uncertainty:
            integral = simpson(s_sub, x=wl_sub, axis=axis)
        else:
            # Quadratic integration
            delta_wl  = wl_sub[2::2] - wl_sub[:-2:2]

            indices_a = np.arange(0, wl_sub.size-2, 2)  # indices: 0, 2, 4, ...
            indices_b = np.arange(1, wl_sub.size-1, 2)  # indices: 1, 3, 5, ...
            indices_c = np.arange(2, wl_sub.size,   2)  # indices: 2, 4, 6, ...
            
            # Extract the segments along the specified axis using np.take
            a = np.take(s_sub, indices_a, axis=axis)
            b = np.take(s_sub, indices_b, axis=axis)
            c = np.take(s_sub, indices_c, axis=axis)
            
            # Expand delta_wl so it peut se diffuser correctement avec a, b, c
            # On veut insérer des axes de taille 1 dans toutes les dimensions sauf celle d'intégration
            shape = [1] * s_sub.ndim
            shape[axis] = delta_wl.size
            delta_wl_expanded = delta_wl.reshape(shape)
            
            # Each Simpson segment contributes:
            # (delta_wl/6)^2*(a^2 + c^2) + (4*delta_wl/6)^2*(b^2)
            integral = np.sqrt(np.sum(
                (  delta_wl_expanded/6.0)**2 * (a**2 + c**2) +
                (4*delta_wl_expanded/6.0)**2 * (b**2),
                axis=axis
            ))
        
    elif method == 'trapezoid':
        if not uncertainty:
            integral = np.trapezoid(s_sub, x=wl_sub, axis=axis)
        else: # Quadratic integration
            delta_wl = np.diff(wl_sub)

            # Build slicers to extract the "left" and "right" parts along the integration axis.
            slicer_left  = [slice(None)] * s_sub.ndim
            slicer_right = [slice(None)] * s_sub.ndim
            slicer_left[axis]  = slice(0, -1)
            slicer_right[axis] = slice(1, None)
            
            s_left  = s_sub[tuple(slicer_left)]
            s_right = s_sub[tuple(slicer_right)]
            
            # Reshape delta_wl for broadcasting along the integration axis.
            new_shape = [1] * s_sub.ndim
            new_shape[axis] = delta_wl.shape[0]
            delta_wl_expanded = delta_wl.reshape(new_shape)
            
            # For each segment, variance_seg = (Δwl/2)^2*(s_i^2 + s_(i+1)^2)
            integral = np.sqrt(np.sum((0.5 * delta_wl_expanded)**2 * (s_left**2 + s_right**2), axis=axis))
    else:
        raise ValueError("Method must be either 'simpson' or 'trapezoid'.")

    return integral

def interpolate_nans(arr, method: str = "linear"):
    """
    Interpolate NaNs in each row of a 1-D or 2-D array.

    Parameters
    ----------
    arr : array-like
        Input data containing NaNs.
    method : {"linear", "pchip"}, optional
        Interpolation scheme to use (default: "linear").

    Returns
    -------
    np.ndarray
        Array with NaNs replaced by interpolated values.
        If the input was 1-D, the output is 1-D.
    """


    # -- standardise input ----------------------------------------------------
    a = np.asarray(arr, dtype=float)
    if a.ndim > 2:
        raise ValueError("Only 1-D or 2-D arrays are supported.")
    was_1d = a.ndim == 1
    if was_1d:
        a = a[np.newaxis, :]          # shape -> (1, N)

    x = np.arange(a.shape[1])         # common x-axis for all rows
    out = a.copy()

    # -- choose interpolation routine ----------------------------------------
    if method == "linear":
        for row in out:
            mask = np.isnan(row)
            if mask.all(): continue   # keep all-NaN rows unchanged
            row[mask] = np.interp(x[mask], x[~mask], row[~mask])

    elif method == "pchip":
        for row in out:
            mask = ~np.isnan(row)
            if not mask.any(): continue
            f = PchipInterpolator(x[mask], row[mask])
            row[:] = f(x)

    else:
        raise ValueError("method must be 'linear' or 'pchip'")

    return out.squeeze() if was_1d else out


# CALIBRATION
def uvis_lab_calibration(channel:str, filename:str=None) :
        """
        Read laboratory calibration data for the specified UVIS channel and return sensitivity information.

        Sensitivity in units of (counts/second) / (kilorayleigh)
        This is the full-slit, low-resolution, monochromatic extended source sensitivity measured in the laboratory in 1997, 
        and updated in 1999

        Parameters
        ----------
        channel : str
            The UVIS channel for which to read the calibration data. 'FUV' or 'EUV'.
        filename : str, optional
            The path to the calibration data file. If `None`, the default filename is constructed as
            '{channel}_1999_Lab_Cal.dat'.
        """

        if filename is None :
            filename = Path(channel+'_1999_Lab_Cal.dat')
            filename = env.calibration_dir / filename

        data = np.loadtxt(filename, skiprows=1)

        return {'WAVELENGTH'        : np.concatenate((data[:,0], data[:,3])),
                'SENSITIVITY'       : np.concatenate((data[:,1], data[:,4])),
                'SENSITIVITY_ERROR' : np.concatenate((data[:,2], data[:,5]))}

def get_cal_time_variation(channel:str, sctime) :
    """
    Retrieve the spectral modulation array for a given UVIS channel and spacecraft time.

    This function reads calibration trending data from the IDL 'uvis_calibration_trending_v01_data.sav' computed from
    IDL routine uvis_calibration_trending_v01.pro and computes the spectral modulation
    (`specmod`) for the specified UVIS channel at a given spacecraft time (`sctime`).
    It interpolates between the two closest calibration times to compute the spectral modulation.

    Parameters
    ----------
    channel : str
        The UVIS channel for which to get the calibration time variation. 'FUV' or 'EUV'.
    sctime : float
        The spacecraft time in seconds for which to compute the spectral modulation.

    Returns
    -------
    numpy.ndarray
        A NumPy array of size 1024 containing the spectral modulation values.

    Notes
    -----
    - The calibration trending data is read from 'calibration_files/uvis_calibration_trending_v01_data.sav'.
    - Spectral modulation ratios are interpolated linearly in time when necessary.
    - If the `sctime` is outside the calibration data range, default values are used:
      - Before the earliest time: an array of ones.
      - After the latest time: the last available spectral modulation ratio.
    """

    cal_file = Path(env.calibration_dir) / 'uvis_calibration_trending_v01_data.sav'
    cal_trend = readsav(cal_file)


    arr = cal_trend.arr_fuv if channel=='FUV' else cal_trend.arr_euv
    del cal_trend
    sctime_mods = [arr[k].desc.sctime_sec_start[0] for k in range(len(arr))]

    if   sctime <  sctime_mods[0]  : specmod = np.ones(1024)
    elif sctime >= sctime_mods[-1] : specmod = arr[-1].ratio
    else :
        time_index = np.searchsorted(sctime_mods, sctime)
        t1, t2 = sctime_mods[time_index-1], sctime_mods[time_index]
        specmod1 = arr[time_index-1].ratio
        specmod2 = arr[time_index  ].ratio
        specmod  = specmod1 + (specmod2 - specmod1) * (sctime - t1) / (t2 - t1)
    del arr
    return specmod

def get_ff_time_variation(channel:str, sctime) :
    """
    Retrieve the flat-field (FF) time variation array for a given UVIS channel and spacecraft time.

    This function loads flat-field modifier data files corresponding to different spacecraft times and computes the
    flat-field modifier array (`arrmod`) for the specified channel at the given spacecraft time (`sctime`).
    It interpolates between the two closest calibration times to compute the flat-field modifier array.

    Parameters
    ----------
    channel : str
        The UVIS channel for which to get the calibration time variation. 'FUV' or 'EUV'.
    sctime : float
        The spacecraft time in seconds for which to compute the flat-field modifier.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (64, 1024) containing the flat-field modifier values.

    Notes
    -----
    - The function searches for data files matching the pattern '*ff_modifier*.dat'.
    - Spacecraft times are extracted from the filenames, assuming the time is embedded in positions -14 to -4 of the filename.
    - If the `sctime` is outside the calibration data range:
      - Before the earliest time: returns an array of ones.
      - After the latest time: returns the flat-field modifier from the last available file.
    - For times within the calibration data range, the flat-field modifier arrays are interpolated linearly in time.
    - The 62nd row of the modifier array to ones (arrmod[61, :] = 1).
    """
        
    
    calibration_dir = Path(env.calibration_dir)
    fmods = list(calibration_dir.glob(f"*{channel}*ff_modifier*.dat"))
    sctime_mods = np.array([int(f.stem[-10:]) for f in fmods])

    # Sort fmods according to sctime_mods
    sorted_indices = np.argsort(sctime_mods)
    sctime_mods    = sctime_mods[sorted_indices]
    fmods          = [fmods[i] for i in sorted_indices]

    if   sctime < sctime_mods[0]   : arrmod = np.ones((64, 1024), dtype=np.float32)
    elif sctime >= sctime_mods[-1] :
        with fmods[-1].open("rb") as f:
            arrmod = np.fromfile(f, dtype=np.float32, count=1024 * 64).reshape((64, 1024))
    
    else :
        time_index = np.searchsorted(sctime_mods, sctime)
        t1, t2 = sctime_mods[time_index-1], sctime_mods[time_index]

        with fmods[time_index - 1].open("rb") as f:
            arrmod1 = np.fromfile(f, dtype=np.float32, count=1024 * 64).reshape((64, 1024))
        with fmods[time_index].open("rb")     as f:
            arrmod2 = np.fromfile(f, dtype=np.float32, count=1024 * 64).reshape((64, 1024))
        arrmod = arrmod1+(sctime-t1)*(arrmod2-arrmod1)/(t2-t1)
        arrmod[61,:]=1
    return arrmod

def read_spica_ff(filename:str) :
    """
    Read a flat-field calibration file from SPICA observations to account for 'starburn event',
      and return its contents as a NumPy array.

    This function reads files 'FLATFIELD_XUV_POSTBURN.txt' or 'FLATFIELD_XUV_PREBURN.txt'
    and returns the data as a NumPy array reshaped to dimensions (64, 1024).

    Parameters
    ----------
    filename : str
        The path to the file to be read.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (64, 1024) containing the flat-field data.
    """

    # Initialize a list to store the values
    data = []

    # Read the file line by line, skipping the first line
    with open(filename, 'r') as file:
        lines = file.readlines()[1:]  # Ignore first line

    for line in lines:
        # Split the line and convert each number to float
        values = [float(value) for value in line.split()]
        data.extend(values)

    # Convert the list to a NumPy array and reshape to 64x1024
    return np.array(data).reshape(64, 1024)

def smooth_spectrum(spectrum, kernel, mode='nearest'):
    """
    Smooths spectral data by applying a 1D convolution along the spectral (last) dimension.
    
    Parameters
    ----------
    spectrum : numpy.ndarray
        Input data array. Expected shapes:
            - 1D: (nwl,)
            - 2D: (np, nwl)
            - 3D: (nt, np, nwl)
    kernel : numpy.ndarray
        1D convolution kernel.
    
    Returns
    -------
    numpy.ndarray
        The smoothed spectrum.
    
    Notes
    -----
    Convolution is performed with the 'nearest' mode for boundary conditions by default.
    See scipy.ndimage.convolve1d for more information.
    """

    if spectrum.ndim not in (1, 2, 3):
        raise ValueError("Input array must have at most 3 dimensions.")
    
    spectral_axis = spectrum.ndim - 1
    return convolve1d(spectrum, kernel, axis=spectral_axis, mode=mode)



