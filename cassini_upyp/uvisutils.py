from __future__ import annotations
from typing import Literal, Sequence
from numpy.typing import ArrayLike

import numpy as np
from pathlib import Path

from scipy.integrate   import simpson
from scipy.interpolate import PchipInterpolator
from scipy.io          import readsav
from scipy.ndimage     import convolve1d

from .utils import env_config
env = env_config()


# UNCERTAINTY
def poisson_error(
        x: ArrayLike,
        bound: Literal["inf", "sup", "lower", "upper"],
        sigma: float = 1.0
) -> float | np.ndarray:
    """
    Return one Garwood bound (lower/upper) for Poisson counts.

    Parameters
    ----------
    x : int or array-like
        Observed counts (must be >= 0, finite).
    bound : {"inf", "sup", "lower", "upper"}
        Which bound to return. Aliases: "lower"->"inf", "upper"->"sup".
    sigma : float, optional
        Two-sided Gaussian-equivalent level for the central interval (default: 1.0).

    Returns
    -------
    float or np.ndarray
        Requested bound with the same shape as `x` (float if scalar input).

    Notes
    -----
    Uses the Garwood (chi-square) construction:

    U = 0.5 * chi2.ppf(1 - alpha/2, 2*(x+1))

    L = 0.5 * chi2.ppf(alpha/2,     2*x)   (with L=0 for x=0)

    where alpha = 1 - (Phi(sigma) - Phi(-sigma)).
    """
    
    from scipy.stats import chi2, norm

    # Normalize and validate bound
    b = bound.lower().strip()
    if b in {"inf", "lower"}:
        b = "inf"
    elif b in {"sup", "upper"}:
        b = "sup"
    else:
        raise ValueError("Invalid bound type. Use 'inf'/'sup' or 'lower'/'upper'.")

    # Convert and validate x
    x_arr = np.asarray(x)

    if not np.all(np.isfinite(x_arr)):
        raise ValueError("`x` must contain finite values only.")
    if np.any(x_arr < 0):
        raise ValueError("`x` must be >= 0.")
    
    # Validate sigma
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("`sigma` must be a finite, non-negative float.")
    
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

def correction_factor(N: ArrayLike, log: bool = True) -> float | np.ndarray:
    """
    Return the bias-correction factor κ(N) for the sample standard deviation.

    For N Gaussian samples, the usual sample standard deviation s
    (with denominator N - 1) underestimates the true σ. This function
    returns κ(N) such that κ(N) * s is an unbiased estimator of σ
    under the Gaussian assumption. As N → ∞, κ(N) → 1.

    Parameters
    ----------
    N : int or array-like of int
        Sample size(s), expected to be ≥ 2.
    log : bool, optional
        If True (default), use the log-gamma function for numerical
        stability, especially for large values of N. If False, use the regular gamma.

    Returns
    -------
    float or numpy.ndarray
        The correction factor κ(N), with the same shape as `N`.
        Returns a float if `N` is scalar.
    """
    from scipy.special import gammaln, gamma

    N_arr = np.asarray(N, dtype=float)
    scalar_input = np.isscalar(N)

    if log:
        log_ratio = gammaln((N_arr - 1) / 2) - gammaln(N_arr / 2)
        res = np.exp(log_ratio) * np.sqrt((N_arr - 1) / 2)
    else:
        res = (
            gamma((N_arr - 1) / 2)
            / gamma(N_arr / 2)
            * np.sqrt((N_arr - 1) / 2)
        )

    if scalar_input:
        return float(np.asarray(res).item())
    return np.asarray(res, dtype=float)


# SPECTRUM
def UVIS_WL(channel: Literal["FUV", "EUV"], bin: int = 1) -> np.ndarray:
    """
    Calculate the wavelength array for the specified UVIS channel.

    The wavelengths are computed from the optical model of the FUV/EUV
    spectrograph for the 1024 detector columns. If `bin` > 1, the
    wavelengths are averaged over contiguous groups of `bin` columns.

    Parameters
    ----------
    channel : {"FUV", "EUV"}
        The UVIS channel for which to calculate wavelengths.
        - 'FUV': Far     Ultraviolet.
        - 'EUV': Extreme Ultraviolet.

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

def integrate_spectrum(
    wl: ArrayLike,
    s: ArrayLike,
    wl_range: tuple[float, float] | None = None,
    method: Literal["simpson", "trapezoid", "trapz"] = "simpson",
    axis: int = 0,
    uncertainty: bool = False
) -> float | np.ndarray:
    """
    Integrate a spectrum over wavelength with optional quadratic error propagation.

    Parameters
    ----------
    wl : 1D array-like
        Wavelength grid. Must have length equal to ``s.shape[axis]``.
    s : array-like
        Spectral data (if ``uncertainty=False``) or uncertainty array
        (if ``uncertainty=True``). The dimension along `axis` must match
        the length of `wl`.
    wl_range : (float, float) or None, optional
        Integration bounds in wavelength as (min_wl, max_wl). If None,
        the integral is computed over the full `wl` range.
    method : {"simpson", "trapezoid", "trapz"}, optional
        Numerical integration method:
        - "simpson"   : use ``scipy.integrate.simpson``,
        - "trapezoid" : use ``numpy.trapezoid``,
        - "trapz"     : alias for "trapezoid".
        Default is "simpson".
    axis : int, optional
        Axis along which to integrate. Default is 0.
    uncertainty : bool, optional
        If False (default), `s` is interpreted as the spectrum and the
        function returns the integral of `s(wl)`.
        If True, `s` is interpreted as an uncertainty per wavelength bin.
        The function then returns the integrated uncertainty, computed
        by propagating variances under the integral (quadratic sum).

    Returns
    -------
    float or numpy.ndarray
        Integrated value(s). A Python float is returned if all inputs
        are effectively scalar along the integration axis.

    Raises
    ------
    ValueError
        If the wavelength and spectrum sizes do not match along `axis`,
        or if an unknown integration method is requested.
    """
    
    wl = np.asarray(wl)
    s  = np.asarray(s)

    if wl.size != s.shape[axis]:
        raise ValueError("The wavelength and spectrum arrays must have the same shape.")
    

    # Select wavelength range if wl_range is specified
    if wl_range is not None:
        mask   = (wl >= wl_range[0]) * (wl <= wl_range[1])
        wl_sub = wl[mask]
        s_sub  = np.take(s, np.where(mask)[0], axis=axis)
    else:
        wl_sub = wl
        s_sub  = s

    # Normalize method aliases
    if method == "trapz":
        method = "trapezoid"

    # Perform integration depending on the method
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
            
            # Expand delta_wl so it broadcasts along all axes except `axis`
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
    Interpolate NaNs along the last axis of a 1-D or 2-D array.

    For a 1-D input, interpolation is done along the single dimension.
    For a 2-D input with shape (M, N), each row (size N) is treated
    independently and interpolated over its NaN entries.

    Parameters
    ----------
    arr : array-like
        Input data containing NaNs. Cast to float internally.
    method : {"linear", "pchip"}, optional
        Interpolation scheme to use:

        - "linear": piecewise linear interpolation using ``np.interp``.
        - "pchip" : monotonic piecewise cubic interpolation using
          ``scipy.interpolate.PchipInterpolator``.
          
        Default is "linear".

    Returns
    -------
    np.ndarray
        Array with NaNs replaced by interpolated values. The output has
        dtype float and the same shape as the input. If the input was
        1-D, the output is 1-D.

    Raises
    ------
    ValueError
        If the input has more than 2 dimensions or if `method` is invalid.
    """


    # Normalize input
    a = np.asarray(arr, dtype=float)
    if a.ndim > 2:
        raise ValueError("Only 1-D or 2-D arrays are supported.")
    was_1d = a.ndim == 1
    if was_1d:
        a = a[np.newaxis, :]          # shape -> (1, N)

    x = np.arange(a.shape[1])         # common x-axis for all rows
    out = a.copy()

    # Choose interpolation routine
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

def smooth_spectrum(spectrum: ArrayLike, kernel: ArrayLike, mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] ='nearest') -> np.ndarray:
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
    mode : {"reflect", "constant", "nearest", "mirror", "wrap"}, optional
        Boundary handling mode passed to ``scipy.ndimage.convolve1d``.
        Default is "nearest", see scipy.ndimage.convolve1d for more information.
    
    Returns
    -------
    numpy.ndarray
        The smoothed spectrum with the same shape as `spectrum`.
    """

    spectrum = np.asarray(spectrum)
    kernel   = np.asarray(kernel, dtype=float)

    if spectrum.ndim not in (1, 2, 3):
        raise ValueError("Input array must have at most 3 dimensions.")
    
    spectral_axis = spectrum.ndim - 1
    return convolve1d(spectrum, kernel, axis=spectral_axis, mode=mode)



# CALIBRATION

def uvis_lab_calibration(channel: Literal["FUV", "EUV"], filename: str | Path | None = None) -> dict[str, np.ndarray]:
        """
        Read laboratory calibration data for the specified UVIS channel and return sensitivity information.

        The file contains the full-slit, low-resolution monochromatic
        extended-source sensitivity measured in the laboratory (1997,
        updated 1999), in units of (counts s-1) / (kilorayleigh).

        Parameters
        ----------
        channel : {"FUV", "EUV"}
            The UVIS channel for which to read the calibration data.
        filename : str or pathlib.Path, optional
            Path to the calibration data file. If None (default), the file
            name is constructed as "{channel}_1999_Lab_Cal.dat" and looked
            up in the default calibration files directory.

        Returns
        -------
        dict of str -> numpy.ndarray
            A dictionary with three 1D arrays:
            - "WAVELENGTH"        : wavelength grid (Å),
            - "SENSITIVITY"       : sensitivity (counts s⁻¹ / kR),
            - "SENSITIVITY_ERROR" : uncertainty on the sensitivity.
        """

        if filename is None :
            filename = Path(channel+'_1999_Lab_Cal.dat')
            filename = env.calibration_dir / filename

        data = np.loadtxt(filename, skiprows=1)

        return {'WAVELENGTH'        : np.concatenate((data[:,0], data[:,3])),
                'SENSITIVITY'       : np.concatenate((data[:,1], data[:,4])),
                'SENSITIVITY_ERROR' : np.concatenate((data[:,2], data[:,5]))}

def get_cal_time_variation(channel: Literal["FUV", "EUV"], sctime: float) -> np.ndarray:
    """
    Retrieve the spectral modulation array for a given UVIS channel and spacecraft time.

    This function reads calibration trending data from the IDL 'uvis_calibration_trending_v01_data.sav' computed from
    IDL routine uvis_calibration_trending_v01.pro and computes the spectral modulation
    for the specified UVIS channel at a given spacecraft time (`sctime`).

    The spectral modulation is interpolated linearly in time between
    the two closest calibration epochs.

    Parameters
    ----------
    channel : {"FUV", "EUV"}
        The UVIS channel for which to get the calibration time variation.
    sctime : float
        The spacecraft time in seconds for which to compute the spectral modulation.

    Returns
    -------
    numpy.ndarray
        Spectral modulation array of length 1024 for the requested
        `channel` and `sctime`.

    Notes
    -----
    - The calibration trending data is read from 'calibration_files/uvis_calibration_trending_v01_data.sav'.
    - If `sctime` is earlier than the first calibration time, an array
      of ones is returned.
    - If `sctime` is later than the last calibration time, the last
      available modulation ratio is used.
    - Between two calibration times, the modulation is interpolated
      linearly in time.
    """

    cal_file = Path(env.calibration_dir) / 'uvis_calibration_trending_v01_data.sav'
    cal_trend = readsav(cal_file)


    if channel == "FUV":
        arr = cal_trend.arr_fuv
    elif channel == "EUV":
        arr = cal_trend.arr_euv
    else:
        raise ValueError(f"Unknown UVIS channel: {channel!r} (expected 'FUV' or 'EUV').")

    sctime_mods = [arr[k].desc.sctime_sec_start[0] for k in range(len(arr))]

    if   sctime <  sctime_mods[0]  : specmod = np.ones(1024)
    elif sctime >= sctime_mods[-1] : specmod = arr[-1].ratio
    else :
        time_index = np.searchsorted(sctime_mods, sctime)
        t1, t2 = sctime_mods[time_index-1], sctime_mods[time_index]
        specmod1 = arr[time_index-1].ratio
        specmod2 = arr[time_index  ].ratio
        specmod  = specmod1 + (specmod2 - specmod1) * (sctime - t1) / (t2 - t1)
    return specmod

def get_ff_time_variation(channel: Literal["FUV", "EUV"], sctime: float) -> np.ndarray:
    """
    Retrieve the flat-field (FF) time variation array for a given UVIS channel and spacecraft time.

    This function loads flat-field modifier data files corresponding to different spacecraft times and computes the
    flat-field modifier array for the specified channel at the given spacecraft time (`sctime`).
    It interpolates between the two closest calibration times to compute the flat-field modifier array.

    Parameters
    ----------
    channel : {"FUV", "EUV"}
        The UVIS channel for which to get the calibration time variation. 'FUV' or 'EUV'.
    sctime : float
        The spacecraft time in seconds for which to compute the flat-field modifier.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (64, 1024) containing the flat-field modifier values, dtype float32.

    Notes
    -----
    - Files are searched with the pattern ``f"*{channel}*ff_modifier*.dat"`` 
      inside the calibration files directory.
    - Spacecraft times are extracted from the last 10 characters of the
      file name and interpreted as integers.
    - If `sctime` is earlier than the first available time, an array of
      ones is returned.
    - If `sctime` is later than the last available time, the modifier
      from the last available file is returned.
    - For times strictly within the calibration range, the modifier is
      obtained by linear interpolation between the two nearest files in
      time; in this case, the 62nd row is explicitly set to one
      (``arrmod[61, :] = 1``).

    Raises
    ------
    FileNotFoundError
        If no flat-field modifier files are found for the specified channel.
    """
        
    calibration_dir = Path(env.calibration_dir)
    fmods = list(calibration_dir.glob(f"*{channel}*ff_modifier*.dat"))
    if not fmods:
        raise FileNotFoundError(
            f"No flat-field modifier files found for channel {channel!r} "
            f"in {calibration_dir}"
        )
    
    # Extract spacecraft times encoded in the filename
    sctime_mods = np.array([int(f.stem[-10:]) for f in fmods])

    # Sort fmods according to sctime_mods
    sorted_indices = np.argsort(sctime_mods)
    sctime_mods    = sctime_mods[sorted_indices]
    fmods          = [fmods[i] for i in sorted_indices]


    if   sctime < sctime_mods[0]   :
        arrmod = np.ones((64, 1024), dtype=np.float32)
    elif sctime >= sctime_mods[-1] :
        with fmods[-1].open("rb") as f:
            arrmod = np.fromfile(f, dtype=np.float32, count=1024 * 64).reshape((64, 1024))
    
    else : # Linear interpolation in time between two calibration files
        time_index = np.searchsorted(sctime_mods, sctime)
        t1, t2 = sctime_mods[time_index-1], sctime_mods[time_index]

        with fmods[time_index - 1].open("rb") as f:
            arrmod1 = np.fromfile(f, dtype=np.float32, count=1024 * 64).reshape((64, 1024))
        with fmods[time_index].open("rb")     as f:
            arrmod2 = np.fromfile(f, dtype=np.float32, count=1024 * 64).reshape((64, 1024))
        arrmod = arrmod1+(sctime-t1)*(arrmod2-arrmod1)/(t2-t1)
        arrmod[61,:]=1
    return arrmod

def read_spica_ff(filename:str | Path) -> np.ndarray:
    """
    Read a SPICA flat-field calibration file.

    This reads flat-field maps derived from SPICA observations used to
    correct the UVIS detector after the starburn event, e.g.
    ``FLATFIELD_XUV_PREBURN.txt`` or ``FLATFIELD_XUV_POSTBURN.txt``.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the flat-field file.

    Returns
    -------
    numpy.ndarray
        Flat-field array of shape (64, 1024), dtype float.
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


# DATA BINS UTILITIES
def list_ndarray(shape: tuple[int, ...]) -> np.ndarray:
    """
    Create a NumPy array (dtype=object) where each cell is initialized as an independant empty list.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array to create.

    Returns
    -------
    np.ndarray
        A NumPy array of the specified shape where each cell is an empty Python list.

    Examples
    --------
    Create a 2D array of lists for two properties:

    >>> from cassini_upyp.utils import list_ndarray
    >>> shape = (3, 2)
    >>> bins = list_ndarray(shape)
    >>> bins.shape
    (3, 2)

    Append data points to a given bin:

    >>> bins[0, 0].append(42.0)
    >>> bins[0, 0]
    [42.0]
    """

    bins_array = np.empty(shape, dtype=object)
    
    # Initialize each cell with an empty list.
    for index in np.ndindex(shape):
        bins_array[index] = []
    return bins_array

def find_bin_index(value: float | ArrayLike, boundaries: Sequence[float], mode: Literal['center', 'all'] = "center", modulo: float = None) -> int | None:
    """
    Determine the bin index for a given valueerty value (or array of values) relative to the provided boundaries.

    In 'center' mode, 'value' is expected to be a scalar value.
    In 'all' mode, 'value' is expected to be an array; all values must fall within the same bin.

    The binning convention is half-open: [edges[i], edges[i+1]),
    so the last edge is excluded.

    Parameters
    ----------
    value : scalar or array-like
        The valueerty value(s) for which to determine the bin index.
    boundaries : sequence of float
        A sorted list of bin edges.
    mode : {"center", "all"}, optional
        The mode of operation:
        - 'center': use a single representative value.
        - 'all': require that all values in the pixel fall within the same bin.
        Default is "center".
    modulo : float, optional
        If provided, a lower boundary that is higher than the upper boundary will be considered valid,
        and the binning will be treated as cyclic with the given period.
        For example, a boudary of [350, 10] with modulo=360 would mean that values in [350, 360) and [0, 10) belong to the same bin.

    Returns
    -------
    int or None
        The bin index if valid; otherwise, None.
    """
    
    edges = np.array(boundaries)

    if mode == 'center':
        # Check that the value is within the interval [edges[0], edges[-1])
        if (value < edges[0] or value >= edges[-1]) and modulo is None:
            return None

        for i in range(len(edges)-1):
            if modulo is not None and edges[i] > edges[i+1]: # Handle cyclic bin
                if (( value >= 0        and value < edges[i+1] ) or
                    ( value >= edges[i] and value < modulo     )):
                    return i
            elif edges[i] <= value < edges[i+1]:
                return i
            
    elif mode == 'all':
        arr = np.asarray(value)
        if arr.size == 0 or ((np.min(arr) < edges[0] or np.max(arr) >= edges[-1]) and modulo is None):
            return None
        
        for i in range(len(edges)-1):
            if modulo is not None and edges[i] > edges[i+1]: # Handle cyclic bin
                if np.all(((arr >= 0) & (arr < edges[i + 1])) |
                        ((arr >= edges[i]) & (arr < modulo))):
                    return i
            elif np.all((edges[i] <= arr) & (arr < edges[i+1])):
                return i

