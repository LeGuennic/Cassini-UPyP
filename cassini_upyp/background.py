import numpy as np
import math
import time
import os

def bin_array(arr, SPE_UL=None, SPE_LR=None, BIN=1, mean=False):
    """
    Bin a 2D array spatially and spectrally over a defined sub-region.

    This function extracts a sub-region from the input 2D array using the provided
    corner indices. It then applies spatial and spectral binning, effectively reducing
    the resolution by grouping pixels into bins and computing their average values.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array (e.g., a sensitivity map).
    SPA_UL : int
        Upper-left (inclusive) spatial index (row) of the sub-region.
    SPA_LR : int
        Lower-right (inclusive) spatial index (row) of the sub-region.
    SPE_UL : int
        Upper-left (inclusive) spectral index (column) of the sub-region.
    SPE_LR : int
        Lower-right (inclusive) spectral index (column) of the sub-region.
    SPATIAL_BIN : int
        The spatial bin size.
    SPECTRA_BIN : int
        The spectral bin size.

    Returns
    -------
    np.ndarray
        A 2D array containing the binned data. Each output element represents the average
        value of a SPATIAL_BIN x SPECTRA_BIN block from the original sub-region.
    """

    if SPE_UL is None : SPE_UL=0
    if SPE_LR is None : SPE_LR=len(arr)-1

    # Extract the specified sub-region
    arr_win = arr[SPE_UL:SPE_LR+1]
    width = len(arr_win)

    # Compute the number of bins in each dimension
    nbins_width  = math.ceil(width  / BIN)

    # Create index arrays for reduceat
    spe_indices = np.arange(0, nbins_width  *BIN, BIN)

    # Perform binning using np.add.reduceat twice:
    # First along the spatial axis (rows), then along the spectral axis (columns)

    if not mean : BIN=1
    binned = np.add.reduceat(
                arr_win,
                spe_indices
             ) / BIN

    return binned



# FITTING
def gaps(A):
    """
    Return the lengths of all consecutive zero sequences in the input A.

    Parameters
    ----------
    A : array-like
        The input array, which is an array of integers (0 or non-zero).

    Returns
    -------
    list or None
        A list of lengths of each consecutive zero sequence if any exist.
        If no such sequence exists, returns None.
        If all values are zero, returns [len(A)].

    Examples
    --------
    >>> gaps([0, 0, 1, 0])
    [2, 1]
    >>> gaps([1, 2, 3])
    None
    >>> gaps([0, 0, 0])
    [3]
    """
    A = np.array(A, dtype=int) == 0  # Convert directly to a boolean array

    # If all zeros
    if A.all():
        return np.array([len(A)])

    # If no zeros at all
    if not A.any():
        return None

    # Identify transitions using diff
    diff = np.diff(A.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # If the sequence starts with zeros
    if A[0]:
        starts = np.r_[0, starts]

    # If the sequence ends with zeros
    if A[-1]:
        ends = np.r_[ends, len(A)]

    result = ends - starts
    return result if len(result) > 0 else None


def histogram(A, max_value=None):
    """
    Compute the histogram of an array of integers.

    Parameters
    ----------
    A (array-like):
        Input array of integers.

    Returns
    -------
    numpy.ndarray:
        An array where each index represents a value 
        and its content is the frequency of that value in A.
    """

    if A is None :
        return 0
    A = np.array(A, dtype=int)
    if max_value is None :
        max_value = A.max()
    histogram = np.bincount(A, minlength=max_value + 1)

    return histogram



def gap_histogram(N: float, L: int) -> np.ndarray:
    """
    Analytical per-sample histogram of zero-run lengths within a window of length L_window,
    when 'N' hits are thrown uniformly over a full detector of length L_full.

    Parameters
    ----------
    N : float
        Total hits per sample on the *full* detector (before truncation).
    L_window : int
        Window length (after wl_range slicing), i.e. obs_histogram.size - 1.
    L_full : int
        Full detector length over which hits are distributed (e.g. 1024).

    Returns
    -------
    s : ndarray, shape (L_window+1,)
        Expected number of zero-runs of length x per sample inside the window.
    """

    # Probability that a pixel is empty on the full detector (marginally)
    q = (1.0 - 1.0 / L) ** N
    p = 1.0 - q

    x = np.arange(L + 1, dtype=float)
    s = np.zeros(L + 1, dtype=float)

    # Expected number of zero-runs of length x in a Bernoulli sequence of length L
    # (includes boundary runs + internal runs)
    # Valid for 1 <= x <= L-1
    s[1:L] = (q ** x[1:L]) * (2.0 * p + (L - x[1:L] - 1.0) * (p ** 2))

    # All-zero window: probability q^L, contributes exactly one run of length L
    s[L] = q ** L

    return s


def bg_fit(obs_histogram, detector_shape=1024):


    up,low = detector_shape,1
    dc = np.inf
    while dc>1:
        dc = (up-low)//10
        if dc==0 : dc=1

        counts = np.arange(low,up,dc, dtype=int)
        if up not in counts : counts = np.append(counts, up)

        chi2_list = []
        for count in counts :

            s = gap_histogram(count, L=detector_shape)
            chi2 = np.sum(
                ( s - obs_histogram )**2
            )

            chi2_list.append(chi2)

        iup,ilow = np.argmin(chi2_list)+1,np.argmin(chi2_list)-1

        if ilow==-1 : ilow=0
        if iup ==len(counts) : iup=len(counts)-1

        up, low = counts[iup],counts[ilow]

    return counts[np.argmin(chi2_list)]#/detector_shape/exposition/kwargs['SPATIAL_BIN']



def max_gap(bg_level, integration_time, n_wl=1024, SPECTRAL_BIN=1, SPATIAL_BIN=1, alpha=1e-4):
    
    N = bg_level * integration_time * n_wl * SPATIAL_BIN * SPECTRAL_BIN
    h = gap_histogram(N, L=n_wl)
    mu_ge = np.cumsum(h[::-1])[::-1]
    m_thr = np.where(mu_ge <= alpha)[0][0]

    return m_thr


