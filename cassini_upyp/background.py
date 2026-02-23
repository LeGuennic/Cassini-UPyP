import numpy as np

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

    # Identify transitions
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
    Compute the frequency histogram of a sequence of non-negative integers.

    This function returns an array of counts such that the element at index ``k``
    is the number of occurrences of value ``k`` in ``A``. 

    Parameters
    ----------
    A : array-like of int or None
        Input values. Values must be non-negative integers. If ``None``, the
        function returns ``0`` (legacy behavior).

    max_value : int or None, optional
        If provided, forces the output length to be at least ``max_value + 1``.
        If ``None``, the maximum of ``A`` is used.

    Returns
    -------
    numpy.ndarray or int
        If ``A`` is not ``None``, returns a 1D array of counts with length
        ``max_value + 1``. If ``A`` is ``None``, returns ``0`` (legacy behavior).

    Notes
    -----
    - Negative values are not supported by ``numpy.bincount`` and will raise an
    error.
    - If ``A`` is empty and ``max_value`` is ``None``, ``A.max()`` raises a
    ``ValueError``.

    Examples
    --------
    >>> histogram([0, 2, 2, 3])
    array([1, 0, 2, 1])

    >>> histogram([1, 1, 1], max_value=4)
    array([0, 3, 0, 0, 0])
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

    # Probability that a pixel is empty on the full detector
    q = (1.0 - 1.0 / L) ** N
    p = 1.0 - q

    x = np.arange(L + 1, dtype=float)
    s = np.zeros(L + 1, dtype=float)

    # Expected number of consecutive zeros of length x in a Bernoulli sequence of length L
    # (includes zeros at the edge + inside the array)
    # Valid for 1 <= x <= L-1
    s[1:L] = (q ** x[1:L]) * (2.0 * p + (L - x[1:L] - 1.0) * (p ** 2))

    # All-zero case: probability q^L, contributes exactly one run of length L
    s[L] = q ** L

    return s


def bg_fit(obs_histogram, detector_shape=1024):
    """
    Estimate the background count level by fitting a model gap histogram
    to the observed histogram with a least-squares criterion.

    The search is performed by iteratively narrowing the interval
    ``[low, up]`` around the current best candidate. At each iteration,
    the function evaluates a grid of candidate counts and selects a smaller
    bracket centered on the minimum chi-square.

    Parameters
    ----------
    obs_histogram : array-like
        Observed histogram to fit. It must be compatible in shape with the
        output of ``gap_histogram(count, L=detector_shape)`` (same length).

    detector_shape : int, optional
        Detector length ``L`` passed to ``gap_histogram`` and upper bound for
        the tested count values. Default is ``1024``.

    Returns
    -------
    int
        The count value that minimizes the sum of squared residuals between the
        model gap histogram and ``obs_histogram``.

    Notes
    -----
    - The objective function is an unweighted least-squares metric:
    ``chi2 = sum((model - obs)**2)``. No statistical weighting is applied.

    See Also
    --------
    :func:`gap_histogram` : Compute the theoretical gap histogram for a given count.

    """

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
        
        # Find intervals around the minimum chi2 to narrow the search
        iup,ilow = np.argmin(chi2_list)+1,np.argmin(chi2_list)-1

        if ilow==-1 : ilow=0
        if iup ==len(counts) : iup=len(counts)-1

        up, low = counts[iup],counts[ilow]

    return counts[np.argmin(chi2_list)]



def max_gap(bg_level, integration_time, n_wl=1024, SPECTRAL_BIN=1, SPATIAL_BIN=1, alpha=1e-4):
    """
    Estimate the maximum gap length for a given background level and integration time.

    Given an estimated background level and an integration time, this function
    computes the expected number of background counts per spectrum and derives the
    expected (analytical) gap-length histogram. It then returns the smallest gap
    length ``m`` such that the expected number of gaps of size greater than or
    equal to ``m`` is below a user-defined threshold ``alpha``.

    Large observed gaps beyond this threshold are
    interpreted as unlikely to arise from pure background noise and are therefore
    used as an indicator of signal loss.

    Parameters
    ----------
    bg_level : float
        Estimated background rate (counts per seconds, per spectral bin).
    integration_time : float
        Integration time in seconds.
    n_wl : int, optional
        Number of spectral bins (wavelength samples) in the spectrum. Default is
        ``1024``.
    SPECTRAL_BIN : int, optional
        Spectral binning factor. Default is ``1``.
    SPATIAL_BIN : int, optional
        Spatial binning factor. Default is ``1``.
    alpha : float, optional
        False-positive rate threshold on the expected number of long gaps.
        For example, ``alpha=1e-4`` means that, under the background-only model,
        a gap longer than the returned threshold would be expected in about
        1 spectrum out of 10,000 by chance. Default is ``1e-4``.

    Returns
    -------
    int
        Gap length threshold ``m_thr``. Spectra exhibiting gaps longer than this
        value can be flagged as a signal loss.

    Notes
    -----
    The expected total number of background counts per spectrum is estimated as::

        N = bg_level * integration_time * n_wl * SPATIAL_BIN * SPECTRAL_BIN

    The tail expectation is computed from the gap histogram ``h`` as::

        mu_ge[m] = sum_{k=m}^{inf} h[k]

    See Also
    --------
    :func:`gap_histogram` : Analytical gap-length histogram under a background-only model.
    """
    
    N = bg_level * integration_time * n_wl * SPATIAL_BIN * SPECTRAL_BIN
    h = gap_histogram(N, L=n_wl)
    mu_ge = np.cumsum(h[::-1])[::-1]
    m_thr = np.where(mu_ge <= alpha)[0][0]

    return m_thr


