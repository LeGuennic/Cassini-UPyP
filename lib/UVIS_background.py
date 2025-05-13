import numpy as np
import random
import math

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



def random_noise(n, N):
    """
    Generate a 1D array of size n where N random increments
    are distributed across its elements.

    Parameters
    ----------
    n (int):
        Number of rows in the array.
    N (int):
        Total number of increments to distribute.

    Returns
    -------
    numpy.ndarray:
        A (n, m) array with distributed increments.
    """

    # Génération de tous les indices d'un coup
    indices = np.random.randint(0, n, size=N)

    # Comptage des occurrences de chaque index
    A = np.bincount(indices, minlength=n)

    # Reshape en matrice n x m
    return A

    
def simulate_histogram(count, n_iter, detector_shape=1024,
                       SPE_UL=0, SPE_LR=1023, SPECTRA_BIN=1, SPATIAL_BIN=1, wl_index=None) :

    if wl_index is None : h = np.zeros(SPE_LR-SPE_UL+1)
    else : h = np.zeros(wl_index[1]-wl_index[0]+1)

    
    for i in range (n_iter) :
        sim_sensor = np.zeros(detector_shape, dtype=int)
        for j in range(SPATIAL_BIN) :
            sim_sensor += random_noise(detector_shape, count)
        sim_sensor = bin_array(sim_sensor, SPE_UL, SPE_LR, SPECTRA_BIN)

        if wl_index is not None :
            sim_sensor = sim_sensor[wl_index[0] : wl_index[1]]

            h += histogram(
                gaps(sim_sensor), max_value=wl_index[1]-wl_index[0]
            )
        else :
            h += histogram(
                gaps(sim_sensor), max_value=SPE_LR-SPE_UL
            )
    
    h /= n_iter

    return h



def bg_fit(obs_histogram, n_iter, detector_shape=1024, exposition=1, **kwargs) :
    

    
    up,low = 1000,1
    dc = np.inf
    while dc>1:
        dc = (up-low)//10
        if dc==0 : dc=1

        counts = np.arange(low,up,dc, dtype=int)
        if up not in counts : counts = np.append(counts, up)

        chi2_list = []
    
        for count in counts :
            s = simulate_histogram(count, n_iter, detector_shape, **kwargs)
            
            chi2 = np.sum(
                ( s - obs_histogram )**2
            )


            chi2_list.append(chi2)

        iup,ilow = np.argmin(chi2_list)+1,np.argmin(chi2_list)-1
        if ilow==-1 : iup=0
        if iup ==len(counts) : iup=len(counts)-1

        up, low = counts[iup],counts[ilow]

    return counts[np.argmin(chi2_list)]/detector_shape/exposition



def do_bg_fit(i_fit, H, shape0, expo_time, spec_start, spec_stop, spec_bin, spat_bin, wl_index):
    result = bg_fit(
        H,
        shape0,
        exposition=expo_time,
        SPE_UL=spec_start,
        SPE_LR=spec_stop,
        SPECTRA_BIN=spec_bin,
        SPATIAL_BIN=spat_bin,
        wl_index=wl_index
    )
    return result



def max_gap(bg_level, integration_time, n_wl=1024, n_sim=200, SPATIAL_BIN=1, **kwargs) :
    l=[]

    for _ in range(n_sim) :
        S = np.zeros(n_wl, dtype=int)
        
        for __ in range(SPATIAL_BIN) :
            S += random_noise(n_wl, int(bg_level*integration_time*n_wl))
        
        S = bin_array(S, **kwargs)

        # If no gaps are in S (because binning is too high for example)
        if 0 not in S : return 0

        l.append(max(gaps(S)))

    return max(l)+3*np.std(l)


# CHANGE THIS
# It's only spectral pixels, no need to have 64x1024, just 1024
# cuz we only take single spatial pixel each time



# np.set_printoptions(threshold=np.inf)
# # print(random_noise(100,20))
# import matplotlib.pyplot as plt

# s=simulate_histogram(327, 15, wl_index=(50,60))
# print(len(s))



# fl=[]
# flr=[]
# for k in range(50) :
    
#     f,r=bg_fit(s, 15)

#     fl.append(np.argmin(f))
#     flr.append(r[np.argmin(f)])
#     print(k)

# fl=np.array(fl)

# mu=np.mean(fl)
# mstd=np.std(fl)*3


# fig, ax1 = plt.subplots()
# ax1.plot(f)
# ax2 = ax1.twinx()
# ax2.plot(r, color='lightblue')
# ax1.plot([0,1000],[min(f)+1]*2)


# # Génération d'un échantillon de points
# x = np.linspace(0,1023, 100000)

# # Calcul de la densité de probabilité
# y = (1 / (mstd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / mstd) ** 2)
# print(mu, mstd, np.mean(flr))
# ax1.plot(x,y*100000)

# plt.show()