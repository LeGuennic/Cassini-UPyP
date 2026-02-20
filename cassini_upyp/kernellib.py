import spiceypy as spice
from pathlib import Path

from .utils import env_config
env = env_config()
kernels_dir = env.kernels_dir

mk_path = Path(kernels_dir) / 'mk'


# KERNELS -----------------
def ckernel_covers_yd(ckernel:str, year:str, doy:str) :
    """
    Return whether a CK kernel file spans a given year/day-of-year date.

    The CK filename is expected to follow the pattern ``YYDDD_YYDDD*.bc``, where
    the start and stop coverage dates are encoded as two-digit year (YY) and
    three-digit day-of-year (DDD).

    Parameters
    ----------
    ckernel : str
        CK filename (or basename).
    year : str
        Target year (e.g., ``"2007"`` or ``"07"``). Only the last two digits are
        used.
    doy : str
        Target day of year as a zero-padded string (e.g., ``"032"``).

    Returns
    -------
    bool
        ``True`` if the target date falls within the encoded coverage interval
        (inclusive), otherwise ``False``.
    """
    
    start, stop = ckernel[:11].split('_')
    ystart, dstart = map(int, (start[:2], start[2:]))
    ystop,  dstop  = map(int, (stop[:2],  stop[2:]))
    target_date = int(year[-2:]) * 1000 + int(doy)
    
    return ystart * 1000 + dstart <= target_date <= ystop * 1000 + dstop

def ckernel_covers_et(ckernel:str, et:float) :
    """
    Return whether a CK kernel provides coverage at a given ephemeris time.

    The function checks all objects contained in the CK file and tests whether
    the input ephemeris time ``et`` falls within any of their coverage intervals.

    Parameters
    ----------
    ckernel : str
        Path to the CK kernel file.
    et : float
        Target ephemeris time (seconds past J2000, TDB).

    Returns
    -------
    bool
        ``True`` if any object in the CK kernel covers ``et`` (inclusive),
        otherwise ``False``.
    """

    ids = spice.ckobj(ckernel)

    for obj_id in ids:

        # Get time coverages
        cover = spice.ckcov(ckernel, obj_id, needav=False, level='INTERVAL', tol=0.0, timsys='TDB')

        for i in range(0, len(cover), 2):
            start_et = cover[i]
            end_et   = cover[i+1]
            if start_et <= et <= end_et:
                return True
    return False

def spkernel_covers_et(spkernel:str, et:float, obj_id=-82) :
    """
    Return whether an SPK (Spacecraft and Planet) kernel provides coverage at a given ephemeris time.

    The function queries the time coverage for a specified object ID in the SPK
    file and checks whether the input ephemeris time ``et`` falls within any
    coverage interval.

    Parameters
    ----------
    spkernel : str
        Path to the SPK kernel file.
    et : float
        Target ephemeris time (seconds past J2000, TDB).
    obj_id : int, optional
        NAIF ID of the object whose coverage is queried. Default is ``-82``
        (Cassini spacecraft).

    Returns
    -------
    bool
        ``True`` if the SPK kernel covers ``et`` for the given object ID
        (inclusive), otherwise ``False``.
    """

    # Get time coverages
    cover = spice.spkcov(spkernel, obj_id)

    for i in range(0, len(cover), 2):
        start_et = cover[i]
        end_et   = cover[i+1]
        if start_et <= et <= end_et:
            return True
    return False


def metakernel(et, save=False, savefile: str = None, filter_yd=None):
    """
    Select the SPICE kernels needed for a given epoch and optionally write a meta-kernel file.

    The function inspects the local kernel tree (``kernels_dir``) and builds a list
    of kernel file paths to load with SPICE for the requested ephemeris time ``et``.
    It always includes the standard kernels (LSK, SCLK, FK, IK, PCK), then selects:

    - CK kernels whose coverage includes ``et`` (optionally pre-filtered by an
    encoded year/DOY interval in the CK filename),
    - SPK kernels whose coverage includes ``et`` for the default target object.

    If multiple CK candidates are found, the selection is further restricted to
    files whose name contains ``'ra'`` (reconstructed data).

    Optionally, the resulting list can be written as a SPICE meta-kernel file (``.tm``).

    Parameters
    ----------
    et : float
        Ephemeris time (seconds past J2000, TDB) used to select time-dependent kernels.
    save : bool, optional
        If ``True``, write a meta-kernel file containing the selected kernels.
        Default is ``False``.
    savefile : str or None, optional
        Output meta-kernel path to the file. If provided without extension, ``.tm`` is
        appended. If ``None``, a name is generated from ``filter_yd`` (``YY_DDD.tm``)
        or from ``et`` (``<int(et)>.tm``). Default is ``None``.
    filter_yd : tuple(str, str) or None, optional
        Optional pre-filter for CK filenames using a target (year, day-of-year).
        The CK filename is expected to encode a coverage interval (see
        ``ckernel_covers_yd``). If ``None``, all CK files are considered.
        Default is ``None``.

    Returns
    -------
    list of str
        List of kernel file paths (as strings) to be loaded with ``spice.furnsh``.

    Notes
    -----
    - The function temporarily loads the LSK and SCLK to allow CK coverage queries,
    then unloads them before returning.
    - Paths are returned as strings because SPICE expects string paths.

    See Also
    --------
    :func:`ckernel_covers_yd` : CK filename interval filter (YYDDD_YYDDD*.bc).
    :func:`ckernel_covers_et` : CK coverage test for a given ET.
    :func:`spkernel_covers_et` : SPK coverage test for a given ET.
    """

    kdir = Path(kernels_dir)

    # LSK, SCLK, FK, IK and PCK Kernels
    # ---------------------------------
    lsk  = kdir / 'lsk'  / 'naif0012.tls'      # Leap Second Kernel
    sclk = kdir / 'sclk' / 'cas00172.tsc'      # Spacecraft Clock Kernel
    fk   = kdir / 'fk'   / 'cas_v43.tf'        # Frame kernel
    ik   = kdir / 'ik'   / 'cas_uvis_v07.ti'   # Instrument kernel
    pck  = kdir / 'pck'  / 'pck00010.tpc'      # Planetary constants kernel



    # CK Kernels
    # ----------
    # Load lsk and sclk first (SPICE expects string paths)
    spice.furnsh(str(lsk))
    spice.furnsh(str(sclk))

    ckpath = kdir / 'ck'

    # Filter from the name of the file (use file names, not full paths)
    if filter_yd is not None:
        f_year, f_doy = map(str, filter_yd)
        ck1 = {
            p.name for p in ckpath.iterdir()
            if p.is_file() and ckernel_covers_yd(p.name, f_year, f_doy)
        }
    else:
        # Take all files in CK directory
        ck1 = [p.name for p in ckpath.iterdir() if p.is_file()]

    # Select kernels that actually cover ET
    ck = set()
    for k in ck1:
        ckernel = ckpath / k
        if ckernel_covers_et(str(ckernel), et):
            ck.add(ckernel)

    # If multiple CK candidates remain, choose the reconstructed kernel ('r')
    if len(ck) > 1:
        ck = {e for e in ck if 'ra' in e.name}



    # SPK Kernels
    # -----------
    spkpath = kdir / 'spk'

    spk = set()
    for p in spkpath.iterdir():
        if not p.is_file() or p.suffix != '.bsp':
            continue
        if spkernel_covers_et(str(p), et):
            spk.add(p)

    spk.add(spkpath / 'sat427.bsp')

    # Unload LSK/SCLK after checks
    spice.unload(str(lsk))
    spice.unload(str(sclk))

    # List of kernels to load (as strings for SPICE)
    kernels_to_load = list(map(str, [
        lsk,
        sclk,
        fk,
        ik,
        pck,
        *spk,
        *ck,
    ]))

    if save:
        # Compute savefile name
        if savefile is None:
            if filter_yd is not None:
                savefile = f"{f_year}_{f_doy}.tm"
            else:
                savefile = f"{int(et)}.tm"
        else:
            if '.' not in savefile:
                savefile += '.tm'

        # Build meta-kernel content
        metakernel_content = "\\begindata\nKERNELS_TO_LOAD = (\n"
        for kernel in kernels_to_load:
            metakernel_content += f"    '{kernel}',\n"
        metakernel_content += ")\n\\begintext"

        # Write file with Path.write_text
        metakernel_path = savefile
        metakernel_path.write_text(metakernel_content)

    return kernels_to_load


def yd_to_et(year, doy, hour=0, minute=0, second=0):
    """
    Convert a UTC calendar date given as (year, day-of-year) to ephemeris time (ET).

    The function builds an ISO-8601 day-of-year UTC string (``YYYY-DDDThh:mm:ss.sssZ``)
    and converts it to ephemeris time using SPICE ``str2et``.

    Parameters
    ----------
    year : int or str
        UTC year (e.g., ``2009`` or ``"2009"``).
    doy : int or str
        UTC day of year in ``[1, 365]`` or ``[1, 366]`` for leap years.
    hour : int or str, optional
        UTC hour in ``[0, 23]``. Default is ``0``.
    minute : int or str, optional
        UTC minute in ``[0, 59]``. Default is ``0``.
    second : float or int or str, optional
        UTC seconds in ``[0.0, 60.0)``. Default is ``0``.

    Returns
    -------
    float
        Ephemeris time in seconds past J2000 (TDB).

    Raises
    ------
    ValueError
        If the day-of-year is out of range for the given year, or if the time of day
        fields are invalid.

    Notes
    -----
    This function requires access to the Leap Seconds Kernel (LSK).
    """

    import calendar
    import spiceypy as spice


    y = int(year)
    d = int(doy)
    h = int(hour)
    m = int(minute)
    s = float(second)

    # Validate DOY with leap-year awareness
    max_doy = 366 if calendar.isleap(y) else 365
    if not (1 <= d <= max_doy):
        raise ValueError(f"doy must be in [1, {max_doy}] for year {y}")
    if not (0 <= h < 24 and 0 <= m < 60 and 0.0 <= s < 60.0):
        raise ValueError("invalid time of day")

    # Build an ISO 8601 DOY string; 'Z' marks UTC (accepted by STR2ET)
    # Example: "2009-274T00:00:00.000Z"
    utc_str = f"{y:04d}-{d:03d}T{h:02d}:{m:02d}:{s:06.3f}Z"

    lsk_path = Path(kernels_dir) / "lsk" / "naif0012.tls"

    # Load LSK just for the conversion; ensure unload even if parsing fails
    spice.furnsh(str(lsk_path))
    try:
        et = spice.str2et(utc_str)
    finally:
        spice.unload(str(lsk_path))

    return et

