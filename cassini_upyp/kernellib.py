import spiceypy as spice
from pathlib import Path

from .utils import env_config
env = env_config()
kernels_dir = env.kernels_dir

mk_path = Path(kernels_dir) / 'mk'


# KERNELS -----------------
def ckernel_covers_yd(ckernel:str, year:str, doy:str) :
    """
    Parse ckernel filename : YYDDD_YYDDD[ext].bc
    """
    
    sta, sto = ckernel[:11].split('_')
    ysta, dsta = map(int, (sta[:2], sta[2:]))
    ysto, dsto = map(int, (sto[:2], sto[2:]))
    target_date = int(year[-2:]) * 1000 + int(doy)
    
    return ysta * 1000 + dsta <= target_date <= ysto * 1000 + dsto

def ckernel_covers_et(ckernel:str, et:float) :

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

    # Get time coverages
    cover = spice.spkcov(spkernel, obj_id)

    for i in range(0, len(cover), 2):
        start_et = cover[i]
        end_et   = cover[i+1]
        if start_et <= et <= end_et:
            return True
    return False


def metakernel(et, save=False, savefile: str = None, filter_yd=None):
    """Build the list of kernels to load and optionally write a SPICE meta-kernel.
    """
    # Normalize base directories as Paths
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

    if len(ck) > 1:
        # Keep only files whose name contains 'ra'
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
        metakernel_path = mk_path / savefile
        metakernel_path.write_text(metakernel_content)

    return kernels_to_load


def yd_to_et(year, doy, hour=0, minute=0, second=0):
    """
    Convert UTC year and day-of-year to ephemeris time (ET, seconds past J2000 TDB).
    Accepts optional time-of-day. Uses ISO 8601 DOY format with 'Z' (UTC).
    """
    import os
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

