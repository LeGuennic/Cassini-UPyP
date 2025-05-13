import spiceypy as spice
import os

from config.env  import kernels_dir

mk_path = os.path.join(kernels_dir, 'mk')


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



def metakernel(et, save=False, savefile:str=None, filter_yd=None) :
    if filter_yd is not None :
        f_year, f_doy = filter_yd
        f_year, f_doy = str(f_year), str(f_doy)
    

    # LSK, SCLK, FK, IK and PCK Kernels
    #----------------------------------
    lsk  = os.path.join(kernels_dir, 'lsk',  "naif0012.tls")       # Leap Second Kernel
    sclk = os.path.join(kernels_dir, 'sclk', "cas00172.tsc")       # Spacecraft Clock Kernel
    fk   = os.path.join(kernels_dir, 'fk',   "cas_v43.tf")         # Fame kernel
    ik   = os.path.join(kernels_dir, 'ik',   "cas_uvis_v07.ti")    # UVIS kernel
    pck  = os.path.join(kernels_dir, 'pck',  "pck00010.tpc")       # Planetary constant kernel




    # CK Kernels
    #-----------

    # Load lsk and sclk
    spice.furnsh(lsk)
    spice.furnsh(sclk)

    ckpath = os.path.join(kernels_dir, 'ck')

    # Filter from the name of the file
    if filter_yd is not None :
        ck1 = set()
        for ckernel in os.listdir(ckpath) :
            if ckernel_covers_yd(ckernel, f_year, f_doy) : ck1.add(ckernel)
    else : ck1 = ckpath


    # Select kernels that actually covers ET
    ck=set()
    for k in ck1:
        ckernel=os.path.join(ckpath, k)

        if ckernel_covers_et(ckernel, et) :
            ck.add(ckernel)

    if len(ck) >1 :
        ck={e for e in ck if 'ra' in e}
    

    # SPK Kernels
    #-----------

    spkpath = os.path.join(kernels_dir, 'spk')


    spk = set()
    for k in os.listdir(spkpath):
        if not k.endswith('.bsp'): continue

        spkernel = os.path.join(spkpath, k)
        if spkernel_covers_et(spkernel, et) :
            spk.add(spkernel)
    
    spk.add(os.path.join(spkpath,'sat427.bsp'))


    spice.unload(lsk)
    spice.unload(sclk)


    # List of kernels to load
    kernels_to_load = [
        lsk,
        sclk,
        fk,
        ik,
        pck,
        *spk,
        *ck,
    ]

    if save :

        if savefile is None :
            if filter_yd is not None:
                savefile = f_year+'_'+f_doy+'.tm'
            else:
                savefile=str(int(et))+'.tm'
        else :
            if '.' not in savefile : savefile+='.tm'

        
        # Créer le contenu du métakernel
        metakernel_content = '\\begindata\nKERNELS_TO_LOAD = (\n'
        for kernel in kernels_to_load:
            metakernel_content += f"    '{kernel}',\n"
        metakernel_content += ')\n\\begintext'

        metakernel_path = os.path.join(mk_path,savefile)
        with open(metakernel_path, 'w') as mk:
            mk.write(metakernel_content)
    
    return kernels_to_load
