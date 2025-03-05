# UVIS RELATED CONFIGURATION VARIABLES
# -------------------------------------------------------------------


# -= Slit width ratio =-
#   This is the ratio of the low-res slit width divided by that of the chosen slit.
#   This fractional slit area will be used to scale the low-res sensitivity.

# Slit widths FUV:
#      hi res =  0.075 mm
#      low res = 0.150 mm
#      occ =     0.800 mm
# Slit widths EUV:
#       hi res =  0.100 mm
#       low res = 0.200 mm
#       occ =     0.800 mm

slit_ratios = {'EUV':{'OCCLTATION':0.25,   'LOW_RESOLUTION':1, 'HIGH_RESOLUTION':2},
               'FUV':{'OCCLTATION':0.1875, 'LOW_RESOLUTION':1, 'HIGH_RESOLUTION':2}}

# Define the average pixel bandpass in angstroms (dispersion x pixel width)
pixel_bandpasses = {'EUV':0.605,
                    'FUV':0.78}


# Approximate spacecraft time of the starburn even (June 6, 2002)
sctimeburn = 1402021717