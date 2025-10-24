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
# User guide page 21
pixel_bandpasses = {'EUV':0.6049,
                    'FUV':0.7794}


# Slit width image for PSF in Angstroms
slit_dlambda = {'EUV':{'OCCLTATION':19.4,   'LOW_RESOLUTION':4.8, 'HIGH_RESOLUTION':2.75}, 
                'FUV':{'OCCLTATION':24.9,   'LOW_RESOLUTION':4.8, 'HIGH_RESOLUTION':2.75}}


# Slit width in microns
slit_width = {'EUV':{'OCCLTATION':800,   'LOW_RESOLUTION':200, 'HIGH_RESOLUTION':100}, 
              'FUV':{'OCCLTATION':800,   'LOW_RESOLUTION':150, 'HIGH_RESOLUTION':75}} 


# Approximate spacecraft time of the starburn even (June 6, 2002)
sctimeburn = 1402021717