import numpy as np

PicsPerExposure = 60


default_lat_bins = np.arange(10)*20-90
# [-90, -70, -50, -30, -10, 10,  30,  50,  70,  90]


default_alt_bins = [0]+[k*100-50 for k in range(1,17)]+[np.inf]
# [0, 50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, inf]

# Smoothing kernel
smoothing_kernel = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0