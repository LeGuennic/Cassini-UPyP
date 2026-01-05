# Configuration file used in plotting routine in UVIS_Geometry module
# -------------------------------------------------------------------

import numpy



# VISIBLE OBJECTS IN FIELD OF VIEW
#---------------------------------
FOV_objects = ['SATURN', 'EARTH', 'SUN', 'TITAN', 'RHEA', 'IAPETUS', 'DIONE',
               'TETHYS', 'ENCELADUS', 'MIMAS']

OFFSET = numpy.array((0,75,0))

# GRID TO PLOT LONGITUDE AND LATITUDE LINES ON TARGET
#----------------------------------------------------
lon_line_grid = [0,60,120,180,240,300]
lat_line_grid = [-60,-30,0,30,60]




# PLOTS CONFIGURATION
#--------------------

# Latitude and longitude grid
LATLON_GRID = {
    "color"      : "black",
    "linewidth"  : 0.5,
    "marker"     : ".",
    "markersize" : 0.5,
    "ls"         : '',
    "zorder"     : 3
}

# Background and stars
BACKGROUND_COLOR = "black"

RADEC_LINES = {
    "color": "grey",
    "marker": "+",
    "markersize": 2,
    "ls":'',
    "zorder":-100001
}

STAR_STYLE = {
    "color": "white",
    "marker": ".",
    "linestyle": "",
    "markersize": 1,
    "zorder":-100000
}

UV_STAR_STYLE = {
    "color": "purple",
    "marker": ".",
    "linestyle": "",
    "markersize": 1,
    "zorder":-99999
}

# Planets
PLANET_STYLE = {
    "SATURN": {
        "limb": {"color": "#EBD490", "linewidth": 1, "linestyle": "-"},
        "day_side": {"color": "#EBD490"},
        "night_side": {"color": "gray"},
        "terminator": {"color": "black", "linestyle": "--", "linewidth": 1}
    },
    "EARTH": {
        "limb": {"color": "#254C7E", "linewidth": 1, "linestyle": "-"},
        "day_side": {"color": "#254C7E"},
        "night_side": {"color": "gray"},
        "terminator": {"color": "black", "linestyle": "--", "linewidth": 1}
    },
    "SUN": {
        "limb": {"color": "yellow", "linewidth": 1, "linestyle": "-"},
        "day_side": {"color": "white"},
        "night_side": {"color": "gray"},
        "terminator": {"color": "black", "linestyle": "--", "linewidth": 1}
    },
    "TITAN": {
        "limb": {"color": "#C7966A", "linewidth": 1, "linestyle": "-"},
        "day_side": {"color": "#C7966A"},
        "night_side": {"color": "gray"},
        "terminator": {"color": "black", "linestyle": "--", "linewidth": 1}
    },


    "DEFAULT": {
        "limb": {"color": "white", "linewidth": 1, "linestyle": "-"},
        "day_side": {"color": "white"},
        "night_side": {"color": "gray"},
        "terminator": {"color": "black", "linestyle": "--", "linewidth": 1},
    }
}


# Miscellaneous
TARGET_CENTER ={
    'ls':'',
    'marker':'+',
    'color':'red',
    'zorder':1
}