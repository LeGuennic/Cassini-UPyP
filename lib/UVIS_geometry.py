from __future__ import annotations
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice


import warnings
from pathlib import Path
from scipy.constants import astronomical_unit

from lib.kernellib import *
from lib.UVIS_main import UVIS_Observation
import config.env as env
import config.plotting
from time import time


# STARS AND PIXELS -------------------
#region 

def stars_pickles() :
    """
    Loads the pickles' table 15 numpy file.

    Parameters
    ----------
    None

    Returns
    -------
    numpy.ndarray
        A structured NumPy array containing the star data.

    Raises
    ------
    FileNotFoundError
        If the file 'stars_pickles.npy' does not exist in the specified directory.
    """

    star_file = Path(env.stars_dir) / 'stars_pickles.npy'
    
    if not star_file.exists():
        raise FileNotFoundError(f"File not found: {star_file}")

    return np.load(star_file)


# Plot stars
def plot_stars(stars_pickles, ax=None):
    if ax is None : ax=plt.gca()

    ax.plot(stars_pickles()['tyRA'],stars_pickles()['tyDE'], **config.plotting.STAR_STYLE)


# Pixels FOV
def plot_pixels(pixels, ax=None, **kwargs) :
    # Array indices : center, b_l, b_r, u_r, u_l
    if ax is None : ax=plt.gca()


    for p in pixels :
        # Left side
        line = np.array((p[1,:], p[4,:])).T
        ax.plot(line[0], line[1], **kwargs)

        # Right side
        line = np.array((p[2,:], p[3,:])).T
        ax.plot(line[0], line[1], **kwargs)

        # Parallel lines
        line = np.array((p[1,:], p[2,:])).T
        ax.plot(line[0], line[1], **kwargs)

    # Last parallel line
    line = np.array((pixels[-1,-1,:], pixels[-1,-2,:])).T
    ax.plot(line[0], line[1], **kwargs)

#endregion



# UTILS ------------------------------
#region

def order_bool(bool_list):
    bool_list = np.array(bool_list)

    index = list(range(0,len(bool_list)))

    if np.all(bool_list) or np.all(~bool_list) :
        return np.array(index)
    
    if bool_list[0] and bool_list[-1] :
        bool_list = list(bool_list)
        while bool_list[0] :
            bool_list = bool_list[1:] + [bool_list[0]]
            index     = index[1:]     + [index[0]]
        return np.array(index)

    if not bool_list[0] and not bool_list[-1] :
        bool_list = list(bool_list)
        while not bool_list[-1] :
            bool_list = [bool_list[-1]] + bool_list[:-1] 
            index     = [index[-1]]     + index[:-1] 
        return np.array(index)


def rotation_matrix(ra_center_deg, dec_center_deg):
    ra_rad = np.deg2rad(ra_center_deg)
    dec_rad = np.deg2rad(dec_center_deg)
    
    # Rotation autour de l'axe z de -RA_center
    Rz = np.array([
        [np.cos(-ra_rad), -np.sin(-ra_rad), 0],
        [np.sin(-ra_rad),  np.cos(-ra_rad), 0],
        [0, 0, 1]
    ])
    
    # Rotation autour de l'axe y de (90° - DEC_center)
    Ry = np.array([
        [np.cos(dec_rad), 0, np.sin(dec_rad)],
        [0, 1, 0],
        [-np.sin(dec_rad), 0, np.cos(dec_rad)]
    ])
    
    # Matrice de rotation totale
    R = Ry @ Rz
    return R
#endregion



# GEOMETRY ---------------------------
#region

def is_in_frame(points,xrange, yrange, zrange=None):
    mask =  (points[:, 0] >= xrange[0]) & (points[:, 0] <= xrange[1]) & \
            (points[:, 1] >= yrange[0]) & (points[:, 1] <= yrange[1])
    return mask

def is_vector_in_quadrilateral(v, quad):
    """
    Determine whether one or several 3D vectors lies "inside" a quadrilateral 
    defined by four 3D vectors on the unit sphere.

    Parameters
    ----------
    v : array-like, shape (n, 3) or (3,)
        One or multiple 3D vectors to test. If (3,), it's a single vector.
        If (n, 3), it's an array of n vectors.
    quad : array-like, shape (4, 3)
        Four 3D vectors defining the quadrilateral. These vectors are 
        expected to be non-zero and form a convex quadrilateral on the 
        unit sphere once normalized.

    Returns
    -------
    result : bool or np.ndarray of bool, shape (n,)
        - If `v` is a single vector (3,), returns a single boolean.
        - If `v` is an array of vectors (n, 3), returns a boolean array of length n.

    Notes
    -----
    The test is performed on the unit sphere. All input vectors (both `v` and 
    `quad`) are normalized. Each edge of the quadrilateral is considered and 
    its corresponding normal vector is computed via a cross product of the two 
    adjacent vertices. For each vector to test, the dot product is computed
    with these normals.

    If a vector is consistently on the same "side" of all edges (i.e., 
    the signs of its dot products with the edge normals are coherent), 
    it is considered inside the quadrilateral.

    Edges for which the dot product is close to zero (within a tolerance) 
    do not contribute a strict sign constraint and are thus not 
    disqualifying, allowing for a vector lying exactly on an edge to be counted as inside.
    """

    v=np.asarray(v)
    flag=False
    if v.ndim ==1 : flag=True
    v    = np.atleast_2d(v).astype(float)
    quad = np.array(quad, dtype=float)
    if quad.shape != (4, 3):
        raise ValueError("The quadrilateral must be defined by exactly four 3D vectors (shape (4,3)).")

    n = v.shape[0]

    # Normalize input vectors v
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norms_v == 0):
        raise ValueError("Some input vectors are zero vectors, cannot normalize.")
    v_norm = v / norms_v

    # Normalize the quadrilateral vectors
    norms_q = np.linalg.norm(quad, axis=1, keepdims=True)
    if np.any(norms_q == 0):
        raise ValueError("The quadrilateral contains at least one zero vector, cannot normalize.")
    p = quad / norms_q

    # Compute the normals for each edge of the quadrilateral
    # For edge i, the normal is given by cross(p[i], p[i+1]).
    normals = np.array([np.cross(p[i], p[(i+1) % 4]) for i in range(4)])

    # Compute dot products of each tested vector with the four edge normals
    dot_products = np.dot(v_norm, normals.T)

    # Define a numerical tolerance
    eps = 1e-12

    # mask[i,j] = True if |dot_products[i,j]| > eps, meaning it's a significant test
    mask = (np.abs(dot_products) > eps)

    # Count how many edges provide a significant test per vector
    count_significant = mask.sum(axis=1)

    # Initialize result array
    results = np.zeros(n, dtype=bool)

    # Only process vectors that have at least one significant test
    valid_rows = (count_significant > 0)

    # For each valid vector, pick one significant edge as reference (the first found)
    ref_idx  = np.argmax(mask, axis=1)
    ref_sign = np.sign(dot_products[valid_rows, ref_idx[valid_rows]])

    # Extract the signs for all edges of these valid vectors
    sign_dp = np.sign(dot_products[valid_rows])

    # Check sign coherence: 
    # All significant edges should share the same sign as the reference edge.
    # Edges that are not significant (|dot| <= eps) are considered neutral and do not invalidate the test.
    eq_sign = (sign_dp == ref_sign[:, None])

    # Combine the sign test with the mask: 
    # Non-significant edges are treated as "OK" regardless of sign.
    masked_equals = eq_sign | (~mask[valid_rows, :])

    # A vector is inside if all edges (significant or not) are coherent with the reference sign
    results[valid_rows] = np.all(masked_equals, axis=1)

    if flag : return results[0]
    return results

def is_visible(points, body_position, radii, threshold=1.e-6, starmode=False):


    points =np.asarray(points)
    flag=False
    if points.ndim ==1 : flag=True
    points = np.atleast_2d(points)

    close_obj  = (points @ (-body_position) < 0)
    visibility = np.ones(points.shape[0], dtype=bool)

    # Calculer les interceptions vectorisées
    if np.any(close_obj) :
        
        close_dir = points[close_obj]

        intercepts, found = intersect(-body_position, close_dir, radii)
        intercepts = np.atleast_2d(intercepts)
        found      = np.atleast_1d(found)

        if starmode:
            # Si starmode est True, dès qu'il y a une intersection, le point est invisible
            visibility[close_obj] = ~found
            if flag : return visibility[0]
            return visibility

        # Si une intersection est trouvée, vérifier la visibilité
        if np.any(found):
            distance_to_intercept = np.linalg.norm(intercepts[found] + body_position, axis=1)
            distance_to_point = np.linalg.norm(close_dir[found], axis=1)

            # Le point est visible si la distance à l'interception est supérieure
            # ou égale à la distance au point (moins le seuil)
            vis_found = distance_to_intercept >= (distance_to_point - threshold)
            visibility_close = np.ones(found.shape[0], dtype=bool)
            visibility_close[found] = vis_found
            visibility[close_obj] = visibility_close


    if flag : return visibility[0]
    return visibility

def xyz2radec(vectors, return_r=False, units='radians', ra_range=None):
    """
    Convert Cartesian coordinates to spherical coordinates (RA, DEC).

    Parameters:
    - vectors: array-like, shape (..., 3)
        Cartesian coordinates to convert.
    - return_r: bool, default=False
        If True, include the radial distance in the output.
    - units: str, either 'degrees' or 'radians', default='radians'
        Units for the output angles.
    - ra_range: tuple of two floats, default=(0, 360)
        The desired range for Right Ascension (RA). For example:
        - (0, 360) for RA in [0, 360) degrees
        - (-180, 180) for RA in [-180, 180) degrees
        Similarly for radians, use (0, 2*np.pi) or (-np.pi, np.pi).

    Returns:
    - If return_r is False:
        Array of shape (..., 2) with RA and DEC.
    - If return_r is True:
        Array of shape (..., 3) with R, RA, and DEC.
    """
    
    if ra_range is None :
        if units == 'degrees' : ra_range=(0,360)
        if units == 'radians' : ra_range=(0,2*np.pi)
    vectors = np.asarray(vectors)
    flag=False
    if vectors.ndim == 1 : flag=True
    vectors = np.atleast_2d(vectors)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    # Compute the radial distance
    r = np.linalg.norm(vectors, axis=1)
    
    # Compute declination (DEC)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Clip z/r to avoid invalid values for arcsin
        lat = np.where(r != 0, np.arcsin(np.clip(z / r, -1.0, 1.0)), 0.0)
        if units == 'degrees' : lat = np.degrees(lat)

    # Compute right ascension (RA) using arctan2 to get the correct quadrant
    lon = np.arctan2(y, x)
    if units == 'degrees':
        lon = np.degrees(lon)
    # If units are radians, no conversion needed

    # Adjust RA based on the specified ra_range
    min_ra, max_ra = ra_range
    width = max_ra - min_ra

    # Wrap RA to be within [min_ra, max_ra)
    lon = (lon - min_ra) % width + min_ra

    # Optional: Handle edge cases where RA might exactly equal max_ra
    # by setting them to min_ra
    lon = np.where(lon == max_ra, min_ra, lon)

    # Prepare the output
    if return_r:
        sph_coords = np.column_stack((r, lon, lat))
        if flag : return sph_coords[0]
        return sph_coords
    else:
        sph_coords = np.column_stack((lon, lat))
        if flag : return sph_coords[0]
        return sph_coords
    

def radec2xyz(coords, units:Literal['radians', 'degrees']='radians'):
    """
    Convertit des coordonnées RA/DEC en coordonnées cartésiennes (x, y, z).

    Parameters:
        coords : array_like
            Tableau des coordonnées RA et DEC. Forme (N, 2) ou (2,) pour un ou plusieurs vecteurs.
        return_r : bool, optional
            Si True, inclut le rayon 'r' dans la sortie. Par défaut, False.
        units : str, optional
            Unités des angles d'entrée. Peut être 'radians' ou 'degrees'. Par défaut, 'radians'.

    Returns:
        ndarray
            Coordonnées cartésiennes. Forme (N, 3) ou (N, 4) si return_r est True.
            Pour un vecteur unique, retourne une forme (3,) ou (4,).
    """
    # Assure que les coordonnées sont au moins en 2D
    coords=np.asarray(coords)
    flag=False
    if coords.ndim == 1 : flag=True
    coords = np.atleast_2d(coords)
    ra, dec = coords[:, 0], coords[:, 1]

    # Conversion des unités si nécessaire
    if units == 'degrees':
        ra = np.radians(ra)
        dec = np.radians(dec)
    elif units != 'radians':
        raise ValueError("L'unité doit être 'degrees' ou 'radians'.")

    # Supposition d'une sphère unité si le rayon n'est pas fourni
    r = np.ones_like(ra)

    # Calcul des coordonnées cartésiennes
    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    z = r * np.sin(dec)


    xyz = np.column_stack((x, y, z))

    # Retourne un vecteur 1D si l'entrée était un vecteur unique
    if xyz.shape[0] == 1:
        if flag : return xyz[0]
        return xyz
    return xyz

def intersect(observer, directions, radii, closest_point=False):
    a, b, c = radii
    px, py, pz = observer

    directions=np.asarray(directions)
    flag=False
    if directions.ndim == 1 : flag=True
    directions = np.atleast_2d(directions)

    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    inv_a2, inv_b2, inv_c2 = 1.0 / a**2, 1.0 / b**2, 1.0 / c**2

    # Coefficients du polynôme quadratique
    A = dx**2 * inv_a2 + dy**2 * inv_b2 + dz**2 * inv_c2
    B = 2.0 * (px * dx * inv_a2 + py * dy * inv_b2 + pz * dz * inv_c2)
    C = px**2 * inv_a2 + py**2 * inv_b2 + pz**2 * inv_c2 - 1.0

    delta = B**2 - 4 * A * C
    found = (delta >= 0)

    if closest_point:
        # t correspondant au point le plus proche du centre de l'ellipsoïde
        t = -B / (2 * A)
        intercepts = observer + directions * t[:, None]
        if flag : return intercepts[0], found[0]
        return intercepts, found

    # Préparer le tableau des intersections
    intercepts = np.full((directions.shape[0], 3), np.nan, dtype=np.float64)

    if not np.any(found):
        if flag : return intercepts[0], found[0]
        return intercepts, found

    # Calcul des racines pour les rays où l'intersection existe
    sqrt_delta = np.sqrt(delta[found])
    twoA = 2.0 * A[found]
    t1 = (-B[found] - sqrt_delta) / twoA
    t2 = (-B[found] + sqrt_delta) / twoA

    # On choisit le t minimum (correspondant au premier point rencontré sur le ray)
    t_min = np.minimum(t1, t2)[:, None]

    # Calcul des points d'intersection
    intercepts[found] = observer + directions[found] * t_min

    if flag : return intercepts[0], found[0]
    return intercepts, found

def ellipsoid_coords(radii, lon, lat):
    """
    Calculate Cartesian coordinates (x, y, z) on a triaxial ellipsoid
    for given longitude and latitude values. The function supports both
    scalar and array-like inputs for longitude and latitude.

    Parameters
    ----------
    radii : array-like of float
        The three radii of the ellipsoid in the order (a, b, c), where:
        - a : Equatorial radius along the x-axis (in kilometers).
        - b : Equatorial radius along the y-axis (in kilometers).
        - c : Polar radius along the z-axis (in kilometers).
    lon : float or array-like
        Longitude(s) in radians.
    lat : float or array-like
        Latitude(s) in radians.

    Returns
    -------
    coords : ndarray
        Cartesian coordinates on the ellipsoid. If the inputs are scalars,
        returns a 1D array of shape (3,). If the inputs are array-like,
        returns a 2D array of shape (N, 3), where N is the number of points.

    Raises
    ------
    ValueError
        If `radii` does not contain exactly three elements.

    Notes
    -----
    - Supports broadcasting of input arrays for `lon` and `lat`.
    - Ensures that if the inputs are scalars, the output is a single coordinate vector.

    Examples
    --------
    >>> import numpy as np
    >>> # Single coordinate
    >>> radii = [6378.137, 6356.752, 6356.752]  # in kilometers
    >>> lon = 0.5  # radians
    >>> lat = 0.3  # radians
    >>> coords = ellipsoid_coords(radii, lon, lat)
    >>> print(coords)
    [6127.27132226 1954.80084321 1887.75285779]

    >>> # Array of coordinates
    >>> radii = [6378.137, 6356.752, 6356.752]  # in kilometers
    >>> lon = np.array([0.1, 0.2, 0.3])  # radians
    >>> lat = np.array([0.4, 0.5, 0.6])  # radians
    >>> coords = ellipsoid_coords(radii, lon, lat)
    >>> print(coords)
    [[ 6122.4502675   624.47524032 2523.7580287 ]
     [ 6075.33722722 1227.03339716 3046.70976819]
     [ 5964.75458913 1857.55541576 3684.20191952]]
    """

    # Ensure radii is an array and has exactly three elements
    radii = np.asarray(radii, dtype=float)
    if radii.shape[-1] != 3:
        raise ValueError("`radii` must contain exactly three elements: (a, b, c).")
    a, b, c = radii

    # Convert longitude and latitude to at least 1D arrays for vectorized operations
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)

    # Attempt to broadcast lon and lat to compatible shapes
    try:
        lon, lat = np.broadcast_arrays(lon, lat)
    except ValueError as e:
        raise ValueError("`lon` and `lat` could not be broadcast to a common shape.") from e


    # Compute trigonometric functions
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    # Calculate Cartesian coordinates
    x = a * cos_lat * cos_lon
    y = b * cos_lat * sin_lon
    z = c * sin_lat

    # Stack the coordinates into a single array of vectors
    coords = np.column_stack((x, y, z))

    # If inputs were scalars, return a 1D array of shape (3,)
    if coords.shape[0] == 1:
        return coords[0]
    
    return coords

def ellipsoid_xyz(radii, vec, return_altitude=True, units:Literal['radians', 'degrees']='radians'):
    """
    Computes longitude, latitude, and altitude for points relative to an ellipsoid.

    Parameters
    ----------
    radii : tuple or array-like
        Radii of the ellipsoid along the x, y, and z axes (a, b, c).
    vec   : array-like
        Coordinates of the point(s) as (x, y, z). Can be a single point or an array of points.
    return_altitude : bool, optional
        If True, returns the altitude above the ellipsoid surface; if False, returns the distance from the origin.
    units : str
        Units 'degrees', or 'radians' for longitude and latitude to be returned. Default is radians.

    Returns
    -------
    longitude : float or ndarray
        Longitude in radians, between -π and π.
    latitude : float or ndarray
        Latitude in radians, between -π/2 and π/2.
    altitude_or_distance : float or ndarray
        Altitude above the ellipsoid surface if return_altitude is True; otherwise, the distance from the origin.
    """

    a, b, c = radii

    # Validate radii
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("All radii must be positive numbers.")

    vec = np.atleast_2d(vec)
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]

    # Normalized coordinates
    x_norm = x / a
    y_norm = y / b
    z_norm = z / c

    # Compute s with a small epsilon to avoid division by zero
    s = np.sqrt(x_norm**2 + y_norm**2 + z_norm**2)

    norm_vec = np.linalg.norm(vec, axis=1)

    # Compute longitude and latitude
    longitude = np.arctan2(y_norm, x_norm)
    longitude = np.mod(longitude, 2*np.pi)
    latitude  = np.arcsin(np.clip(z_norm / s, -1.0, 1.0))  # Ensure the value is within [-1, 1]

    if units=="degrees" :
        longitude = np.mod(np.degrees(longitude), 360)
        latitude  = np.degrees(latitude)

    if return_altitude:
        # Compute altitude efficiently
        altitude = norm_vec * (1 - 1 / s)
        if vec.shape[0]==1 :
            return longitude.squeeze(), latitude.squeeze(), altitude.squeeze()
        else : return longitude, latitude, altitude
    else:
        if vec.shape[0]==1 :
            return longitude.squeeze(), latitude.squeeze(), norm_vec.squeeze()
        else : return longitude, latitude, norm_vec

def vec_angle(v1, v2, units:Literal['radians', 'degrees']='degrees'):
    """
    Calculates the angle(s) between vectors.

    If one of the inputs is a single vector and the other is an array of vectors,
    the function calculates the angle between the single vector and each vector in the array.

    Parameters
    ----------
    v1 : array_like
        First  vector or array of vectors.
    v2 : array_like
        Second vector or array of vectors.
    units : str, optional
        'degrees' (default) or 'radians' to specify the unit of the returned angle.

    Returns
    -------
    angles : float or ndarray
        The angle(s) between `v1` and `v2` in the specified units.

    Raises
    ------
    ValueError
        If the vector dimensions do not match or if the shapes are incompatible for broadcasting.

    Examples
    --------
    >>> vec_angle([1, 0, 0], [0, 1, 0])
    90.0
    >>> vec_angle([1, 0, 0], [[0, 1, 0], [1, 0, 0]])
    array([90.,  0.])
    >>> vec_angle([[1, 0], [0, 1]], [1, 1])
    array([45., 45.])
    """

    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)

    if v1.shape[-1] != v2.shape[-1]:
        raise ValueError("Vectors must have the same dimensions.")

    # Attempt to broadcast v1 and v2 to compatible shapes
    try:
        v1, v2 = np.broadcast_arrays(v1, v2)
    except ValueError:
        raise ValueError(
            "v1 and v2 must be broadcastable to the same shape or one of them must be a single vector."
        )

    # Compute the dot product and norms
    dot_product = np.einsum('...i,...i->...', v1, v2)
    norm_v1 = np.linalg.norm(v1, axis=-1)
    norm_v2 = np.linalg.norm(v2, axis=-1)

    # Compute the cosine of the angle
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    # Convert to degrees if specified
    if units == 'degrees':
        return np.degrees(angle_rad).squeeze()
    elif units == 'radians':
        return angle_rad.squeeze()
    else:
        raise ValueError("units must be 'degrees' or 'radians'")

def max_angular_diameter(points):
    """
    Calculates the maximal angular diameter (maximum angle) among a list of vectors.

    This function computes the largest angle between any two vectors in the provided list.
    It effectively determines the angular diameter of the geometric shape formed by the vectors.

    Parameters
    ----------
    points (array-like): An array of vectors, where each vector is a list or array of coordinates.

    Returns
    -------
    float: The maximal angle in degrees.
    
    Raises
    ------
    ValueError: If `points` contains zero vectors, which cannot be normalized.
    """

    
    # Compute the norms of each vector
    norms = np.linalg.norm(points, axis=1)
    
    # Check for zero vectors to avoid division by zero
    if np.any(norms == 0):
        raise ValueError("Input contains zero vectors, which cannot be normalized.")
    
    # Normalize the vectors
    vectors_norm = points / norms[:, np.newaxis]
    
    # Compute the dot product matrix between all pairs of normalized vectors
    dot_products = np.dot(vectors_norm, vectors_norm.T)
    
    # Clip the dot products to the valid range of arccos to avoid numerical issues
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Set the diagonal to 1 to exclude self-comparisons
    np.fill_diagonal(dot_products, 1.0)
    
    # Find the minimum dot product, which corresponds to the maximum angle
    min_dot = np.min(dot_products)
    
    # Calculate the maximum angle in degrees
    angle_max = np.degrees(np.arccos(min_dot))
    
    return angle_max

#endregion



class geometry:
    def __init__(self, ET:float,
                 meta_kernel = None, other_bodies=config.plotting.FOV_objects,
                 main=None, u:"UVIS_Observation"=None, target=None, offset=np.array((0,0,0))):

        self.ET   = ET
        self.main = main is None



        # SPICE kernels
        if self.main :
            # Automatically find necessary kernels for time ET
            if meta_kernel is None :
                meta_kernel=metakernel(ET, filter_yd=(u.YEAR, u.DOY))
                for kernel in meta_kernel:
                    spice.furnsh(kernel)

            else : spice.furnsh(meta_kernel)



        # Call geometry engine
        if target : self.target = target
        elif u is not None : self.target = u.target
        else : raise TypeError("Geometry object requires a valid target.")

        if self.main : self.geo_engine = geometer(self.target, 'CASSINI', self.ET, offset=offset)
        else : self.geo_engine = geometer(self.target, 'CASSINI', self.ET)
        




        
        # Basic geometry
        self.target_center = {
            'XYZ'   : self.geo_engine.planet_from_obs_j2k,
            'RADEC' : np.array((
                np.degrees(self.geo_engine.target_RA),
                np.degrees(self.geo_engine.target_DEC)
            ))
        }
        self.target_radius   = self.geo_engine.radii






        # TARGET GEOMETRY
        #-----------------
        # Target limb
        
        target_limb_xyz, self.dayside = self.geo_engine.get_ellipse()

        self.target_limb = {
            'XYZ'   : target_limb_xyz,
            'RADEC' : xyz2radec(target_limb_xyz, units='degrees')
        }





        if self.target!='SUN' :
            # Terminator line
            self.terminator = {
                'XYZ'   : self.geo_engine.get_terminator()
            }
            self.terminator['RADEC'] = xyz2radec(self.terminator['XYZ'], units='degrees')

            # Night side
            self.night_side = {}
            if np.any(~self.dayside) :
                # Concatenate night side limb points with terminator line points in the right order
                if   (abs(np.linalg.norm(self.terminator['XYZ'][ 0] - target_limb_xyz[~self.dayside][-1]))
                    < abs(np.linalg.norm(self.terminator['XYZ'][-1] - target_limb_xyz[~self.dayside][-1])))  :
                    self.night_side['XYZ'] = np.concatenate((
                        target_limb_xyz[~self.dayside],
                        self.terminator['XYZ']
                    ))
                else :
                    self.night_side['XYZ'] = np.concatenate((
                        target_limb_xyz[~self.dayside],
                        self.terminator['XYZ'][::-1]
                    ))
                self.night_side['RADEC'] = xyz2radec(self.night_side['XYZ'], units='degrees')

            else :
                self.night_side['RADEC'] = np.array([[np.nan, np.nan]])
                self.night_side['XYZ']   = np.array([[np.nan, np.nan, np.nan]])
            
            # Target distance to sun
            self.target_HD = np.linalg.norm(self.geo_engine.planet_from_obs_j2k -
                                 self.geo_engine.sun_from_obs_j2k)*1000   / astronomical_unit
            
        if not self.main :
            self.angular_diameter = max_angular_diameter(self.target_limb['XYZ'])
            self.zorder= self.geo_engine.zorder
        

        # Geometry for main target
        if self.main :

            # STARS
            #-----------------

            self.stars = stars_pickles()

            if u is not None :
                year, doy = u.YEAR, u.DOY
            else :
                year, doy, _ = spice.et2utc(ET, 'C', 0).split('-')
                year, doy = int(year), int(doy)

            delta_t = (year - 2000) + (doy / 365.0)

            # Proper motion (mas/year → degree/an → degree)
            correction_ra  = (self.stars['pmRA'] * 1e-3) / 3600.0 * delta_t
            correction_dec = (self.stars['pmDE'] * 1e-3) / 3600.0 * delta_t

            # Update corrected coordinates
            self.stars['RA_cor']  = self.stars['tyRA'] + correction_ra
            self.stars['DEC_cor'] = self.stars['tyDE'] + correction_dec

            ra_rad  = np.radians(self.stars['RA_cor'])
            dec_rad = np.radians(self.stars['DEC_cor'])
            xyz     = radec2xyz(np.array((ra_rad,dec_rad)).T)
            self.stars['XYZ'] = xyz
            #-------------------------------------------



            # Spacecraft position
            self.sub_sc_lon, self.sub_sc_lat, self.sc_altitude = ellipsoid_xyz(self.geo_engine.radii, self.geo_engine.obs_from_planet_brf)
            if self.target =='TITAN' : self.sub_sc_lon = 2*np.pi-self.sub_sc_lon

            self.sub_sc_lat = np.degrees(self.sub_sc_lat)
            self.sub_sc_lon = np.mod(np.degrees(self.sub_sc_lon), 360)


            # Sun position
            self.sub_solar_lon, self.sub_solar_lat,_ = ellipsoid_xyz(self.geo_engine.radii, self.geo_engine.planet_to_sun_brf)
            if self.target =='TITAN' : self.sub_solar_lon = 2*np.pi-self.sub_solar_lon

            self.sub_solar_lat = np.degrees(self.sub_solar_lat)
            self.sub_solar_lon = np.mod(np.degrees(self.sub_solar_lon), 360)

            # Heliocentric distance
            self.HD = np.linalg.norm(self.geo_engine.sun_from_obs_j2k)*1000   / astronomical_unit


            # Longitude and latitude lines
            self.lon_lines   , self.lat_lines   = [],[]
            for lon in config.plotting.lon_line_grid :
                lon_line = self.geo_engine.get_lon_line(np.radians(lon))

                self.lon_lines.append({
                    'XYZ'   : lon_line,
                    'RADEC' : xyz2radec(lon_line, units='degrees')
                })

            for lat in config.plotting.lat_line_grid :
                lat_line = self.geo_engine.get_lat_line(np.radians(lat))

                self.lat_lines.append({
                    'XYZ'   : lat_line,
                    'RADEC' : xyz2radec(lat_line, units='degrees')
                })






            # RA / DEC background lines
            ra_values = np.linspace(0, 360, 18*2+1)
            dec_ra    = np.arange(-90, 90, 0.1)

            # Créer une grille RA x DEC pour les lignes de RA
            ra_grid, dec_grid_ra = np.meshgrid(ra_values, dec_ra, indexing='ij')
            radec_ra = np.column_stack((ra_grid.ravel(), dec_grid_ra.ravel()))

            # Générer les lignes de DEC (181 points de -90 à 90 degrés)
            dec_values = np.linspace(-90, 90, 91)
            ra_dec = np.arange(0, 360, 0.1)
            # Créer une grille RA x DEC pour les lignes de DEC
            ra_grid_dec, dec_grid_dec = np.meshgrid(ra_dec, dec_values, indexing='ij')
            radec_dec = np.column_stack((ra_grid_dec.ravel(), dec_grid_dec.ravel()))

            # Combiner toutes les lignes RA et DEC
            radec_all = np.vstack((radec_ra, radec_dec))

            # Convertir toutes les coordonnées RA/DEC en XYZ en une seule opération
            xyz_all = radec2xyz(radec_all, units='degrees')

            # Construire le dictionnaire final
            self.radec_lines = {
                'RADEC' : radec_all,
                'XYZ'   : xyz_all
            }



            # OTHER OBJECTS IN FOV
            #----------------------
            self.other_targets=[]

            if other_bodies :
                other_bodies = [e.upper() for e in other_bodies]

                for target2 in other_bodies :
                    # Don't recompute main target
                    if target2 == self.target : continue

                    try :

                        self.other_targets.append(
                            geometry(self.ET, main=self, target=target2)
                        )
                    except spice.utils.exceptions.SpiceSPKINSUFFDATA :
                        warnings.warn(f"Insufficient ephemeris for {target2}, removing it.", RuntimeWarning)
    
            

            if u is not None :

                # UVIS FOV
                #----------
                # Array indices : center, b_l, b_r, u_r, u_l
                rotation_UVIS_J2k = spice.pxform(u.instrument.frame, "J2000", ET)
                pixel_vectors_reshaped = u.instrument.pixels_corners.reshape(-1, 3)
                pixel_vectors_j2000    = np.dot(rotation_UVIS_J2k, pixel_vectors_reshaped.T).T

                self.pixels = {
                    'XYZ'   : pixel_vectors_j2000.reshape(64, 5, 3),
                    'RADEC' : xyz2radec(pixel_vectors_j2000, units='degrees').reshape(64,5,2)
                }

                self.FOV_center = {
                    'XYZ'   : (self.pixels['XYZ'][32,1,:]+self.pixels['XYZ'][32,2,:])/2}
                self.FOV_center['RADEC'] = xyz2radec(self.FOV_center['XYZ'], units='degrees')
                

                # Binned pixels view
                xyz_trim = self.pixels['XYZ'][u.spat_start:u.spat_stop+1]
                first_pixels_indices = np.arange(u.n_pixels) * u.spat_bin
                last_pixels_indices  = first_pixels_indices + u.spat_bin - 1

                used_pixels_xyz = np.zeros((u.n_pixels, 5, 3))
                used_pixels_xyz[:, 1, :] =  xyz_trim[first_pixels_indices, 1, :]
                used_pixels_xyz[:, 2, :] =  xyz_trim[first_pixels_indices, 2, :]
                used_pixels_xyz[:, 3, :] =  xyz_trim[last_pixels_indices,  3, :]
                used_pixels_xyz[:, 4, :] =  xyz_trim[last_pixels_indices,  4, :]
                used_pixels_xyz[:, 0, :] = (xyz_trim[first_pixels_indices, 0, :] +
                                            xyz_trim[last_pixels_indices,  0, :]) / 2
                

                self.used_pixels = {
                    'XYZ'   : used_pixels_xyz,
                    'RADEC' : xyz2radec(used_pixels_xyz.reshape(-1,3), units='degrees').reshape(u.n_pixels, 5,2)
                }

                # FOV LOS                
                self.pixels_LOS = self.geo_engine.LOS_tangent(pixel_vectors_j2000)
                self.pixels_LOS = self.pixels_LOS.reshape(64, 5)

                self.n_used_pixels = u.n_pixels
                self.used_pixels_LOS = self.geo_engine.LOS_tangent(used_pixels_xyz.reshape(-1,3)).reshape(u.n_pixels,5)


                # STARS IN PIXEL
                #----------------
                angles = vec_angle(self.stars['XYZ'], self.FOV_center['XYZ'])
                mask   = angles < u.instrument.fov_height*2
                stars  = self.stars[mask]

                stars_xyz = stars['XYZ']


                # uv_stars  = np.stack(stars_UV['XYZ'].values)
                # angles = vec_angle(uv_stars, self.FOV_center['XYZ'])
                # mask   = angles < u.instrument.fov_height*2
                # uv_stars  = stars_UV[mask]

                # if not uv_stars.empty:
                #     uv_stars_xyz = np.stack(uv_stars['XYZ'].values)
                # else:
                #     uv_stars_xyz = np.empty((0, 3))  

                self.pixel_stars = [{} for _ in range(self.n_used_pixels)]

                for i_pixel in range(self.n_used_pixels):

                    on_disk = np.all(self.used_pixels_LOS[i_pixel,:]['alt']<0)
                    if on_disk :
                        n_star    = 0
                        final_mag = None
                        is_UV     = False

                    else :
                        pixel_corners     = self.used_pixels['XYZ'][i_pixel,1:,:]

                        is_in_pixel       = is_vector_in_quadrilateral(stars_xyz, pixel_corners)
                        is_star_visible   = self.geo_engine.is_visible(stars_xyz, starmode=True)
                        stars_in_pixel    = stars[is_in_pixel*is_star_visible]

                        # is_in_pixel_uv     = is_vector_in_quadrilateral(uv_stars_xyz, pixel_corners)
                        # is_star_visible_uv = self.geo_engine.is_visible(uv_stars_xyz, starmode=True)
                        # UV_stars_in_pixel  = uv_stars[is_in_pixel_uv*is_star_visible_uv]

                        n_star = (is_in_pixel*is_star_visible).sum() #+ (is_in_pixel_uv*is_star_visible_uv).sum()

                        if stars_in_pixel.size>0:
                            brightest_star = stars_in_pixel["fBt"].min()
                        else:
                            brightest_star = None

                        if brightest_star is None :
                            final_mag = None
                            is_UV = False
                        else:
                            final_mag = brightest_star
                            is_UV = False

                        # Handle empty pixels
                        # if not stars_in_pixel.empty:
                        #     brightest_star = stars_in_pixel["fBt"].max()
                        # else:
                        #     brightest_star = None

                        # if not UV_stars_in_pixel.empty:
                        #     brightest_uv_star = UV_stars_in_pixel["MAG"].max()
                        # else:
                        #     brightest_uv_star = None

                        # if brightest_star is None and brightest_uv_star is None:
                        #     final_mag = None
                        #     is_UV = False
                        # elif brightest_star is None:
                        #     final_mag = brightest_uv_star
                        #     is_UV = True
                        # elif brightest_uv_star is None:
                        #     final_mag = brightest_star
                        #     is_UV = False
                        # else:
                        #     if brightest_star >= brightest_uv_star:
                        #         final_mag = brightest_star
                        #         is_UV = False
                        #     else:
                        #         final_mag = brightest_uv_star
                        #         is_UV = True

                    self.pixel_stars[i_pixel]['number']  = n_star
                    self.pixel_stars[i_pixel]['MAG']     = final_mag
                    self.pixel_stars[i_pixel]['is_UV']   = is_UV
                    self.pixel_stars[i_pixel]['on_disk'] = on_disk




        # Apply rotation to Observer Reference Frame
        # TODO: MOVE OUTSIDE __INIT__.
        self.rotated    = False
        self.orf_center = None
        # self.only_J2K = True
        # if rotate :
        #     self.orf_center = orf_center
        #     self.only_J2K = False

        #     if orf_center is None : raise ValueError('ORF center for reference frame')
        #     if orf_center == 'target' :
                
        #         orf_center = (self.target_center['RADEC'][0] , self.target_center['RADEC'][1])
        #     elif orf_center == 'FOV'    :
        #         pixel_center = self.pixels['RADEC'][31]
        #         FOV_center   = (pixel_center[-2,:]+pixel_center[-1,:])/2
        #         orf_center = (FOV_center[0] , FOV_center[1])
        #     self.rotate(view_center=orf_center)
        


        # Clear SPICE kernels
        if self.main :
            spice.kclear()
        

    def rotate(self, view_center = None, units:Literal['radians', 'degrees']='degrees', ra_range=None) :
        if self.orf_center is not None:
            if self.orf_center == view_center: return

        
        if units=='degrees' : ra_range = (-180   , 180  )
        if units=='raidans' : ra_range = (-np.pi , np.pi)
        self.rotate_units = units

        # Build rotation matrix from J2000 to Observer Reference Frame
        if view_center is None :
            view_center=(self.target_center['RADEC'][0] , self.target_center['RADEC'][1])
        self.R = rotation_matrix(view_center[0], view_center[1])

        self.target_center['ORF']  = xyz2radec(self.target_center['XYZ'] @ self.R.T, units=units, ra_range=ra_range)

        self.target_limb['ORF']    = xyz2radec(self.target_limb['XYZ']   @ self.R.T, units=units, ra_range=ra_range)

        if self.target != 'SUN' :
            self.terminator['ORF'] = xyz2radec(self.terminator['XYZ']    @ self.R.T, units=units, ra_range=ra_range)
            self.night_side['ORF'] = xyz2radec(self.night_side['XYZ']    @ self.R.T, units=units, ra_range=ra_range)




        if self.main :
            self.stars_orf    = list(xyz2radec(self.stars['XYZ']   @ self.R.T, units=units, ra_range=ra_range))
            # self.UV_stars_orf = list(xyz2radec(np.array(stars_UV     ['XYZ'].tolist())    @ self.R.T, units=units, ra_range=ra_range))

            # Longitude and latitude lines
            for lon in self.lon_lines :
                lon['ORF'] = xyz2radec(lon['XYZ'] @ self.R.T, units=units, ra_range=ra_range)

            for lat in self.lat_lines :
                lat['ORF'] = xyz2radec(lat['XYZ'] @ self.R.T, units=units, ra_range=ra_range)



            self.pixels['ORF']      = self.pixels['XYZ'].reshape(-1,3)
            self.used_pixels['ORF'] = self.used_pixels['XYZ'].reshape(-1,3)

            self.pixels['ORF']      = xyz2radec(self.pixels['ORF']      @ self.R.T, units=units, ra_range=ra_range)
            self.used_pixels['ORF'] = xyz2radec(self.used_pixels['ORF'] @ self.R.T, units=units, ra_range=ra_range)

            self.pixels['ORF']      = self.pixels['ORF'].reshape(64, 5, 2)
            self.used_pixels['ORF'] = self.used_pixels['ORF'].reshape(self.n_used_pixels,5,2)

            self.radec_lines['ORF'] = xyz2radec(self.radec_lines['XYZ'] @ self.R.T, units=units, ra_range=ra_range)
        self.rotated    = True
        self.orf_center = view_center


    def plot(self, mode:Literal['target', 'FOV','allsky', 'manual']='target', orf_center=None,
             save=False, savename=None,
             RA_range=None, DEC_range=None,
             ax=None, pixel_notes=False) :
        
        frame = 'ORF' if mode in ['target', 'FOV', 'manual'] else 'RADEC'
        
        if ax is None :
            
            match mode:
                case 'target' :
                    orf_center = (self.target_center['RADEC'][0] , self.target_center['RADEC'][1])
                    self.rotate(view_center=orf_center)
                    
                    if RA_range is None and DEC_range is None  :
                        # Fix target disk in frame
                        RA_range  = min(self.target_limb['ORF'][:,0])*2, max(self.target_limb['ORF'][:,0])*2
                        DEC_range = min(self.target_limb['ORF'][:,1])*2, max(self.target_limb['ORF'][:,1])*2

                    else :
                        if self.rotate_units == 'degrees' :
                            RA_range, DEC_range = (-180,180), (-90,90)
                        if self.rotate_units == 'radians' :
                            RA_range, DEC_range = (-np.pi,np.pi), (-np.pi/2,np.pi/2)
                case 'FOV' :
                    pixel_center = self.pixels['RADEC'][31]
                    FOV_center   = (pixel_center[-2,:]+pixel_center[-1,:])/2
                    orf_center   = (FOV_center[0] , FOV_center[1])
                    self.rotate(view_center=orf_center)

                    if RA_range is None and DEC_range is None  :
                        # Fix Cassini FOV in frame
                        FOV_size  = abs(np.linalg.norm(self.pixels['ORF'][0,1,:] - self.pixels['ORF'][-1,-1,:]))
                        scale = 0.6
                        RA_range  = -FOV_size*scale, FOV_size*scale
                        DEC_range = -FOV_size*scale, FOV_size*scale
                    else :
                        if self.rotate_units == 'degrees' :
                            RA_range, DEC_range = (-180,180), (-90,90)
                        if self.rotate_units == 'radians' :
                            RA_range, DEC_range = (-np.pi,np.pi), (-np.pi/2,np.pi/2)
                
                case 'allsky' :
                    RA_range, DEC_range = (0,360), (-90,90)
                case 'manual' :
                    if orf_center is None : raise ValueError('Manual mode requires a (RA/DEC) central position')
                    if RA_range   is None : raise ValueError('Manual mode requires a valid RA range to plot')
                    if DEC_range  is None : raise ValueError('Manual mode requires a valid RA range to plot')
                    self.rotate(view_center=orf_center)

        
            fig, ax = plt.subplots()
            ax.set_facecolor(config.plotting.BACKGROUND_COLOR)
            ax.set_aspect('equal')

            xmin,xmax = RA_range
            ymin,ymax = DEC_range
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)

        if self.target in config.plotting.PLANET_STYLE :
            planet_style = config.plotting.PLANET_STYLE[self.target]
        else :
            planet_style = config.plotting.PLANET_STYLE['DEFAULT']

        # Target limb + day side
        ax.fill(
            self.target_limb[frame][:,0],
            self.target_limb[frame][:,1],
            **planet_style['limb'],
            zorder=0
        )

        if self.target != 'SUN' :
            # Night side
            ax.fill(
                self.night_side[frame][:,0],
                self.night_side[frame][:,1],
                **planet_style['night_side'],
                zorder = 1
            )

            # Terminator line
            ax.plot(
                self.terminator[frame][:,0],
                self.terminator[frame][:,1],
                **planet_style['terminator'],
                zorder = 2
            )
        
        if self.main :
            
            if frame=='ORF' :
                # RA / DEC background lines
                ax.plot(self.radec_lines[frame][:, 0], self.radec_lines[frame][:, 1],
                        **config.plotting.RADEC_LINES)


                # Background stars
                stars_orf = np.array(self.stars_orf)
                stars_orf = stars_orf[is_in_frame(stars_orf, RA_range, DEC_range)]
                ax.plot(stars_orf[:,0],stars_orf[:,1], **config.plotting.STAR_STYLE)

                # UV_stars_orf = np.array(self.UV_stars_orf)
                # UV_stars_orf = UV_stars_orf[is_in_frame(UV_stars_orf, RA_range, DEC_range)]
                # ax.plot(UV_stars_orf[:,0],UV_stars_orf[:,1], **config.plotting.UV_STAR_STYLE)

            else :
                ax.plot(self.stars['RA_cor'],self.stars['DEC_cor'], **config.plotting.STAR_STYLE)
                # plot_UV_stars(stars_UV,      ax=ax)

            # OTHER BODIES IN THE SKY
            for t2 in self.other_targets:

                if mode!='allsky' : t2.rotate(view_center=self.orf_center, units = self.rotate_units)
                
                if np.any(is_in_frame(t2.target_limb[frame], (xmin, xmax), (ymin, ymax))) :
                    plt.annotate( t2.target,
                         (t2.target_center[frame][0], t2.target_center[frame][1]),
                         color='white', textcoords="offset points", xytext=(5, 5), ha='center', fontsize=10, zorder=t2.zorder)
                    
                    if t2.angular_diameter < 2 :
                        if t2.target in config.plotting.PLANET_STYLE :
                            plt.plot([t2.target_center[frame][0]], [t2.target_center[frame][1]],
                                    ls='', marker='o', ms = 5, color=config.plotting.PLANET_STYLE[t2.target]['limb']['color'], zorder=t2.zorder)
                        else :
                            plt.plot([t2.target_center[frame][0]], [t2.target_center[frame][1]],
                                    ls='', marker='o', ms = 5, color=config.plotting.PLANET_STYLE['DEFAULT']['limb']['color'], zorder=t2.zorder)

                        continue
                    else : t2.plot(mode=mode, ax=ax)


            # LONGITUDE AND LATITUDE LINES
            for lon_line in self.lon_lines :
                ax.plot(
                    lon_line[frame][:,0],
                    lon_line[frame][:,1],
                    **config.plotting.LATLON_GRID,
                )


            for lat_line in self.lat_lines :
                ax.plot(
                    lat_line[frame][:,0],
                    lat_line[frame][:,1],
                    **config.plotting.LATLON_GRID,
                )

            # PIXELS
            # Total pixels
            plot_pixels(self.pixels[frame],
                        ax=ax, linewidth=1, color='lightgray', ls='-', marker='', zorder = 19)
            
            plot_pixels(self.used_pixels[frame],
                        ax=ax, linewidth=2, color='red', ls='-', marker='', zorder = 20)

            # TARGET CENTER
            plt.plot([self.target_center[frame][0]], [self.target_center[frame][1]], **config.plotting.TARGET_CENTER)

            # SUB-SPACECRAFT LATITUDE AND LONGITUDE
            plt.annotate(f"{round(self.sub_sc_lon)}° , {round(self.sub_sc_lat)}°",
                        (self.target_center[frame][0], self.target_center[frame][1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

            # PIXEL PARAMETERS (LOS PROPERTIES)
            if pixel_notes :
                for xi, yi, value in zip(self.used_pixels[frame][:,0,0], self.used_pixels[frame][:,0,1], self.used_pixels_LOS[:,0][pixel_notes]):
                    annotation = f"{value:.1f}"
                    ax.text(xi, yi, annotation, color='white', fontsize=8,
                            ha='center', va='center', 
                            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'), clip_on=True)



        
        if self.main :
            if save :
                plt.savefig(savename)
                plt.close(fig)
            else : plt.show()
        

    def plot3D(self, ax=None, target_center=(0,0,0)):
        if ax is None :
                    
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('equal')
            ax.set_facecolor('black')
            ax.set_axis_off()


        # Rayons de l'ellipsoïde
        a, b, c = self.target_radius

        # Générer les angles theta (longitude) et phi (latitude)
        theta      = np.linspace(0, 2 * np.pi, 100)  # Longitude
        phi        = np.linspace(0, np.pi, 50)         # Latitude
        theta, phi = np.meshgrid(theta, phi)

        # Coordonnées cartésiennes des points sur l'ellipsoïde
        x = a * np.sin(phi) * np.cos(theta) + target_center[0]
        y = b * np.sin(phi) * np.sin(theta) + target_center[1]
        z = c * np.cos(phi)                 + target_center[2]

        # Tracer la surface
        ax.plot_surface(x, y, z, alpha=1, color=planet_style['limb']['color'])
        if self.target!='SUN' :
            full_term = self.geo_engine.get_terminator(full=True) - self.geo_engine.planet_from_obs_j2k+ target_center
            ax.plot(full_term[:,0], full_term[:,1], full_term[:,2],
                  color='black')

        if self.main :
            obs_vect = -self.geo_engine.planet_from_obs_j2k/0.5
            ax.quiver(0, 0, 0, obs_vect[0], obs_vect[1], obs_vect[2],
                    color='b', arrow_length_ratio=0.1)
            
            sun_vect = (self.geo_engine.sun_from_obs_j2k-self.geo_engine.planet_from_obs_j2k)/2.e3
            ax.quiver(0, 0, 0, sun_vect[0], sun_vect[1], sun_vect[2],
                    color='gold', arrow_length_ratio=0.1)
            
            stars=stars_pickles[stars_pickles['fBt']<6]
            stars=np.array(stars['XYZ'].to_list())
            # uv_stars=np.array(stars_UV['XYZ'].to_list())
            ax.plot(stars[:,0],stars[:,1],stars[:,2], ls='', marker='.',color='white',ms=5)
            # ax.plot(uv_stars[:,0],uv_stars[:,1],uv_stars[:,2], ls='', marker='.',color='purple',ms=2)

            ax.plot(self.radec_lines['XYZ'][:, 0]*1.e10, self.radec_lines['XYZ'][:, 1]*1.e10,self.radec_lines['XYZ'][:, 2]*1.e10,
                           color='#3A3939', marker='.', ls='',ms=2)

            for t2 in self.other_targets:

                
                # plt.annotate( t2.target,
                #         (t2.target_center[frame][0], t2.target_center[frame][1]),
                #         color='white', textcoords="offset points", xytext=(5, 5), ha='center', fontsize=10, zorder=t2.zorder)
                # if t2.target!='SUN' :
                t2.plot3D(ax=ax,
                            target_center= t2.geo_engine.planet_from_obs_j2k - self.geo_engine.planet_from_obs_j2k)

        ax.set_xlim([-1.e5, 1.e5])
        ax.set_ylim([-1.e5, 1.e5])
        ax.set_zlim([-1.e5, 1.e5])
        ax.set_aspect('equal', adjustable='datalim')
        plt.show()



class geometer:
    def __init__(self, target, observer, et, offset=np.array((0,0,0))):
        self.planet   = target.upper()
        self.observer = observer.upper()
        self.et = et

        self.radii = spice.bodvrd(self.planet, 'RADII', 3)[1]


        # Position vectors (Body Reference Frame of J2000)
        self.obs_from_planet_brf,_   = spice.spkpos(
            self.observer, self.et, 'IAU_'+self.planet, 'XLT+S', self.planet
        )
        self.obs_from_planet_brf  += offset
        
        self.planet_from_obs_j2k,_   = spice.spkpos(
            self.planet, self.et, 'J2000', 'LT+S', self.observer
        )
        dummy = np.copy(self.planet_from_obs_j2k)
        # self.planet_from_obs_j2k+=np.array((0,75,0))

        rad1 = xyz2radec(dummy)
        rad2 = xyz2radec(self.planet_from_obs_j2k)

        self.sun_from_obs_j2k,_      = spice.spkpos(
            'SUN', self.et, 'J2000', 'LT+S', self.observer
        )

        

        


        self.target_RA, self.target_DEC = xyz2radec(self.planet_from_obs_j2k, ra_range=(0,2*np.pi))
        self.target_distance = np.linalg.norm(self.planet_from_obs_j2k)

        # Zorder for plotting
        self.zorder = -round(self.target_distance/1000)

        self.rotation_IAU_J2K = spice.pxform('IAU_' + self.planet, 'J2000', self.et)
        self.rotation_J2K_IAU = spice.pxform('J2000', 'IAU_' + self.planet, self.et)


        # Vecteur de la planète vers le Soleil depuis l'observateur
        planet_to_sun_j2000 = self.sun_from_obs_j2k - self.planet_from_obs_j2k
        self.planet_to_sun_brf = np.dot(self.rotation_J2K_IAU, planet_to_sun_j2000)


    def get_ellipse(self, npoints=100, altitude:float=0, flag_dayside=True):

        angles = np.linspace(0, 2 * np.pi, npoints)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Get limb ellipse vectors
        limb = spice.edlimb(self.radii[0]+altitude, self.radii[1]+altitude, self.radii[2]+altitude,
                            self.obs_from_planet_brf)
        center, semi_major, semi_minor = spice.el2cgv(limb)

        # Compute points
        limb_point_brf   = center + cos_angles[:, np.newaxis] * semi_major + sin_angles[:, np.newaxis] * semi_minor
        observer_to_limb = limb_point_brf - self.obs_from_planet_brf  # Shape: (npoints, 3)
        
        if flag_dayside :
            # Flag points in day side

            # Vecteur de la planète vers le Soleil depuis l'observateur
            planet_to_sun_j2000 = self.sun_from_obs_j2k - self.planet_from_obs_j2k
            planet_to_sun_brf   = np.dot(self.rotation_J2K_IAU, planet_to_sun_j2000)

            sun_to_limb =  limb_point_brf - planet_to_sun_brf

            dayside = is_visible(sun_to_limb, -planet_to_sun_brf, self.radii, threshold=100)

            # Offset array to put every visible point together then every invisible point
            order_indices = order_bool(dayside)
            observer_to_limb = observer_to_limb[order_indices]
            dayside          = dayside[order_indices]

        else : dayside = None

        limb_points_rec = observer_to_limb @ self.rotation_IAU_J2K.T  # Shape: (npoints, 3)

        return limb_points_rec, dayside

    def is_visible(self, points, starmode = False, threshold=1.e-6):
        return is_visible(points, -self.obs_from_planet_brf, self.radii, threshold=threshold)

    def get_terminator(self, npoints=100, full=False) :
        angles = np.linspace(0, 2 * np.pi, npoints)
        cos_angles = np.cos(angles)  # Shape: (npoints,)
        sin_angles = np.sin(angles)  # Shape: (npoints,)


        limb = spice.edlimb(self.radii[0], self.radii[1], self.radii[2],
                            self.planet_to_sun_brf)
        center, semi_major, semi_minor = spice.el2cgv(limb)

        limb_point_brf = center + cos_angles[:, np.newaxis] * semi_major + sin_angles[:, np.newaxis] * semi_minor
        observer_to_limb = limb_point_brf - self.obs_from_planet_brf  # Shape: (npoints, 3)
        if full : is_point_visible = np.ones(len(observer_to_limb), dtype=bool)
        else : is_point_visible = self.is_visible(observer_to_limb)  # Shape: (npoints,)

        limb_point_j2000 = observer_to_limb @ self.rotation_IAU_J2K.T  # Shape: (npoints, 3)


        # Offset array to put every visible point together then every invisible point
        order_indices = order_bool(is_point_visible)
        limb_point_j2000=limb_point_j2000[order_indices]
        is_point_visible=is_point_visible[order_indices]

        
        terminator = limb_point_j2000[is_point_visible]  # Shape: (npoints, 3)


        return terminator
    
    def get_lon_line(self, lon, latgrid = None) :
        if latgrid is None:
            latgrid = np.linspace(-np.pi, np.pi, 37)
        

        vec    = ellipsoid_coords(self.radii, lon, latgrid)
        
        points = vec - self.obs_from_planet_brf
        points = points[self.is_visible(points)]

        
        points = points @ self.rotation_IAU_J2K.T

        return points
    
    def get_lat_line(self, lat, longrid = None) :
        if longrid is None:
            longrid = np.linspace(0, 2*np.pi, 37)
        
        vec    = ellipsoid_coords(self.radii, longrid, lat)
        
        points = vec - self.obs_from_planet_brf
        points = points[self.is_visible(points)]

        
        points = points @ self.rotation_IAU_J2K.T

        return points

    def LOS_tangent(self, LOS, J2000=True) :
        if J2000 :
            LOS = LOS @ self.rotation_J2K_IAU.T  # Shape: (npoints, 3)
        
        planet_to_sun_brf,_ = spice.spkpos('SUN', self.et, 'IAU_' + self.planet, 'LT+S', self.planet)

        self.sub_solar_longitude, _, _ = ellipsoid_xyz(self.radii, planet_to_sun_brf, units='degrees')
        if self.planet=='TITAN':
            self.sub_solar_longitude = 360 - self.sub_solar_longitude
        

        tangent_point  , found  = intersect(self.obs_from_planet_brf, LOS, self.radii, closest_point=True)
        # intersect_point, found  = intersect(self.obs_from_planet_brf, LOS, self.radii, closest_point=False)
        # tangent_point[found] = intersect_point[found]


        tangent_point_to_sun = planet_to_sun_brf - tangent_point

        lons, lats, alts = ellipsoid_xyz(self.radii, tangent_point, units='degrees')
        if self.planet=='TITAN': lons = 360-lons
        sza   = vec_angle(tangent_point, tangent_point_to_sun)
        phase = vec_angle(LOS,           tangent_point_to_sun)
        ems   = vec_angle(LOS,           tangent_point)

        lst = 12.0 - (lons - self.sub_solar_longitude) * (24.0 / 360.0)
        # On peut éventuellement ramener la LST dans l'intervalle 0-24 :
        lst = lst % 24.0


        keys  = ['lon', 'lat', 'alt', 'sza', 'phase', 'ems', "lt"]
        param = [ lons,  lats,  alts,  sza,   phase,   ems,   lst]
        dtype = [(k,float) for k in keys]
        
        params = np.zeros(LOS.shape[0], dtype=dtype)
        for k,p in zip(keys,param) :
            params[k] = p

        return params
