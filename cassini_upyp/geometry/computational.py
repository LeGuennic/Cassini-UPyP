from __future__ import annotations
from typing import Literal

import numpy as np


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
