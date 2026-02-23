from __future__ import annotations
from typing import Literal
from numpy.typing import ArrayLike

import numpy as np



def xyz2radec(vectors:ArrayLike, return_r:bool = False, units:Literal['radians', 'degrees'] = 'radians', ra_range:tuple[float, float]=None) -> ArrayLike:
    """
    Convert Cartesian vectors to (RA, Dec) spherical coordinates.

    Parameters
    ----------
    vectors : array-like, shape (..., 3)
        Cartesian vectors to convert.
    return_r : bool, optional
        If True, also return the radial distance ``r = ||vector||``. Default is False.
    units : {'radians', 'degrees'}, optional
        Units of the returned angles. Default is 'radians'.
    ra_range : (float, float) or None, optional
        Desired wrapping interval for right ascension (RA). If None, defaults to
        (0, 2*pi) for radians or (0, 360) for degrees. Values are wrapped into
        ``[min_ra, max_ra)``.

    Returns
    -------
    numpy.ndarray
        If ``return_r`` is False: array of shape (..., 2) with columns (RA, Dec).
        If ``return_r`` is True:  array of shape (..., 3) with columns (r, RA, Dec).
        If the input is a single vector of shape (3,), a 1D array is returned.

    Notes
    -----
    - Dec is computed as ``asin(z / r)`` (with safe handling for r=0).
    - RA is computed with ``atan2(y, x)`` and wrapped to ``ra_range``.
    """
    
    # Set default RA range if not provided
    if ra_range is None :
        if units == 'degrees' : ra_range=(0,360)
        if units == 'radians' : ra_range=(0,2*np.pi)
    
    # Prepare inputs
    vectors = np.asarray(vectors)
    flag=False
    if vectors.ndim == 1 : flag=True
    vectors = np.atleast_2d(vectors)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    # Radial distance
    r = np.linalg.norm(vectors, axis=1)
    
    # Declination (DEC)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Clip z/r to avoid invalid values for arcsin
        lat = np.where(r != 0, np.arcsin(np.clip(z / r, -1.0, 1.0)), 0.0)
        if units == 'degrees' : lat = np.degrees(lat)

    # Right ascension (RA) using arctan2 to get the correct quadrant
    lon = np.arctan2(y, x)
    if units == 'degrees':
        lon = np.degrees(lon)

    # Adjust RA based on the specified ra_range
    min_ra, max_ra = ra_range
    width = max_ra - min_ra

    # Wrap RA to be within [min_ra, max_ra)
    lon = (lon - min_ra) % width + min_ra

    # Handle edge cases where RA might exactly equal max_ra
    # by setting them to min_ra
    lon = np.where(lon == max_ra, min_ra, lon)

    if return_r:
        sph_coords = np.column_stack((r, lon, lat))
        if flag : return sph_coords[0]
        return sph_coords
    else:
        sph_coords = np.column_stack((lon, lat))
        if flag : return sph_coords[0]
        return sph_coords
    
def radec2xyz(coords:ArrayLike, units:Literal['radians', 'degrees']='radians') -> ArrayLike:
    """
    Convert (RA, Dec) sky coordinates to Cartesian unit vectors.

    Parameters
    ----------
    coords : array-like, shape (..., 2)
        Right ascension and declination pairs (RA, Dec).
    units : {'radians', 'degrees'}, optional
        Units of the input angles. Default is 'radians'.

    Returns
    -------
    numpy.ndarray
        Cartesian unit vectors of shape (..., 3). If the input is a single pair
        (shape (2,)), a 1D array of shape (3,) is returned.

    Raises
    ------
    ValueError
        If ``units`` is not 'radians' or 'degrees'.
    """

    # Inputs
    coords=np.asarray(coords)
    flag=False
    if coords.ndim == 1 : flag=True
    coords = np.atleast_2d(coords)
    ra, dec = coords[:, 0], coords[:, 1]

    # Convert units
    if units == 'degrees':
        ra = np.radians(ra)
        dec = np.radians(dec)
    elif units != 'radians':
        raise ValueError("units must be 'degrees' or 'radians'.")

    # Assume a unit sphere if radius is not provided
    r = np.ones_like(ra)

    # Cartesian coordinates
    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    z = r * np.sin(dec)

    xyz = np.column_stack((x, y, z))

    if xyz.shape[0] == 1:
        if flag : return xyz[0]
        return xyz
    return xyz

def vec_angle(v1:ArrayLike, v2:ArrayLike, units:Literal['radians', 'degrees'] = 'degrees') -> ArrayLike:
    """
    Calculates the angle(s) between vectors.

    If one of the inputs is a single vector and the other is an array of vectors,
    the function calculates the angle between the single vector and each vector in the array.

    Parameters
    ----------
    v1, v2 : array-like
        Vectors or arrays of vectors. The last dimension must be the vector size.
    units : {'radians', 'degrees'}, optional
        Units of the returned angle. Default is 'degrees'.

    Returns
    -------
    float or numpy.ndarray
        The angle(s) between `v1` and `v2` in the specified units.

    Raises
    ------
    ValueError
        If the vector dimensions do not match or if the shapes are incompatible for broadcasting.
        If `units` is not 'degrees' or 'radians'.

    Examples
    --------
    >>> vec_angle([1, 0, 0], [0, 1, 0])
    90.0
    >>> vec_angle([1, 0, 0], [[0, 1, 0], [1, 0, 0]])
    array([90.,  0.])
    >>> vec_angle([[1, 0], [0, 1]], [1, 1])
    array([45., 45.])
    """

    # Inputs
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

    dot_product = np.einsum('...i,...i->...', v1, v2)
    norm_v1 = np.linalg.norm(v1, axis=-1)
    norm_v2 = np.linalg.norm(v2, axis=-1)

    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    if units == 'degrees':
        return np.degrees(angle_rad).squeeze()
    elif units == 'radians':
        return angle_rad.squeeze()
    else:
        raise ValueError("units must be 'degrees' or 'radians'")

def max_angular_diameter(points:ArrayLike) -> float:
    """
    Compute the maximum angular separation among a set of vectors.

    This returns the largest angle between any pair of input vectors, i.e. the
    maximum angular diameter of the point cloud on the unit sphere (after
    normalization).

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Input vectors.

    Returns
    -------
    float
        Maximum angular separation in degrees.

    Raises
    ------
    ValueError
        If any input vector has zero norm.
    """

    norms = np.linalg.norm(points, axis=1)
    
    # Check for zero vectors to avoid division by zero
    if np.any(norms == 0):
        raise ValueError("Input contains zero vectors, which cannot be normalized.")
    
    vectors_norm = points / norms[:, np.newaxis]
    
    # Compute the dot product matrix between all pairs of normalized vectors
    dot_products = np.dot(vectors_norm, vectors_norm.T)
    
    # Clip the dot products to the valid range of arccos to avoid numerical issues
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Set the diagonal to 1 to exclude self-comparisons
    np.fill_diagonal(dot_products, 1.0)
    
    # Minimum dot product = maximum angle
    min_dot = np.min(dot_products)
    
    angle_max = np.degrees(np.arccos(min_dot))
    return angle_max

def rotation_matrix(ra_center_deg: float, dec_center_deg: float) -> np.ndarray:
    """
    Build a rotation matrix for the observer reference frame (ORF) centered on (RA, Dec).

    The returned matrix rotates J2000 Cartesian vectors so that the provided sky
    direction becomes the new reference center. The transformation implemented is:

    - rotation about z by ``-RA_center``
    - rotation about y by ``Dec_center``

    Parameters
    ----------
    ra_center_deg : float
        Center right ascension in degrees.
    dec_center_deg : float
        Center declination in degrees.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """

    ra_rad  = np.deg2rad(ra_center_deg)
    dec_rad = np.deg2rad(dec_center_deg)
    
    # Rotation around z by -RA_center
    Rz = np.array([
        [np.cos(-ra_rad), -np.sin(-ra_rad), 0],
        [np.sin(-ra_rad),  np.cos(-ra_rad), 0],
        [0, 0, 1]
    ])
    
    # Rotation around y by (90° - DEC_center)
    Ry = np.array([
        [np.cos(dec_rad), 0, np.sin(dec_rad)],
        [0, 1, 0],
        [-np.sin(dec_rad), 0, np.cos(dec_rad)]
    ])
    
    # Total rotation matrix
    R = Ry @ Rz
    return R

def is_in_frame(points: np.ndarray, xrange: tuple, yrange: tuple) -> np.ndarray:
    """
    Return a boolean mask selecting 2D points inside (x, y) bounds.

    Parameters
    ----------
    points : array-like, shape (N, >=2)
        Points array. Only columns 0 and 1 are used (x and y).
    xrange : tuple[float, float]
        Inclusive bounds for x.
    yrange : tuple[float, float]
        Inclusive bounds for y.

    Returns
    -------
    numpy.ndarray of bool, shape (N,)
        True for points inside the rectangle.
    """

    mask =  (points[:, 0] >= xrange[0]) & (points[:, 0] <= xrange[1]) & \
            (points[:, 1] >= yrange[0]) & (points[:, 1] <= yrange[1])
    return mask

def is_vector_in_quadrilateral(v: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Determine whether one or several 3D vectors lies "inside" a quadrilateral 
    defined by four 3D vectors on the unit sphere.

    Parameters
    ----------
    v : array-like, shape (3,) or (N, 3)
        Vector(s) to test.
    quad : array-like, shape (4, 3)
        Quadrilateral vertices (need not be unit length; they are normalized).

    Returns
    -------
    bool or numpy.ndarray of bool
        Boolean result(s). A scalar is returned if the input is a single vector.

    Notes
    -----
    The test is performed by checking the consistency of the signs of dot products against the
    edge normals (cross products of consecutive vertices). Vectors lying on an
    edge (within a small tolerance) are considered inside.

    Raises
    ------
    ValueError
        If ``quad`` does not have shape (4, 3) or if any input vector has zero norm.
    """

    # Inputs
    v=np.asarray(v)
    flag=False
    if v.ndim ==1 : flag=True
    v    = np.atleast_2d(v).astype(float)
    quad = np.array(quad, dtype=float)
    if quad.shape != (4, 3):
        raise ValueError("The quadrilateral must be defined by exactly four 3D vectors (shape (4,3)).")

    n = v.shape[0]

    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norms_v == 0):
        raise ValueError("Some input vectors are zero vectors, cannot normalize.")
    v_norm = v / norms_v

    norms_q = np.linalg.norm(quad, axis=1, keepdims=True)
    if np.any(norms_q == 0):
        raise ValueError("The quadrilateral contains at least one zero vector, cannot normalize.")
    p = quad / norms_q

    # Normals for each edge of the quadrilateral
    # For edge i, the normal is given by cross(p[i], p[i+1]).
    normals = np.array([np.cross(p[i], p[(i+1) % 4]) for i in range(4)])

    # Dot products of each tested vector with the four edge normals
    dot_products = np.dot(v_norm, normals.T)

    
    eps = 1e-12 # Numerical tolerance

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

    # Signs for all edges of these valid vectors
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

def is_visible(points: ArrayLike, body_position: ArrayLike, radii: ArrayLike, threshold: float = 1.e-6, starmode: bool = False):
    """
    Determine whether points are visible (not occulted) by a triaxial ellipsoid.

    Visibility is evaluated for rays starting at the observer (located at
    ``-body_position`` in the same reference frame as ``points``) and pointing
    towards the input direction vectors ``points``. If a ray intersects the
    ellipsoid before reaching the target point, the point is considered occulted.

    Parameters
    ----------
    points : array-like, shape (3,) or (N, 3)
        Direction vectors from the observer toward the points to test.
    body_position : array-like, shape (3,)
        Vector from the body center to the observer (same frame as ``points``).
    radii : array-like, shape (3,)
        Ellipsoid radii (a, b, c).
    threshold : float, optional
        Tolerance used when comparing the intersection distance to the point
        distance. Default is 1e-6.
    starmode : bool, optional
        If True, any intersection implies invisibility (no distance comparison).
        Default is False.

    Returns
    -------
    bool or numpy.ndarray of bool
        Visibility mask. A scalar is returned if the input is a single vector.

    Notes
    -----
    This function relies on :func:`intersect` for the ray/ellipsoid intersection.
    """

    points =np.asarray(points)
    flag=False
    if points.ndim ==1 : flag=True
    points = np.atleast_2d(points)

    close_obj  = (points @ (-body_position) < 0)
    visibility = np.ones(points.shape[0], dtype=bool)

    # Compute intersections only for points that are in the general direction of the body (dot product < 0)
    if np.any(close_obj) :
        
        close_dir = points[close_obj]

        intercepts, found = intersect(-body_position, close_dir, radii)
        intercepts = np.atleast_2d(intercepts)
        found      = np.atleast_1d(found)

        if starmode:
            # If starmode is True, any intersection implies the point is invisible
            visibility[close_obj] = ~found
            if flag : return visibility[0]
            return visibility

        # If an intersection is found, check visibility by comparing distances
        if np.any(found):
            distance_to_intercept = np.linalg.norm(intercepts[found] + body_position, axis=1)
            distance_to_point = np.linalg.norm(close_dir[found], axis=1)

            # The point is visible if the distance to the intersection is greater
            # or equal to the distance to the point (minus the threshold)
            vis_found = distance_to_intercept >= (distance_to_point - threshold)
            visibility_close = np.ones(found.shape[0], dtype=bool)
            visibility_close[found] = vis_found
            visibility[close_obj] = visibility_close


    if flag : return visibility[0]
    return visibility

def order_bool(bool_list: ArrayLike) -> np.ndarray:
    """
    Return indices that rotate a boolean sequence to group identical values.

    This helper returns a permutation of indices intended to reorder a boolean list
    so that transitions are moved to the ends (i.e., values are grouped as much as
    possible) by circularly rotating the sequence.

    Parameters
    ----------
    bool_list : array-like of bool
        Input boolean sequence.

    Returns
    -------
    numpy.ndarray of int
        Indices that can be used to reorder the input (e.g., ``arr[idx]``).

    Notes
    -----
    If the input is all True or all False, the identity ordering is returned.
    """

    bool_list = np.array(bool_list)

    index = list(range(0,len(bool_list)))

    # If all values are the same, return the original order
    if np.all(bool_list) or np.all(~bool_list) :
        return np.array(index)
    
    # If the first and last values are True, rotate until the first value is False
    if bool_list[0] and bool_list[-1] :
        bool_list = list(bool_list)
        while bool_list[0] :
            bool_list = bool_list[1:] + [bool_list[0]]
            index     = index[1:]     + [index[0]]
        return np.array(index)

    # If the first and last values are False, rotate until the last value is True
    if not bool_list[0] and not bool_list[-1] :
        bool_list = list(bool_list)
        while not bool_list[-1] :
            bool_list = [bool_list[-1]] + bool_list[:-1] 
            index     = [index[-1]]     + index[:-1] 
        return np.array(index)

def intersect(observer: ArrayLike, directions: ArrayLike, radii: ArrayLike, closest_point: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute intersections between rays and a triaxial ellipsoid.

    Rays are defined by an observer position and one or more direction vectors.
    When ``closest_point`` is False, the function returns the first intersection
    point along each ray (smallest root). When ``closest_point`` is True, it
    returns the point on the ray closest to the ellipsoid center (not necessarily
    an intersection point), along with a mask indicating whether the ray intersects
    the ellipsoid.

    Parameters
    ----------
    observer : array-like, shape (3,)
        Ray origin.
    directions : array-like, shape (3,) or (N, 3)
        Ray direction vectors.
    radii : array-like, shape (3,)
        Ellipsoid radii (a, b, c).
    closest_point : bool, optional
        If True, return the closest point along each ray (t = -B/(2A)) and the
        intersection mask. Default is False.

    Returns
    -------
    (intercepts, found) : tuple
        intercepts : numpy.ndarray, shape (N, 3)
            Intersection points (or closest points if ``closest_point`` is True).
            Non-intersecting rays have NaNs when ``closest_point`` is False.
        found : numpy.ndarray of bool, shape (N,)
            True where an intersection exists (delta >= 0).

    Notes
    -----
    Directions do not need to be unit vectors.
    """

    a,  b,  c  = radii
    px, py, pz = observer

    directions=np.asarray(directions)
    flag=False
    if directions.ndim == 1 : flag=True
    directions = np.atleast_2d(directions)

    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    inv_a2, inv_b2, inv_c2 = 1.0 / a**2, 1.0 / b**2, 1.0 / c**2

    # Coefficients of the quadratic equation A*t^2 + B*t + C = 0 for ray-ellipsoid intersection
    A = dx**2 * inv_a2 + dy**2 * inv_b2 + dz**2 * inv_c2
    B = 2.0 * (px * dx * inv_a2 + py * dy * inv_b2 + pz * dz * inv_c2)
    C = px**2 * inv_a2 + py**2 * inv_b2 + pz**2 * inv_c2 - 1.0

    delta = B**2 - 4 * A * C
    found = (delta >= 0)

    if closest_point:
        # t corresponding to the point closest to the ellipsoid center along the ray
        t = -B / (2 * A)
        intercepts = observer + directions * t[:, None]
        if flag : return intercepts[0], found[0]
        return intercepts, found

    intercepts = np.full((directions.shape[0], 3), np.nan, dtype=np.float64)

    if not np.any(found):
        if flag : return intercepts[0], found[0]
        return intercepts, found

    # Calculate roots for rays where intersection exists
    sqrt_delta = np.sqrt(delta[found])
    twoA = 2.0 * A[found]
    t1 = (-B[found] - sqrt_delta) / twoA
    t2 = (-B[found] + sqrt_delta) / twoA

    # Minimum t corresponding to the first point encountered on the ray
    t_min = np.minimum(t1, t2)[:, None]

    intercepts[found] = observer + directions[found] * t_min

    if flag : return intercepts[0], found[0]
    return intercepts, found

def ellipsoid_coords(radii: ArrayLike, lon: ArrayLike, lat: ArrayLike) -> np.ndarray:
    """
    Compute Cartesian surface coordinates on a triaxial ellipsoid.

    Parameters
    ----------
    radii : array-like, shape (3,)
        Ellipsoid radii (a, b, c).
    lon : float or array-like
        Longitude(s) in radians.
    lat : float or array-like
        Latitude(s) in radians.

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates on the ellipsoid surface. Shape (3,) for scalar
        inputs or (N, 3) for array inputs.

    Raises
    ------
    ValueError
        If radii does not contain exactly three elements or if lon/lat cannot be
        broadcast to a common shape.
    """

    # Inputs
    radii = np.asarray(radii, dtype=float)
    if radii.shape[-1] != 3:
        raise ValueError("`radii` must contain exactly three elements: (a, b, c).")
    a, b, c = radii

    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)

    # Attempt to broadcast lon and lat to compatible shapes
    try:
        lon, lat = np.broadcast_arrays(lon, lat)
    except ValueError as e:
        raise ValueError("`lon` and `lat` could not be broadcast to a common shape.") from e

    # Cartesian coordinates
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    x = a * cos_lat * cos_lon
    y = b * cos_lat * sin_lon
    z = c * sin_lat

    # Stack into a single array of vectors
    coords = np.column_stack((x, y, z))

    if coords.shape[0] == 1:
        return coords[0]
    return coords

def ellipsoid_xyz(radii: ArrayLike, vec: ArrayLike, return_altitude: bool = True, units:Literal['radians', 'degrees'] = 'radians'):
    """
    Convert Cartesian coordinates to (lon, lat, altitude) relative to a triaxial ellipsoid.

    Longitude/latitude are computed from normalized coordinates (x/a, y/b, z/c).
    The returned "altitude" is a radial height-like quantity derived from the
    distance to the origin and the scaled radius factor ``s``.

    Parameters
    ----------
    radii : array-like, shape (3,)
        Ellipsoid radii (a, b, c).
    vec : array-like, shape (3,) or (N, 3)
        Cartesian coordinates.
    return_altitude : bool, optional
        If True, return a radial altitude-like quantity. If False, return the
        Euclidean distance to the origin. Default is True.
    units : {'radians', 'degrees'}, optional
        Units for longitude and latitude. Default is 'radians'.

    Returns
    -------
    longitude : float or numpy.ndarray
        Longitude (wrapped into [0, 2*pi) in radians, or [0, 360) in degrees).
    latitude : float or numpy.ndarray
        Latitude in radians (or degrees if requested).
    altitude_or_distance : float or numpy.ndarray
        Altitude if ``return_altitude`` is True, else distance.

    Raises
    ------
    ValueError
        If any radius is non-positive.
    """

    a, b, c = radii

    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("All radii must be positive numbers.")

    vec = np.atleast_2d(vec)
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]

    x_norm = x / a
    y_norm = y / b
    z_norm = z / c

    s = np.sqrt(x_norm**2 + y_norm**2 + z_norm**2)

    # Distance
    norm_vec = np.linalg.norm(vec, axis=1)

    # Longitude and latitude
    longitude = np.arctan2(y_norm, x_norm)
    longitude = np.mod(longitude, 2*np.pi)
    latitude  = np.arcsin(np.clip(z_norm / s, -1.0, 1.0))

    if units=="degrees" :
        longitude = np.mod(np.degrees(longitude), 360)
        latitude  = np.degrees(latitude)

    if return_altitude:
        altitude = norm_vec * (1 - 1 / s)
        if vec.shape[0]==1 :
            return longitude.squeeze(), latitude.squeeze(), altitude.squeeze()
        else : return longitude, latitude, altitude
    else:
        if vec.shape[0]==1 :
            return longitude.squeeze(), latitude.squeeze(), norm_vec.squeeze()
        else : return longitude, latitude, norm_vec
