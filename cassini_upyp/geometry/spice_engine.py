from typing import Optional, Tuple
import numpy.typing as npt

import spiceypy as spice
import numpy as np

from .computational import (
    xyz2radec, is_visible, order_bool,
    ellipsoid_coords, ellipsoid_xyz, intersect,
    vec_angle
)

class Geometer:
    """
    Low-level geometry engine for apparent target/line-of-sight computations.

    `Geometer` wraps a set of SPICE and geometric routine calls to compute viewing geometry for a given
    target body as seen from a given observer at a specified ephemeris time. It
    provides vectors and helpers to compute the apparent limb (ellipse), the terminator, visible
    latitude/longitude grid lines, and line-of-sight (LOS) tangent/intersection
    quantities used by the higher-level :class:`cassini_upyp.geometry.geometry.Geometry` container.

    Parameters
    ----------
    target : str
        SPICE name of the target body (e.g., ``"TITAN"``). Case-insensitive.
    observer : str
        SPICE name of the observer (e.g., ``"CASSINI"``). Case-insensitive.
    et : float
        Ephemeris time (seconds past J2000, TDB).
    offset : array-like, optional
        Optional offset vector passed to the geometry engine to be applied
        on the main target body, for example to compensate some known bias.
        Default is ``(0, 0, 0)`` as it should remain unused.

    Attributes
    ----------
    planet : str
        Uppercase target name.
    observer : str
        Uppercase observer name.
    et : float
        Ephemeris time (seconds past J2000, TDB).
    radii : ndarray, shape (3,)
        Target tri-axial radii [km] as returned by SPICE ``BODVRD``.
    obs_from_planet_brf : ndarray, shape (3,)
        Observer position vector in the target body-fixed frame (``IAU_<TARGET>``),
        i.e. vector from target center to observer, with optional ``offset`` added.
    planet_from_obs_j2k : ndarray, shape (3,)
        Target position vector in J2000 as seen from the observer (LT+S).
    sun_from_obs_j2k : ndarray, shape (3,)
        Sun position vector in J2000 as seen from the observer (LT+S).
    target_RA, target_DEC : float
        Apparent target center coordinates (RA, Dec) in radians (J2000).
    target_distance : float
        Apparent distance from observer to target center (same units as SPICE output).
    rotation_IAU_J2K, rotation_J2K_IAU : ndarray, shape (3, 3)
        Rotation matrices between the target body-fixed frame and J2000 at ``et``.
    planet_to_sun_brf : ndarray, shape (3,)
        Vector from target to Sun expressed in the target body-fixed frame.

    Notes
    -----
    All SPICE kernels required for the target/observer configuration must be loaded
    before instantiating this class.

    See Also
    --------
    :class:`cassini_upyp.geometry.geometry.Geometry` : High-level container building additional products (FOV, stars, plots).
    """

    def __init__(self, target: str, observer: str, et: float, offset: npt.ArrayLike = np.array((0,0,0))):
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
        self.zorder = -round(self.target_distance)

        self.rotation_IAU_J2K = spice.pxform('IAU_' + self.planet, 'J2000', self.et)
        self.rotation_J2K_IAU = spice.pxform('J2000', 'IAU_' + self.planet, self.et)


        # Vecteur de la planète vers le Soleil depuis l'observateur
        planet_to_sun_j2000 = self.sun_from_obs_j2k - self.planet_from_obs_j2k
        self.planet_to_sun_brf = np.dot(self.rotation_J2K_IAU, planet_to_sun_j2000)


    def get_ellipse(self, npoints:int = 100, altitude:float = 0, flag_dayside:bool = True) -> Tuple[npt.NDArray[np.floating], Optional[npt.NDArray[np.bool_]]]:
        """
        Compute the apparent target limb as an ellipse and optionally flag day/night points.

        The limb is computed in the target body-fixed frame using SPICE ``edlimb`` for a
        triaxial ellipsoid (with an optional altitude offset applied to the radii to compute iso-altitude ellipse).
        The resulting limb points are returned as direction vectors from the observer,
        expressed in the J2000 frame.

        If ``flag_dayside`` is ``True``, each limb point is classified as belonging to
        the illuminated hemisphere (day side) or not, based on the Sun direction in the
        target body-fixed frame. The returned arrays are reordered so that all day-side
        points are grouped together (then night-side points), following the ordering
        produced by :func:`cassini_upyp.geometry.computational.order_bool`.

        Parameters
        ----------
        npoints : int, optional
            Number of points used to sample the limb ellipse. Default is ``100``.
        altitude : float, optional
            Altitude added to each ellipsoid radius when computing the limb [km].
            Default is ``0``.
        flag_dayside : bool, optional
            If ``True``, also return a boolean mask identifying points on the day side.
            Default is ``True``.

        Returns
        -------
        tuple
            ``(limb_points, dayside)`` where:

            - limb_points : numpy.ndarray, shape (npoints, 3)
                Observer-to-limb direction vectors expressed in J2000.
            - dayside : numpy.ndarray of bool, shape (npoints,), or None
                Boolean mask indicating day-side limb points after reordering, or
                ``None`` if ``flag_dayside`` is ``False``.

        Notes
        -----
        - The limb is computed for a triaxial ellipsoid using SPICE ``edlimb`` and
        converted to center/axes form with ``el2cgv``.
        - Day-side classification relies on a visibility test performed in the body-fixed
        frame (see :func:`cassini_upyp.geometry.computational.is_visible`).
        """

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

    def is_visible(self, points:npt.ArrayLike, starmode:bool = False, threshold:float = 1.e-6) -> npt.NDArray[np.bool_]:
        """
        Test whether points are visible from the observer above the target ellipsoid.

        Calls :func:`cassini_upyp.geometry.computational.is_visible`.

        Parameters
        ----------
        points : array-like
            Vectors (in the target body-fixed frame) from the target center to the
            points to be tested.
        starmode : bool, optional
            Reserved for future use. Currently ignored. Default is ``False``.
        threshold : float, optional
            Visibility threshold passed to :func:`cassini_upyp.geometry.computational.is_visible`. Default is ``1e-6``.

        Returns
        -------
        numpy.ndarray of bool
            Boolean mask indicating which points are visible.
        """
        return is_visible(points, -self.obs_from_planet_brf, self.radii, threshold=threshold)

    def get_terminator(self, npoints:int = 100, full:bool = False) -> npt.NDArray[np.floating]:
        """
        Compute the apparent terminator curve (day/night boundary) as seen by the observer.

        The terminator is computed by calling SPICE ``edlimb`` with the Sun direction
        vector expressed in the target body-fixed frame, producing the limb of the
        illumination ellipse. The resulting points are then expressed as direction
        vectors from the observer and rotated into the J2000 frame.

        By default, only the portion of the terminator that is visible from the
        observer is returned. If ``full`` is ``True``, visibility is not enforced and
        the full sampled curve is returned.

        Parameters
        ----------
        npoints : int, optional
            Number of points used to sample the terminator ellipse. Default is ``100``.
        full : bool, optional
            If ``True``, return the full sampled terminator without applying the
            visibility mask. Default is ``False``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(M, 3)`` containing observer-to-terminator direction vectors
            expressed in J2000, where ``M <= npoints`` when ``full`` is ``False``.
        """

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
    
    def get_lon_line(self, lon:float, latgrid:npt.ArrayLike = None) -> npt.NDArray[np.floating]:
        """
        Compute the visible portion of a target longitude line.

        The longitude line is generated on the target ellipsoid, transformed into
        observer-centered vectors in the target body-fixed frame, filtered for
        visibility from the observer, then rotated into the J2000 frame.

        Parameters
        ----------
        lon : float
            Longitude of the meridian to compute, in radians (body-fixed).
        latgrid : array-like or None, optional
            Latitude sampling grid in radians. If ``None``, uses 37 points uniformly
            spaced from ``-π/2`` to ``π/2`` with a step of 5°.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(M, 3)`` containing observer-to-surface direction vectors
            along the visible part of the meridian, expressed in J2000.
        """

        if latgrid is None:
            latgrid = np.linspace(-np.pi/2, np.pi/2, 37)
        

        vec    = ellipsoid_coords(self.radii, lon, latgrid)
        
        points = vec - self.obs_from_planet_brf
        points = points[self.is_visible(points)]

        
        points = points @ self.rotation_IAU_J2K.T

        return points
    
    def get_lat_line(self, lat:float, longrid:npt.ArrayLike = None) -> npt.NDArray[np.floating]:
        """
        Compute the visible portion of a target latitude line.

        The latitude line is generated on the target ellipsoid, transformed into
        observer-centered vectors in the target body-fixed frame, filtered for
        visibility from the observer, then rotated into the J2000 frame.

        Parameters
        ----------
        lat : float
            Latitude of the parallel to compute, in radians (body-fixed).
        longrid : array-like or None, optional
            Longitude sampling grid in radians. If ``None``, uses 37 points uniformly
            spaced from ``0`` to ``2*π`` with a step of 10°.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(M, 3)`` containing observer-to-surface direction vectors
            along the visible part of the parallel, expressed in J2000.
        """

        if longrid is None:
            longrid = np.linspace(0, 2*np.pi, 37)
        
        vec    = ellipsoid_coords(self.radii, longrid, lat)
        
        points = vec - self.obs_from_planet_brf
        points = points[self.is_visible(points)]

        
        points = points @ self.rotation_IAU_J2K.T

        return points

    def LOS_tangent(self, LOS:npt.ArrayLike, J2000:bool = True) -> npt.NDArray[np.floating]:
        """
        Compute line-of-sight (LOS) tangent-point geometry on the target ellipsoid.

        For each input LOS direction vector, this method computes the closest approach
        (tangent point) and/or the intersection point to the target triaxial ellipsoid as seen from the observer,
        then derives standard geometric quantities at that point (longitude, latitude,
        altitude, solar zenith angle, phase angle, emission angle, and local solar time).

        By default, LOS vectors are assumed to be expressed in J2000 and are rotated
        into the target body-fixed frame (``IAU_<TARGET>``) before intersection/tangent
        computations.

        Parameters
        ----------
        LOS : array-like
            Array of LOS direction vectors. Expected shape is ``(N, 3)``.
        J2000 : bool, optional
            If ``True`` (default), interpret ``LOS`` in J2000 and rotate it into the
            target body-fixed frame before computations. If ``False``, ``LOS`` is
            assumed to already be expressed in the target body-fixed frame.

        Returns
        -------
        numpy.ndarray
            Structured array of shape ``(N,)`` with fields:

            - ``lon``   : Intersection longitude [°].
            - ``lat``   : Intersection latitude [°].
            - ``alt``   : Tangent point altitude above the reference ellipsoid [km]. (Negative if LOS intersects the ellipsoid)
            - ``sza``   : Solar zenith angle [°] at the intersection point.
            - ``phase`` : Phase angle [°] at the intersection point.
            - ``ems``   : Emission angle [°] at the intersection point.
            - ``lt``    : Local solar time [hours] (0–24) at the intersection point.
            - ``t_lon`` : Longitude of the closest point to the ellipsoid center along the LOS (tangent point) [°].
            - ``t_lat`` : Latitude of the closest point to the ellipsoid center along the LOS (tangent point) [°].
            - ``t_sza`` : Solar zenith angle at the tangent point [°].
            - ``t_phase`` : Phase angle at the tangent point [°].
            - ``t_ems`` : Emission angle at the tangent point [°].
            - ``t_lt`` : Local solar time at the tangent point [hours] (0–24).

        Notes
        -----
        - The tangent point is computed via :func:`cassini_upyp.geometry.computational.intersect` with ``closest_point=True``.
          The intersection point (if it exists) is computed with ``closest_point=False``.
        - Longitudes are flipped for Titan (``lon = 360 - lon``) to match the westward-positive convention.
        - Local solar time is computed from longitude relative to the sub-solar
          longitude and wrapped into ``[0, 24)``.
        - The intersection altitude is not given since it is a trivial 0.

        See Also
        --------
        :func:`cassini_upyp.geometry.computational.intersect` : Compute LOS intersection / closest-approach with an ellipsoid.
        :func:`cassini_upyp.geometry.computational.ellipsoid_xyz` : Convert Cartesian coordinates to (lon, lat, alt) on an ellipsoid.
        :func:`cassini_upyp.geometry.computational.vec_angle` : Compute angles between vectors.
        """

        if J2000 :
            LOS = LOS @ self.rotation_J2K_IAU.T  # Shape: (npoints, 3)
        
        planet_to_sun_brf,_ = spice.spkpos('SUN', self.et, 'IAU_' + self.planet, 'LT+S', self.planet)

        self.sub_solar_longitude, _, _ = ellipsoid_xyz(self.radii, planet_to_sun_brf, units='degrees')

        
        # Properties for closest point to the target center (tangent point)
        tangent_point, found  = intersect(self.obs_from_planet_brf, LOS, self.radii, closest_point=True)
        tangent_point_to_sun = planet_to_sun_brf - tangent_point

        t_lons, t_lats, alts = ellipsoid_xyz(self.radii, tangent_point, units='degrees')

        t_lst = 12.0 + ((t_lons - self.sub_solar_longitude)) * (24.0 / 360.0)
        t_lst = t_lst % 24.0

        

        
        t_sza   = vec_angle(tangent_point, tangent_point_to_sun)
        t_phase = vec_angle(-LOS,          tangent_point_to_sun)
        t_ems   = vec_angle(-LOS,          tangent_point)
        
        # Properties for intersection point with the surface
        # (no difference if LOS is not looking at the disk)
        tangent_point, found  = intersect(self.obs_from_planet_brf, LOS, self.radii, closest_point=False)
        tangent_point_to_sun = planet_to_sun_brf - tangent_point

        lons, lats, _ = ellipsoid_xyz(self.radii, tangent_point, units='degrees')




        lst = 12.0 + ((lons - self.sub_solar_longitude)) * (24.0 / 360.0)
        lst = lst % 24.0

        
        if self.planet.upper()=='TITAN':
            self.sub_solar_longitude = 360 - self.sub_solar_longitude
            lons   = 360-lons
            t_lons = 360-t_lons
            
        sza   = vec_angle(tangent_point, tangent_point_to_sun)
        phase = vec_angle(-LOS,          tangent_point_to_sun)
        ems   = vec_angle(-LOS,          tangent_point)


        keys  = ['lon', 'lat', 'alt', 'sza', 'phase', 'ems', "lt", "t_lon", "t_lat", "t_sza", "t_phase", "t_ems", "t_lt"]
        param = [ lons,  lats,  alts,  sza,   phase,   ems,   lst,  t_lons,  t_lats,  t_sza,   t_phase,   t_ems,   t_lst]
        dtype = [(k,float) for k in keys]
        
        params = np.zeros(LOS.shape[0], dtype=dtype)
        for k,p in zip(keys,param) :
            params[k] = p

        return params
