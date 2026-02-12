import spiceypy as spice
import numpy as np

from .computational import (
    xyz2radec, is_visible, order_bool,
    ellipsoid_coords, ellipsoid_xyz, intersect,
    vec_angle
)

class Geometer:
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
        self.zorder = -round(self.target_distance)

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
            latgrid = np.linspace(-np.pi/2, np.pi/2, 37)
        

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
        phase = vec_angle(-LOS,           tangent_point_to_sun)
        ems   = vec_angle(-LOS,           tangent_point)

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
