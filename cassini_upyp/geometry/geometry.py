from __future__ import annotations
from typing import Literal

import numpy as np
import spiceypy as spice


import warnings
from pathlib import Path
from scipy.constants import astronomical_unit

from ..kernellib import *
from ..uvisdata import UVIS_Observation
from ..utils import env_config, plot_config
env        = env_config()
plotconfig = plot_config()

from .spice_engine import Geometer
from .computational import(
    rotation_matrix,
    xyz2radec, radec2xyz,
    vec_angle, max_angular_diameter,
    ellipsoid_xyz, is_vector_in_quadrilateral
)


def stars_pickles():
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

    star_file = Path(env.star_file) # stars.npy
    
    if not star_file.exists():
        raise FileNotFoundError(f"File not found: {star_file}")

    return np.load(star_file)


class Geometry:
    def __init__(self, ET:float,
                 meta_kernel = None, other_bodies=plotconfig.FOV_objects,
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
            else :
                spice.furnsh(meta_kernel)

        # Call geometry engine
        if target : self.target = target
        elif u is not None : self.target = u.target
        else : raise TypeError("Geometry object requires a valid target.")

        if self.main : self.geo_engine = Geometer(self.target, 'CASSINI', self.ET, offset=offset)
        else : self.geo_engine = Geometer(self.target, 'CASSINI', self.ET)
        

        self.UTC_time = spice.et2utc(self.ET, 'C', 0)


        
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
            if np.any(~self.dayside) and len(self.terminator['XYZ'])>0 :
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
            for lon in plotconfig.lon_line_grid :
                lon_line = self.geo_engine.get_lon_line(np.radians(lon))

                self.lon_lines.append({
                    'XYZ'   : lon_line,
                    'RADEC' : xyz2radec(lon_line, units='degrees')
                })

            for lat in plotconfig.lat_line_grid :
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

            if other_bodies is not None and len(other_bodies) > 0:
                other_bodies = [e.upper() for e in other_bodies]

                for target2 in other_bodies :
                    # Don't recompute main target
                    if target2 == self.target : continue

                    try :

                        self.other_targets.append(
                            Geometry(self.ET, main=self, target=target2)
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
            self.stars_orf    = xyz2radec(self.stars['XYZ']   @ self.R.T, units=units, ra_range=ra_range)
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


    def plot(self, *args, **kwargs):
        from .plot import plot
        return plot(self, *args, **kwargs)

