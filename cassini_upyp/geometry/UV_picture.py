from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from matplotlib.axes   import Axes
from matplotlib.figure import Figure
if TYPE_CHECKING:
    from ..uvisdata import UVIS_Observation


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as mpath
from pathlib import Path
import spiceypy as spice


def UV_picture(
        obs: UVIS_Observation,
        wl_range: tuple[float, float] | None = None,
        alt_circles: Sequence[float] | None = None,
        cmap: str = 'plasma',
        vmin: float | None = None,
        vmax: float | None = None,
        levels: int = 100,
        nx: int = 20,
        ny: int = 20,
        interpol: bool = True,
        ax: Axes | None = None,
        annotate: bool = True,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None
    ):
    """
    Build and plot a projected UV radiance image.

    The function projects each UVIS pixel footprint onto a 2D plane centered on the
    target body and fill the quadrilateral with the integrated radiance.
    The projection is built from per-pixel tangent-point geometry of its 4 corners (longitude,
    latitude, altitude).
    Regions of space covered by multiple pixel footprints are averaged,
    while grid cells not covered byany footprint are masked.

    Parameters
    ----------
    obs : :class:`cassini_upyp.uvisdata.UVIS_Observation`
        Observation providing geometry and radiance information.

    wl_range : tuple[float, float] or None, optional
        Wavelength interval passed to ``obs.integrate_radiance``. If ``None``,
        the full wavelength range is used (see :func:`cassini_upyp.uvisdata.UVIS_Observation.integrate_radiance`).

    alt_circles : sequence of float or None, optional
        Altitudes (km) of dashed reference circles plotted around the apparent body
        disk.

    cmap : str, optional
        Matplotlib colormap name. Default is ``"plasma"``.

    vmin, vmax : float or None, optional
        Lower and upper color scale bounds. If ``None``, they are inferred from the
        computed image.

    levels : int, optional
        Number of contour levels used when ``interpol=True``. Default is ``100``.

    nx, ny : int, optional
        Number of grid samples along the x and y axes of the projected map.
        Defaults are ``20`` and ``20``.

    interpol : bool, optional
        If ``True``, render the map with ``Axes.tricontourf`` using valid grid
        points only. If ``False``, render the gridded image directly with
        ``Axes.imshow``. Default is ``True``.
        Interpol is useful to fill the gaps between the pixels FOV, but it may produce artifacts if the grid is too coarse.

    ax : matplotlib.axes.Axes or None, optional
        Existing axes on which to draw. If ``None``, a new figure and axes are
        created.

    annotate : bool, optional
        If ``True``, add colorbar, compass arrows, longitude label, pole marker,
        and axis labels. If ``False``, axes are hidden and only the map/body outline
        are drawn as a clean raw image.

    xlim, ylim : tuple[float, float] or None, optional
        Explicit x/y limits (km) for the projected grid. If ``None``, limits are
        inferred from the projected pixel footprints.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the UV image.
    ax : matplotlib.axes.Axes
        Axes containing the UV image.

    Raises
    ------
    ValueError
        If no valid projected radiance values are available.

    Notes
    -----
    - The body disk is drawn as an apparent ellipse derived from SPICE body radii
      and the sub-spacecraft latitude.
    - A target-specific longitude convention is applied for Titan (x-axis flip).
    - Pixel index 59 is skipped (instrument-specific exclusion).
    - Grid cells covered by multiple pixel footprints are averaged.

    See Also
    --------
    :func:`cassini_upyp.uvisdata.UVIS_Observation.check_stars`
    """

    # 1 - PREPARE DATA
    #-----------------


    # Get body radius
    from ..utils import env_config
    env = env_config()
    pck = Path(env.pck_path) # Planetary constants kernel

    spice.furnsh(str(pck))
    radii = spice.bodvrd(obs.target, 'RADII', 3)[1]
    spice.unload(str(pck))

    r_p, r_e = radii[0], np.mean(radii[1:])

    # Pixel coordinates
    from .computational import ellipsoid_radius
    lats = obs.pixel_LOS["lat"]
    lons = obs.pixel_LOS["lon"]
    lt   = obs.pixel_LOS["lt"]
    alts = obs.pixel_LOS["alt"] + ellipsoid_radius(radii, lons, lats)

    xx = alts*np.cos(np.radians(lats))
    yy = alts*np.sin(np.radians(lats))

    

    # Sort pixels left/right by longitude
    sc_lon = np.mean(obs.sub_sc_point[:,0])
    lon_1 = sc_lon-180
    if lon_1<0:
        mask = (
            ((lons >= lon_1+360) & (lons < 0))     |
            ((lons >= 0)         & (lons < sc_lon))
        )
        xx[mask] *= -1
    else:
        mask = ((lons >= lon_1)  & (lons < sc_lon))
        xx[mask] *= -1

    # Invert longitude for Titan
    if obs.target.upper() == 'TITAN':
        xx*= -1


    # 2 - BUILD IMAGE
    #----------------
    signal = obs.integrate_radiance(wl_range=wl_range)

    # GRID
    if xlim is None:
        xlim = (np.nanmin(xx), np.nanmax(xx))
    if ylim is None:
        ylim = (np.nanmin(yy), np.nanmax(yy))

    X=np.linspace(xlim[0], xlim[1], nx)
    Y=np.linspace(ylim[0], ylim[1], ny)

    XX, YY  = np.meshgrid(X,Y)

    weights = np.zeros_like(XX)
    image   = np.zeros_like(XX)

    for i_pic in range(signal.shape[0]):
        for i_pix in range(signal.shape[1]):
            if obs.pixel_corrupted[i_pic, i_pix]:
                continue
            if i_pix==59: continue # pixel 59 is erroneous

            # Pixel quadrilateral in physical coordinates
            corners = []
            for j in range(1,5):
                corners.append([xx[i_pic, i_pix, j], yy[i_pic, i_pix, j]])
            corners.append([xx[i_pic, i_pix, 1], yy[i_pic, i_pix, 1]])
            pixel = mpath(corners)
            corners = np.array(corners)

            # Bounding box to limit the grid points to check
            xmin = np.min(corners[:, 0])
            xmax = np.max(corners[:, 0])
            ymin = np.min(corners[:, 1])
            ymax = np.max(corners[:, 1])

            ix0 = np.searchsorted(X, xmin, side="left")
            ix1 = np.searchsorted(X, xmax, side="right")
            iy0 = np.searchsorted(Y, ymin, side="left")
            iy1 = np.searchsorted(Y, ymax, side="right")

            ix0 = max(ix0, 0)
            ix1 = min(ix1, nx)
            iy0 = max(iy0, 0)
            iy1 = min(iy1, ny)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            
            # Sub-grid only inside bbox
            XX_sub = XX[iy0:iy1, ix0:ix1]
            YY_sub = YY[iy0:iy1, ix0:ix1]

            points_sub = np.column_stack([XX_sub.ravel(), YY_sub.ravel()])

            # Find points inside the pixel FOV
            inside = pixel.contains_points(points_sub).reshape(XX_sub.shape)

            if np.any(inside):
                image[iy0:iy1, ix0:ix1][inside]   += signal[i_pic, i_pix]
                weights[iy0:iy1, ix0:ix1][inside] += 1.0


    mask = weights > 0
    image[mask] /= weights[mask]
    image[~mask] = np.nan

    if not np.isfinite(image).any():
        raise ValueError("No valid data available to build UV picture.")




    # 3 - PLOT
    #---------
    if vmin is None:
        vmin = np.nanmin(image)
    if vmax is None:
        vmax = np.nanmax(image)

    if vmax<np.nanmax(image):
        extend = 'max'
    else: extend = None


    # PLOT IMAGE
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if not interpol:
        im = ax.imshow(image, extent=(X[0], X[-1], Y[0], Y[-1]), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        
    else:
        # Retrieve points within pixel FOV
        # tricontourf() will use them to interpolate between the pixels FOV
        X_final = XX.ravel()
        Y_final = YY.ravel()
        Z_final = image.ravel()

        mask_valid = np.isfinite(Z_final)

        X_final = X_final[mask_valid]
        Y_final = Y_final[mask_valid]
        Z_final = Z_final[mask_valid]

        levels_array = np.linspace(vmin, vmax, levels)
        im= ax.tricontourf(X_final, Y_final, Z_final, levels=levels_array, vmin=vmin, vmax=vmax, cmap=cmap, extend=extend)
    
    ax.set_aspect('equal')



    # 4- MISCELLANEOUS ELEMENTS
    #--------------------------
    # Draw body's disk as apparent ellipse
    sc_lat = np.mean(obs.sub_sc_point[:,1])

    A = r_e
    B = np.sqrt(r_e*r_e*(np.sin(np.radians(sc_lat)))**2 + r_p*r_p*(np.cos(np.radians(sc_lat)))**2)
    angles = np.linspace(0, 2*np.pi, 100)
    ax.plot(A*np.cos(angles), B*np.sin(angles), color='black', lw=1.5)


    if not annotate:
        ax.set_axis_off()
        return fig, ax
    plt.colorbar(im, ax=ax, label='Integrated radiance (kR)', aspect=40, extend=extend)

    # Draw altitude circle
    if alt_circles is not None:
        for alt in alt_circles:
            ax.plot((A+alt)*np.cos(angles),  (B+alt)*np.sin(angles), color='black', ls='--', lw=1)

    # Arrows / compass
    r_ref = min(r_e, r_p)
    arrow_len = 0.12 * r_ref
    label_offset = 0.035 * r_ref
    text_box = dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.6,
        boxstyle="round,pad=0.12,rounding_size=0.2",
    )

    arrow_kw = dict(
        arrowstyle="-|>",   # cleaner than ax.arrow
        color="black",
        lw=1.5,
        mutation_scale=12,  # arrow head size in points (screen units)
        shrinkA=0,
        shrinkB=0,
    )

    # North arrow
    ax.annotate(
        "",
        xy=(0, arrow_len),
        xytext=(0, 0),
        arrowprops=arrow_kw,
    )
    ax.text(
        0,
        arrow_len + label_offset,
        "N",
        ha="center",
        va="bottom",
        bbox=text_box,
    )

    # East arrow
    ax.annotate(
        "",
        xy=(arrow_len, 0),
        xytext=(0, 0),
        arrowprops=arrow_kw,
    )
    ax.text(
        arrow_len + label_offset,
        0,
        "E",
        ha="left",
        va="center",
        bbox=text_box,
    )

    # Longitude label (placed relative to the current visible extent)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0

    ax.text(
        x0 + 0.03 * dx,
        y0 + 0.03 * dy,
        f"Longitude {np.mean(obs.sub_sc_point[:, 0]):.1f}°",
        ha="left",
        va="bottom",
        bbox=text_box,
    )

    # Pole marker
    phi = np.radians(sc_lat)

    # Projected y position of the visible rotational pole
    y_pole = r_p * np.cos(phi)
    # cos(phi) is positive for physical sub-observer latitudes in [-90, 90]
    # so apply the hemisphere sign explicitly:
    y_pole = np.sign(sc_lat) * abs(y_pole)

    text_box = dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.5,
        boxstyle="round,pad=0.08,rounding_size=0.2",
    )

    offset = 0.04 * min(r_e, r_p)

    if sc_lat >= 0:
        ax.plot(0, y_pole, marker="x", color="black")
        ax.text(0, y_pole - offset, "North pole", ha="center", va="top", bbox=text_box)
    else:
        ax.plot(0, y_pole, marker="x", color="black")
        ax.text(0, y_pole + offset, "South pole", ha="center", va="bottom", bbox=text_box)


    ax.set_xlabel('Distance from center (km)')
    ax.set_ylabel('Distance from center (km)')

    return fig, ax