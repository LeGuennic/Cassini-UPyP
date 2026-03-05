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

    Each UVIS pixel footprint is projected onto a regular grid using the
    tangent-point geometry of its four corners (local time, latitude, altitude).
    Grid cells covered by several footprints are averaged; cells with no
    coverage are masked.  The result is rendered either as a contour fill
    (``interpol=True``) or as a raw grid image (``interpol=False``).

    Parameters
    ----------
    obs : UVIS_Observation
        Observation object supplying geometry arrays and integrated radiance.
    wl_range : tuple of float, optional
        ``(lambda_min, lambda_max)`` wavelength interval in nm passed to
        :meth:`~cassini_upyp.uvisdata.UVIS_Observation.integrate_radiance`.
        If *None*, the full bandpass is integrated.
    alt_circles : sequence of float, optional
        Altitudes in km at which dashed reference circles are drawn around
        the apparent body disk.  Skipped when *None*.
    cmap : str, default ``"plasma"``
        Any Matplotlib-recognised colormap name.
    vmin : float, optional
        Lower bound of the colour scale.  Defaults to the minimum finite
        value of the projected image.
    vmax : float, optional
        Upper bound of the colour scale.  Defaults to the maximum finite
        value of the projected image.  When *vmax* is smaller than the
        actual maximum, the colorbar is extended with an arrow cap.
    levels : int, default 100
        Number of contour levels passed to
        :func:`~matplotlib.axes.Axes.tricontourf`.
        Ignored when ``interpol=False``.
    nx : int, default 20
        Number of grid columns along the x axis (cross-equatorial direction).
        Increase for finer spatial sampling at the cost of computation time.
    ny : int, default 20
        Number of grid rows along the y axis (latitudinal direction).
        Increase for finer spatial sampling at the cost of computation time.
    interpol : bool, default True
        Rendering mode.

        ``True``
            Use :func:`~matplotlib.axes.Axes.tricontourf` on valid grid
            points only; gaps between pixel footprints are filled by
            triangular interpolation.  May introduce artefacts near the
            edges of the field of view when the grid is coarse.

        ``False``
            Render the raw gridded image with
            :func:`~matplotlib.axes.Axes.imshow`; masked cells appear blank.

    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.  A new figure and axes are created when *None*.
    annotate : bool, default True
        If *True*, add a colorbar, a N/E compass, a pole marker, and axis
        labels.  If *False*, the axes are hidden and only the radiance map
        and the body outline are drawn (useful for publication figures).
    xlim : tuple of float, optional
        ``(x_min, x_max)`` extent in km of the projected grid along the
        x axis.  Derived from the pixel footprints when *None*.
    ylim : tuple of float, optional
        ``(y_min, y_max)`` extent in km of the projected grid along the
        y axis.  Derived from the pixel footprints when *None*.

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
    - Y axis is defined by the LOS tangent point local time.
    - Pixel index 59 is skipped (instrument-specific exclusion).
    - Grid cells covered by multiple pixel footprints are averaged.
    - The function should be used for observations where the spacecraft remains in a stable position relatively to the sun.
      Image construction may be inaccurate for observations with rapidly changing geometry.

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

    r_e, r_p = np.mean(radii[:2]), radii[2] # Equatorial and polar radii

    # Pixel coordinates
    from .computational import ellipsoid_radius
    lats = obs.pixel_LOS["t_lat"]
    lons = obs.pixel_LOS["t_lon"]
    t_lt = obs.pixel_LOS["t_lt"]
    lt   = obs.pixel_LOS["lt"]
    alts = obs.pixel_LOS["alt"] + ellipsoid_radius(radii, lons, lats)

    sol_lon = t_lt*360/24 # Angle to the sun

    xx = alts*np.cos(np.radians(lats))
    yy = alts*np.sin(np.radians(lats))

    

    # Sort pixels left/right by local time
    sc_lon = np.mean(obs.spacecraft_position['lt'])*360/24
    lon_1 = sc_lon-180
    if lon_1<0:
        mask = (
            ((sol_lon >= lon_1+360) & (sol_lon < 360))    |
            ((sol_lon >= 0)         & (sol_lon < sc_lon))
        )
        xx[mask] *= -1
    else:
        mask = ((sol_lon >= lon_1)  & (sol_lon < sc_lon))
        xx[mask] *= -1

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
    sc_lat = np.mean(obs.spacecraft_position['lat'])

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