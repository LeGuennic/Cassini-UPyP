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
    Build and plot a projected UV radiance image in the plane of sky.

    For each exposure, pixel unit vectors (J2000) are projected onto a tangent
    plane centred on the target body.  The plane axes are defined geometrically:
    the y axis is the projection of the north-pole direction onto the image
    plane (computed via SPICE body-fixed → J2000 rotation), and the x axis is
    the cross product of y with the observer–target unit vector, then negated so
    that right ascension increases to the left (standard sky orientation).
    Frame-to-frame sign continuity of the y axis is enforced to prevent axis
    flips across exposures.

    Projected pixel footprints are rasterised onto a regular grid; cells covered
    by multiple footprints are averaged, cells with no coverage are masked.  The
    result is rendered as a contour fill (``interpol=True``) or as a raw grid
    image (``interpol=False``).

    Parameters
    ----------
    obs : UVIS_Observation
        Observation object supplying geometry arrays and integrated radiance.
        Must expose ``used_pixels['XYZ']``, ``target_vector``, ``ET_middle``,
        and ``spacecraft_position``.
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
        Number of grid columns along the x axis (plane-of-sky horizontal,
        east–west direction).
    ny : int, default 20
        Number of grid rows along the y axis (plane-of-sky vertical,
        north–south direction).
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
        If no valid projected radiance values are available after gridding.

    Notes
    -----
    - Pixel coordinates are computed by projecting J2000 unit vectors onto the
      tangent plane at distance *D* (observer–target distance) using a
      perspective divide: ``x_proj = D * dot(v, x̂) / dot(v, ĉ)``, where ``ĉ``
      is the observer–target unit vector.  Rays with ``dot(v, ĉ) ≤ 0`` (pointing
      away from the target plane) are rejected and set to NaN.
    - The image axes are aligned with the body's rotational pole: y points toward
      the north pole projected on the sky, x points east (right ascension
      decreases to the right).
    - The body disk is drawn as an apparent ellipse derived from SPICE body radii
      and the mean sub-spacecraft latitude.
    - Pixel 59 is unconditionally excluded (known instrumental artefact).
    - Grid cells covered by multiple pixel footprints are averaged (uniform
      weighting).
    - Best suited for observations where the spacecraft geometry does not change
      rapidly; the projection is computed independently per frame and averaged on
      the grid, which may introduce blurring for long or fast-moving observations.

    See Also
    --------
    :func:`cassini_upyp.uvisdata.UVIS_Observation.check_stars`
    """
    # 1 - PREPARE DATA
    #-----------------
    et_range = obs.ET_middle


    # Get body radius
    from ..utils import env_config
    env = env_config()
    pck = Path(env.pck_path) # Planetary constants kernel

    spice.furnsh(str(pck))
    radii = spice.bodvrd(obs.target, 'RADII', 3)[1]
    pole_N = []
    for et in et_range:
        rotation = spice.pxform('IAU_'+obs.target.upper(), 'J2000', et)
        pole_N.append(rotation[:, 2])
    spice.unload(str(pck))
    pole_N = np.array(pole_N)

    r_e, r_p = np.mean(radii[:2]), radii[2] # Equatorial and polar radii

    # 1.5 - PIXEL COORDINATES
    #------------------------
    n_frames = obs.target_vector.shape[0]
    xx = np.full(obs.used_pixels['XYZ'].shape[:-1], np.nan, dtype=float)
    yy = np.full(obs.used_pixels['XYZ'].shape[:-1], np.nan, dtype=float)

    prev_y = None
    for i in range(n_frames):
        D = np.linalg.norm(obs.target_vector[i])   # Distance to center
        c = obs.target_vector[i]/D                 # Observer - target unit vector
        n = pole_N[i] / np.linalg.norm(pole_N[i])  # North pole unit vector

        # North projected onto the image plane
        y = n - np.dot(n, c) * c
        y /= np.linalg.norm(y)

        # Keep a stable sign from one frame to the next
        if prev_y is not None and np.dot(y, prev_y) < 0:
            y *= -1.0
        prev_y = y.copy()

        x = np.cross(y, c)
        x /= np.linalg.norm(x)

        
        # Projection onto the plane at distance D
        v = obs.used_pixels['XYZ'][i]                   # Pixel vectors
        v /= np.linalg.norm(v, axis=-1, keepdims=True)

        cos = np.dot(v, c)
        xx_i = D * np.dot(v, x) / cos
        yy_i = D * np.dot(v, y) / cos

        # Reject rays pointing away from the target plane
        xx_i[cos <= 0] = np.nan
        yy_i[cos <= 0] = np.nan

        xx[i] = xx_i
        yy[i] = yy_i

    # Invert right ascension to have a real picture
    xx=-xx


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



    # 4 - MISCELLANEOUS ELEMENTS
    #---------------------------
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