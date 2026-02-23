from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .computational import is_in_frame
from ..utils import plot_config
plotconfig = plot_config()


if TYPE_CHECKING:
    from .geometry import geometry

# Plot stars
def plot_stars(stars_pickles, ax=None):
    import matplotlib.pyplot as plt
    if ax is None : ax=plt.gca()

    ax.plot(stars_pickles()['tyRA'],stars_pickles()['tyDE'], **plotconfig.STAR_STYLE)


# Pixels FOV
def plot_pixels(pixels, ax=None, **kwargs) :
    import matplotlib.pyplot as plt
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


# MAIN PLOT ROUTINE
def plot(
    g_obj: geometry,
    mode: Literal['target', 'FOV', 'allsky', 'manual'] = 'target',
    orf_center: tuple[float, float] | None = None,

    *,
    save: bool = False,
    savename: str | Path | None = None,
    show: bool = True,

    scale: float = 0.6,
    RA_range  : tuple[float, float] | None = None,
    DEC_range : tuple[float, float] | None = None,

    ax: "plt.Axes" | None = None,
    pixel_notes: Literal['lon', 'lat', 'alt', 'sza', 'phase', 'ems', "lt"] | None = None,
    pixel_numbers: bool = True,
    date:bool = False,
    dpi: float = 1900 / 5,
) -> "plt.Axes":

    """
    Plot a precomputed UVIS geometry scene.

    This function draws the target limb/disk (and, when available, night side and terminator)
    in the requested coordinate frame. For the main target, it additionally overlays
    context elements (RA/DEC grid lines, background stars, lat/lon grid, UVIS pixels) and
    optionally other targets present in the field of view.

    The function supports two typical use-cases:

    - **Top-level plot** (``ax`` is None, ``g_obj.main`` is True):
      a new figure/axes is created and styled; axis limits
      are set automatically from ``mode`` and ``scale`` (unless user ranges are provided).
    - **Overlay plot** (``ax`` is provided): the scene is drawn onto an existing axes without
      restyling or changing axis limits (unless your implementation explicitly does so).

    Parameters
    ----------
    g_obj : geometry
        Geometry instance to plot. Must expose precomputed attributes used by this function
        (e.g., ``target_limb``, ``night_side``, ``terminator``, ``pixels``, ``used_pixels``,
        ``target_center``, and when ``main=True``: ``stars_orf``/``stars``, ``radec_lines``,
        ``lon_lines``/``lat_lines``, ``other_targets``).
    mode : {'target', 'FOV', 'allsky', 'manual'}, default='target'
        Plotting mode controlling the view center and the default axis ranges:

        - ``'target'``: centers on the target disk and uses a range derived from the limb.
        - ``'FOV'``: centers on the UVIS FOV and uses a range derived from the FOV size.
        - ``'allsky'``: uses full-sky RA/DEC limits (0..360, -90..90).
        - ``'manual'``: uses ``orf_center`` and user-provided ranges.
    orf_center : tuple[float, float] or None, default=None
        Object Reference Frame (ORF) center (RA, DEC) used in ``'manual'`` mode (and internally computed in other modes).
        Interpreted in the RA/DEC frame used by ``g_obj.rotate``.
    save : bool, default=False
        If True, save the figure to ``savename`` (only meaningful for a top-level plot).
    savename : str or pathlib.Path or os.PathLike or None, default=None
        Output path for saving. Must be provided (or set by the caller) when ``save=True``.
    show : bool, default=True
        Whether to show the figure.
    scale : float, default=0.6
        Scale factor applied when computing automatic ranges in ``'FOV'`` mode (and possibly
        other modes depending on the implementation).
    RA_range : tuple[float, float] or None, default=None
        X-axis limits in the active plotting frame. For modes ``'target'``, ``'FOV'`` and
        ``'manual'`` the active frame is typically ORF; for ``'allsky'`` it is RA/DEC.
        If provided, ``DEC_range`` should also be provided.
    DEC_range : tuple[float, float] or None, default=None
        Y-axis limits in the active plotting frame. If provided, ``RA_range`` should also be
        provided.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on. If None, a new figure/axes is created.
    pixel_notes : {'lon', 'lat', 'alt', 'sza', 'phase', 'ems', 'lt'} or None, default=None
        If provided, annotate each used pixel with the corresponding LOS parameter field.
    pixel_numbers : bool, default=True
        If True, annotate first and last pixel with its pixel number to help identify the UVIS slit orientation.
    dpi : float or int, default=1900/5
        DPI used when saving the figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the scene was drawn onto.

    Raises
    ------
    ValueError
        If ``mode='manual'`` and required parameters are missing, or if only one of
        ``RA_range``/``DEC_range`` is provided, or if ``pixel_notes`` is not a valid field name.
    """

    
    frame = 'ORF' if mode in ['target', 'FOV', 'manual'] else 'RADEC'
    
    # SETUP AXES ---------------------------------------
    if (ax is None) or g_obj.main :
        
        match mode:
            case 'target' :
                orf_center = (g_obj.target_center['RADEC'][0] , g_obj.target_center['RADEC'][1])
                g_obj.rotate(view_center=orf_center)
                
                if RA_range is None and DEC_range is None  :
                    # Fix target disk in frame
                    RA_range  = min(g_obj.target_limb['ORF'][:,0])*2, max(g_obj.target_limb['ORF'][:,0])*2
                    DEC_range = min(g_obj.target_limb['ORF'][:,1])*2, max(g_obj.target_limb['ORF'][:,1])*2
                elif (RA_range is None) or (DEC_range is None):
                    raise ValueError("If you pass RA_range or DEC_range, you must pass both.")
            case 'FOV' :
                pixel_center = g_obj.pixels['RADEC'][31]
                FOV_center   = (pixel_center[-2,:]+pixel_center[-1,:])/2
                orf_center   = (FOV_center[0] , FOV_center[1])
                g_obj.rotate(view_center=orf_center)

                if RA_range is None and DEC_range is None  :
                    # Fix Cassini FOV in frame
                    FOV_size  = abs(np.linalg.norm(g_obj.pixels['ORF'][0,1,:] - g_obj.pixels['ORF'][-1,-1,:]))
                    RA_range  = -FOV_size*scale, FOV_size*scale
                    DEC_range = -FOV_size*scale, FOV_size*scale
                elif (RA_range is None) or (DEC_range is None):
                    raise ValueError("If you pass RA_range or DEC_range, you must pass both.")
            
            case 'allsky' :
                RA_range, DEC_range = (0,360), (-90,90)
            case 'manual' :
                if orf_center is None : raise ValueError('Manual mode requires a (RA/DEC) central position')
                if RA_range   is None : raise ValueError('Manual mode requires a valid RA range to plot')
                if DEC_range  is None : raise ValueError('Manual mode requires a valid RA range to plot')
                g_obj.rotate(view_center=orf_center)
    
    ax_created = False
    if ax is None :
        ax_created = True
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_facecolor(plotconfig.BACKGROUND_COLOR)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig = ax.figure
    
    if RA_range is not None and DEC_range is not None:
        xmin, xmax = RA_range
        ymin, ymax = DEC_range
    else:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

    if ax_created:
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        if frame in ("RADEC", "ORF"):
            ax.invert_xaxis()
    # ------------------------------------


    if g_obj.target in plotconfig.PLANET_STYLE :
        planet_style = plotconfig.PLANET_STYLE[g_obj.target]
    else :
        planet_style = plotconfig.PLANET_STYLE['DEFAULT']

    # Target limb + day side
    ax.fill(
        g_obj.target_limb[frame][:,0],
        g_obj.target_limb[frame][:,1],
        **planet_style['limb'],
        zorder=0+g_obj.zorder
    )

    if g_obj.target != 'SUN' :
        # Night side
        ax.fill(
            g_obj.night_side[frame][:,0],
            g_obj.night_side[frame][:,1],
            **planet_style['night_side'],
            zorder = 1+g_obj.zorder
        )

        # Terminator line
        ax.plot(
            g_obj.terminator[frame][:,0],
            g_obj.terminator[frame][:,1],
            **planet_style['terminator'],
            zorder = 2+g_obj.zorder
        )
    
    # Main target section
    if g_obj.main :
        
        if frame=='ORF' :
            # RA / DEC background lines
            ax.plot(g_obj.radec_lines[frame][:, 0], g_obj.radec_lines[frame][:, 1],
                    **plotconfig.RADEC_LINES,zorder=-10000000001)


            # Background stars
            stars_orf = np.array(g_obj.stars_orf)
            stars_orf = stars_orf[is_in_frame(stars_orf, (xmin, xmax), (ymin, ymax))]
            ax.plot(stars_orf[:,0],stars_orf[:,1], **plotconfig.STAR_STYLE, zorder=-10000000000)

        else :
            ax.plot(g_obj.stars['RA_cor'],g_obj.stars['DEC_cor'], **plotconfig.STAR_STYLE, zorder=-10000000000)

        # OTHER BODIES IN THE SKY
        for t2 in g_obj.other_targets:

            if mode!='allsky' : t2.rotate(view_center=g_obj.orf_center, units = g_obj.rotate_units)
            
            if np.any(is_in_frame(t2.target_limb[frame], (xmin, xmax), (ymin, ymax))) :
                ax.annotate( t2.target,
                        (t2.target_center[frame][0], t2.target_center[frame][1]),
                        color='white', textcoords="offset points", xytext=(5, 5), ha='center', fontsize=10, zorder=t2.zorder, clip_on=True)
                
                if t2.angular_diameter < np.radians(1) :
                    if t2.target in plotconfig.PLANET_STYLE :
                        ax.plot([t2.target_center[frame][0]], [t2.target_center[frame][1]],
                                ls='', marker='o', ms = 5, color=plotconfig.PLANET_STYLE[t2.target]['limb']['color'], zorder=t2.zorder)
                    else :
                        ax.plot([t2.target_center[frame][0]], [t2.target_center[frame][1]],
                                ls='', marker='o', ms = 5, color=plotconfig.PLANET_STYLE['DEFAULT']['limb']['color'], zorder=t2.zorder)

                    continue
                else : t2.plot(mode=mode, ax=ax, show=False, save=False)


        # LONGITUDE AND LATITUDE LINES
        for lon_line in g_obj.lon_lines :
            ax.plot(
                lon_line[frame][:,0],
                lon_line[frame][:,1],
                **plotconfig.LATLON_GRID,
            )


        for lat_line in g_obj.lat_lines :
            ax.plot(
                lat_line[frame][:,0],
                lat_line[frame][:,1],
                **plotconfig.LATLON_GRID,
            )

        # PIXELS
        # Total pixels
        plot_pixels(g_obj.pixels[frame],
                    ax=ax, linewidth=1, color='lightgray', ls='-', marker='', zorder = 19)
        
        plot_pixels(g_obj.used_pixels[frame],
                    ax=ax, linewidth=2, color='red', ls='-', marker='', zorder = 20)

        # TARGET CENTER
        ax.plot([g_obj.target_center[frame][0]], [g_obj.target_center[frame][1]], **plotconfig.TARGET_CENTER)

        # SUB-SPACECRAFT LATITUDE AND LONGITUDE
        ax.annotate(f"{round(g_obj.sub_sc_lon)}° , {round(g_obj.sub_sc_lat)}°",
                    (g_obj.target_center[frame][0], g_obj.target_center[frame][1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

        # PIXEL PARAMETERS (LOS PROPERTIES)
        if pixel_numbers:
            # First pixel
            xi, yi = sum(g_obj.used_pixels[frame][0,1:3,0])/2, sum(g_obj.used_pixels[frame][0,1:3,1])/2
            ax.text(
                xi, yi, "0",
                color='white', fontsize=8,
                ha='center', va='center',
                clip_on=True, zorder=21
                )
            # Last pixel
            xi, yi = sum(g_obj.used_pixels[frame][-1,1:3,0])/2, sum(g_obj.used_pixels[frame][-1,1:3,1])/2
            ax.text(
                xi, yi, f"{len(g_obj.used_pixels[frame])-1}",
                color='white', fontsize=8,
                ha='center', va='center',
                clip_on=True, zorder=21
                )

        if pixel_notes:
            # Error handling
            names = g_obj.used_pixels_LOS.dtype.names
            if (names is None) or (pixel_notes not in names):
                raise ValueError(
                    f"Unknown pixel_notes field {pixel_notes!r}. "
                    f"Available fields: {list(names) if names else 'None'}"
                )
            
            # Display annotations
            for xi, yi, value in zip(g_obj.used_pixels[frame][:,0,0], g_obj.used_pixels[frame][:,0,1], g_obj.used_pixels_LOS[:,0][pixel_notes]):
                annotation = f"{value:.1f}"
                ax.text(
                    xi, yi, annotation,
                    color='white', fontsize=8,
                    ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'),
                    clip_on=True
                    )

        if date:
            ax.annotate(
                f"Date (UTC): {g_obj.UTC_time}",
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom',
                color='black', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.3, boxstyle='round,pad=0.2'),
                zorder=100000
            )
        if show :
            plt.show()
        if save :
            fig.tight_layout()
            if savename is None :
                savename = f"geometry_{g_obj.target}_{mode}.png"
            fig.savefig(savename, dpi=dpi)
            if ax_created: plt.close(fig)
    return ax
    