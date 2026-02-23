# Geometry

Geometric computations are performed by low-level geometric engine described below and by a set of utilities based on the [NAIF SPICE](https://naif.jpl.nasa.gov/naif/) toolkit wrapped in Python by the [SpiceyPy](https://pypi.org/project/spiceypy/) library. The main geometry object is the {ref}`geometry_geometry` class, which provides a high-level interface to the geometric computations and stores the relevant geometric information for a given observation epoch.

## Main geometry object
```{toctree}
:maxdepth: 1
geometry_geometry
```

## Geometry engine

```{toctree}
:maxdepth: 1
geometry_spice_engine
```

## Computational utilities

```{eval-rst}
.. autofunction:: cassini_upyp.geometry.computational.xyz2radec

.. autofunction:: cassini_upyp.geometry.computational.radec2xyz

.. autofunction:: cassini_upyp.geometry.computational.vec_angle

.. autofunction:: cassini_upyp.geometry.computational.max_angular_diameter

.. autofunction:: cassini_upyp.geometry.computational.rotation_matrix

.. autofunction:: cassini_upyp.geometry.computational.is_in_frame

.. autofunction:: cassini_upyp.geometry.computational.is_vector_in_quadrilateral

.. autofunction:: cassini_upyp.geometry.computational.is_visible

.. autofunction:: cassini_upyp.geometry.computational.order_bool

.. autofunction:: cassini_upyp.geometry.computational.intersect

.. autofunction:: cassini_upyp.geometry.computational.ellipsoid_coords

.. autofunction:: cassini_upyp.geometry.computational.ellipsoid_xyz
```

(plotting)=
## Plotting
```{eval-rst}
.. autofunction:: cassini_upyp.geometry.plot.plot
```

## Miscellaneous
See the [resources page](../resources.md).
```{eval-rst}
.. autofunction:: cassini_upyp.geometry.geometry.stars_pickles
```

