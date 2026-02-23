(geometry_geometry)=
# UVIS Geometry

The main geometry object is a high-level interface to the geometric computations and stores the relevant geometric information for a given observation epoch. It relies on a low-level geometric engine described in the next section and on a set of utilities based on the [NAIF SPICE](https://naif.jpl.nasa.gov/naif/) toolkit wrapped in Python by the [SpiceyPy](https://pypi.org/project/spiceypy/) library.
This class allows the observation to be plotted with the `.plot()` method defined in the `cassini_upyp.geometry.plotting` module, which provides a convenient interface to the plotting utilities described in the {ref}`plotting` section.

```{eval-rst}
.. autoclass:: cassini_upyp.geometry.geometry.Geometry
    :members:
    :member-order: bysource
    :show-inheritance:
    :special-members: init
```