"""
Plot geographic maps using cartopy or the simple built-in function. Before you
project an ASI image onto a map, you will need to create a map using the 
following functions.

The simplest way to create a map is via :py:meth:`~asilib.map.create_map` that
by default creates a map above North America. :py:meth:`~asilib.map.create_map` 
is a wrapper that automatically chooses what library to plot the map: cartopy 
if it is installed, or asilib's :py:meth:`~asilib.map.create_simple_map` 
function to create a simple map if cartopy is not installed. All of these functions
output the subplot object with the map.

You can override this automatic behavior by calling the underlying functions directly:
:py:meth:`~asilib.map.create_simple_map` or :py:meth:`~asilib.map.create_cartopy_map`.

The two functions are called similarly with the exception of the ``ax``  kwarg if you 
need to specify what subplot to create the map on:

- for :py:meth:`~asilib.map.create_simple_map` you must pass in a ``ax`` subplot object, and 
- for :py:meth:`~asilib.map.create_cartopy_map`, you need to pass in a ``fig_ax`` tuple containing two elements. The first element is the ```plt.Figure`` object, and the second element is a 3 digit number (or a tuple) specifying where to place that subplot. For example,

    .. code-block:: python

        >>> fig = plt.Figure()
        >>> ax = asilib.map.create_cartopy_map(fig_ax=(fig, 111))
"""

import pathlib
from typing import List, Union
import zipfile

import shapefile
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cartopy_imported = True
except ImportError as err:
    # You can also get a ModuleNotFoundError if cartopy is not installed
    # (as compared to failed to import), but it is a subclass of ImportError.
    cartopy_imported =  False

import asilib


def create_map(
    lon_bounds: tuple = (-160, -50),
    lat_bounds: tuple = (40, 82),
    ax: Union[plt.Axes, tuple] = None,
    coast_color: str = 'k',
    land_color: str = 'g',
    ocean_color: str = 'w',
    ):
    """
    Create a geographic map using cartopy (if installed) or asilib's own 
    map library.

    Parameters
    ----------
    lon_bounds: tuple
        The map's longitude bounds.
    lat_bounds: tuple
        The map's latitude bounds.
    ax: plt.Axes, tuple
        The subplot to put the map on. If cartopy is installed, ```ax``` must be
        a two element tuple specifying the ``plt.Figure`` object and subplot position
        passed directly as ``args`` into  
        `fig.add_subplot() <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.add_subplot>`_.
    coast_color: str
        The coast color. If None will not draw it.
    land_color: str
        The land color. If None will not draw it.
    ocean_color: str
        The ocean color. If None will not draw it.

    Returns
    -------
    plt.Axes
        The subplot object containing the map.

    Examples
    --------
    If you have cartopy installed see the examples in :py:meth:`~asilib.map.create_cartopy_map`.
    Alternatively, if you don't have cartopy installed see the examples in :py:meth:`~asilib.map.create_simple_map`.

    See Also
    --------
    :py:meth:`~asilib.map.create_simple_map`: 
        Create a simple map using asilib's own map library.
    :py:meth:`~asilib.map.create_cartopy_map`:
        Create a map using the cartopy library.
    """
    if cartopy_imported:
        if ax is not None:
            assert isinstance(ax, tuple), (
                f'The ax kwarg must be a tuple with (fig, position) values, '
                f'not {type(ax)}. See the docstring for more information.'
                )
        ax = create_cartopy_map(
            lon_bounds=lon_bounds, lat_bounds=lat_bounds, fig_ax=ax, 
            coast_color=coast_color, land_color=land_color, 
            ocean_color=ocean_color
        )
    else:
        if ax is not None:
            assert isinstance(ax, plt.Axes), f'The ax kwarg must be a subplot, not {type(ax)}.'
        ax = create_simple_map(
            lon_bounds=lon_bounds, lat_bounds=lat_bounds, ax=ax, 
            coast_color=coast_color, land_color=land_color, 
            ocean_color=ocean_color
            )
    return ax

def create_simple_map(
    lon_bounds: tuple = (-140, -60),
    lat_bounds: tuple = (40, 82),
    ax: plt.Axes = None,
    coast_color: str = 'k',
    land_color: str = 'g',
    ocean_color: str = 'w',
    file: Union[str, pathlib.Path] = 'ne_10m_land',
) -> plt.Axes:
    """
    Create a simple map without cartopy.

    Parameters
    ----------
    lon_bounds: tuple
        The map's longitude bounds.
    lat_bounds: tuple
        The map's latitude bounds.
    ax: plt.Axes
        The subplot to put the map on.
    coast_color: str
        The coast color. If None will not draw it.
    land_color: str
        The land color. If None will not draw it.
    ocean_color: str
        The ocean color. If None will not draw it.
    file: str or pathlib.Path
        The path to the shapefile zip archive. If str, it will try to load the
        shapefile in asilib/data/{file}. You can download other shapefiles from
        https://www.naturalearthdata.com/downloads/10m-physical-vectors/.

    Returns
    -------
    plt.Axes
        The subplot object containing the map.

    Examples
    --------
    >>> import asilib.map
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # Create a simple map above Scandinavia in a single plot

    >>> ax = asilib.map.create_simple_map(lon_bounds=[0, 38], lat_bounds=[50, 75])
    >>> ax.set_title('Generated via asilib.map.create_simple_map()')

    # The above examples made a map on one subplot. But what if you have multiple 
    # subplots?

    >>> fig = plt.figure(figsize=(6, 10))
    >>> bx = fig.add_subplot(2,1,1)
    >>> bx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], ax=bx)
    >>> cx = fig.add_subplot(2,1,2)
    >>> cx.plot(np.arange(10), np.random.rand(10))
    >>> fig.suptitle('Two subplots with equal aspect ratios')

    # Another multi-subplot example with different height ratios. The syntax is the same as
    # in plt.subplot() (See the args section in 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html).

    >>> fig = plt.figure(figsize=(6, 10))
    >>> dx = fig.add_subplot(3,1,(1, 2))
    >>> dx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], ax=dx)
    >>> ex = fig.add_subplot(3,1,3)
    >>> ex.plot(np.arange(10), np.random.rand(10))
    >>> fig.suptitle('Two subplots with differing aspect ratios')
    >>> plt.show()
    """
    shp_path = asilib.config['ASILIB_DIR'] / 'data' / f'{file}'

    with zipfile.ZipFile(shp_path.with_suffix('.zip'), 'r') as archive:
        shp = archive.open(f'{file}.shp', "r")
        dbf = archive.open(f'{file}.dbf', "r")
        sf = shapefile.Reader(shp=shp, dbf=dbf)
        i = 0  # I'm unsure what the other shapes are, but i=0 works.
        lats = np.array([point[0] for point in sf.shapes()[i].points])
        lons = np.array([point[1] for point in sf.shapes()[i].points])

    # Since the landmass shapes are represented continuously in (lats, lons),
    # matplotlib draws straight (annoying) lines between them. This code uses
    # the jumps bool array and masked_arrays to remove those lines.
    jumps = (np.abs(lats[1:] - lats[:-1]) > 3) | (np.abs(lons[1:] - lons[:-1]) > 3)
    mlats = numpy.ma.masked_array(lats[:-1], mask=jumps)
    mlons = numpy.ma.masked_array(lons[:-1], mask=jumps)

    split_lats = _consecutive(lats, jumps)
    split_lons = _consecutive(lons, jumps)

    if ax is None:
        _, ax = plt.subplots()

    if ocean_color is not None:
        ax.set_facecolor(ocean_color)
    if coast_color is not None:
        ax.plot(mlats, mlons, coast_color)
    if land_color is not None:
        for split_lon, split_lat in zip(split_lons, split_lats):
            ax.fill(split_lat, split_lon, land_color, zorder=0)

    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)
    return ax

def create_cartopy_map(
    lon_bounds: tuple = (-160, -50),
    lat_bounds: tuple = (40, 82),
    fig_ax: tuple = None,
    coast_color: str = 'k',
    land_color: str = 'g',
    ocean_color: str = 'w',
) -> plt.Axes:
    """
    A helper function to create two map styles: a simple black and white map, and
    a more sophisticated map with green land.

    Parameters
    ----------
    lon_bounds: tuple or list
        A tuple of length 2 specifying the map's longitude bounds.
    lat_bounds: tuple or list
        A tuple of length 2 specifying the map's latitude bounds.
    fig_ax: tuple
        A two element tuple specifying the ``plt.Figure`` object and subplot position
        passed directly as ``args`` into  
        `fig.add_subplot() <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.add_subplot>`_.
        For example:

        .. code-block:: python

            fig = plt.Figure()
            ax = asilib.map.create_cartopy_map(fig_ax=(fig, 111))
        
    coast_color: str
        The coast color. If None will not draw it.
    land_color: str
        The land color. If None will not draw it.
    ocean_color: str
        The ocean color. If None will not draw it.

    Returns
    -------
    ax: plt.Axes
        The subplot object with a cartopy map.
    
    Examples
    --------
    >>> import asilib.map
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # Create a cartopy map above Scandinavia in a single plot

    >>> ax = asilib.map.create_cartopy_map(lon_bounds=[0, 38], lat_bounds=[50, 75])
    >>> ax.set_title('Generated via asilib.map.create_cartopy_map()')
    
    # If you have cartopy installed, the following example creates the same plot.

    >>> bx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75])
    >>> bx.set_title('Generated via asilib.map.create_map()')

    # The above examples made a map on one subplot. But what if you have multiple subplots?

    >>> fig = plt.figure(figsize=(6, 10))
    >>> cx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], ax=(fig, (2,1,1)))
    >>> dx = fig.add_subplot(2,1,2)
    >>> dx.plot(np.arange(10), np.random.rand(10))
    >>> fig.suptitle('Two subplots with equal aspect ratios')

    # Another multi-subplot example with different height ratios. The syntax is the same as
    # in plt.subplot() (See the args section in 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html).

    >>> fig = plt.figure(figsize=(6, 10))
    >>> cx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], ax=(fig, (3,1,(1,2))))
    >>> dx = fig.add_subplot(3,1,3)
    >>> dx.plot(np.arange(10), np.random.rand(10))
    >>> fig.suptitle('Two subplots with differing aspect ratios')
    >>> plt.show()
    """
    plot_extent = [*lon_bounds, *lat_bounds]
    central_lon = np.mean(lon_bounds)
    central_lat = np.mean(lat_bounds)
    projection = ccrs.Orthographic(central_lon, central_lat)

    if fig_ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection=projection)
    else:
        fig = fig_ax[0]
        ax = fig.add_subplot(*fig_ax[1], projection=projection)

    if land_color is not None:
        ax.add_feature(cfeature.LAND, color=land_color)
    if ocean_color is not None:
        ax.add_feature(cfeature.OCEAN, color=ocean_color)
    if coast_color is not None:
        ax.add_feature(cfeature.COASTLINE, edgecolor=coast_color)
    ax.gridlines(linestyle=':')
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
    return ax


def _consecutive(data, jump_bool):
    """
    Calculate where the array jumps.

    Taken from: https://stackoverflow.com/questions/7352684/
    how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    """
    return np.split(data, np.where(jump_bool)[0] + 1)

if __name__ == '__main__':
    import asilib.map
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a simple map above Scandinavia in a single plot

    ax = asilib.map.create_simple_map(lon_bounds=[0, 38], lat_bounds=[50, 75])
    ax.set_title('Generated via asilib.map.create_simple_map()')

    # The above examples made a map on one subplot. But what if you have multiple subplots?

    fig = plt.figure(figsize=(6, 10))
    bx = fig.add_subplot(2,1,1)
    bx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], ax=bx)
    cx = fig.add_subplot(2,1,2)
    cx.plot(np.arange(10), np.random.rand(10))
    fig.suptitle('Two subplots with equal aspect ratios')

    # Another multi-subplot example with different height ratios. The syntax is the same as
    # in plt.subplot() (See the args section in 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html).

    fig = plt.figure(figsize=(6, 10))
    dx = fig.add_subplot(3,1,(1, 2))
    dx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], ax=dx)
    ex = fig.add_subplot(3,1,3)
    ex.plot(np.arange(10), np.random.rand(10))
    fig.suptitle('Two subplots with differing aspect ratios')

    plt.show()