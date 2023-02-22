"""
Create a geographic map using the simple built-in function or cartopy.
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


def map(
    lon_bounds: tuple = (-160, -50),
    lat_bounds: tuple = (40, 82),
    ax: Union[plt.Axes, dict] = None,
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
    ax: plt.Axes, dict
        The subplot to put the map on. If cartopy is installed, ```ax``` must be
        a dictionary specifying the figure object and subplot position
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

    Example
    -------

    See Also
    --------
    asilib.make_simple_map: 
        Make a simple map using asilib's own map library.
    asilib.make_cartopy_map:
        Make a map using the cartopy library.
    """
    if cartopy_imported:
        # use cartopy
        pass
    else:
        # asilib engine
        pass
    return

def simple_map(
    lon_bounds: tuple = (-140, -60),
    lat_bounds: tuple = (40, 82),
    ax: plt.Axes = None,
    coast_color: str = 'k',
    land_color: str = 'g',
    ocean_color: str = 'w',
    file: Union[str, pathlib.Path] = 'ne_10m_land',
) -> plt.Axes:
    """
    Makes a simple map using the mercator projection with a shapefile read in by the pyshp package.

    A good place to download shapefiles is
    https://www.naturalearthdata.com/downloads/10m-physical-vectors/.

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
        shapefile in asilib/data/{file}.

    Returns
    -------
    plt.Axes
        The subplot object containing the map.

    Example
    -------
    | import asilib
    |
    | ax = asilib.make_map(lon_bounds=(-127, -100), lat_bounds=(45, 65))
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

def cartopy_map(
    lon_bounds: tuple = (-160, -50),
    lat_bounds: tuple = (40, 82),
    fig_ax: dict = None,
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
    fig_ax: dict
        Make a map on an existing figure. The dictionary key:values must be
        'fig': figure object, and 'ax': the subplot position in the
        (nrows, ncols, index) format, or a GridSpec object. For example
        fig_ax = {}
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
    
    Example
    -------
    TODO: Complete example.
    import asilib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = asilib.cartopy_map(fig_ax={fig:})

    Raises
    ------
    ValueError:
        When a map_style other than 'green' or 'white' is chosen.
    """
    plot_extent = [*lon_bounds, *lat_bounds]
    central_lon = np.mean(lon_bounds)
    central_lat = np.mean(lat_bounds)
    projection = ccrs.Orthographic(central_lon, central_lat)

    if fig_ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
    # else:
    #     if hasattr(fig_ax['ax'], '__len__') and len(fig_ax['ax']) == 3:
    #         # If fig_ax['ax'] is in the format (X,Y,Z)
    #         ax = fig_ax['fig'].add_subplot(*fig_ax['ax'], projection=projection)
    #     else:
    #         # If fig_ax['ax'] is in the format XYZ or a gridspec object.
    #         ax = fig_ax['fig'].add_subplot(fig_ax['ax'], projection=projection)

    if land_color is not None:
        ax.add_feature(cfeature.LAND, color=land_color)
    if ocean_color is not None:
        ax.add_feature(cfeature.OCEAN, color=ocean_color)
    if coast_color is not None:
        ax.add_feature(cfeature.COASTLINE, color=coast_color)
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
    import matplotlib.pyplot as plt

    # Test asilib's maps
    ax = asilib.map.map()
    plt.show()

    
    