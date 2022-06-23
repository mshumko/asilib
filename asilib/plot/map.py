import pathlib
import zipfile
from typing import Union

import numpy as np
import numpy.ma as ma
import shapefile  # A pure python-library. Yay!
import matplotlib.pyplot as plt


def make_map(
    file: Union[str, pathlib.Path]='ne_10m_land', 
    coast_color: str='k', 
    landmass_color: str='g', 
    ocean_color: str='w', 
    ax: plt.Axes=None,
    lon_bounds: tuple = (-160, -50),
    lat_bounds: tuple = (40, 82),
    ) -> plt.Axes:
    """
    Makes a map using the mercator projection with a shapefile read in by the pyshp package. A good place
    to download shapefiles is https://www.naturalearthdata.com/downloads/10m-physical-vectors/.

    Parameters
    ----------
    file: str or pathlib.Path
        The path to the shapefile zip archive. If str, it will try to load the
        shapefile in asilib/data/{file}.
    coast_color: str
        The coast color. If None will not draw it.
    landmass_color: str
        The landmass color. If None will not draw it.
    ocean_color: str
        The ocean color. If None will not draw it.
    ax: plt.Axes
        The subplot to put the map on.
    lon_bounds: tuple
        The map's longitude bounds.
    lat_bounds: tuple
        The map's latitude bounds.

    Returns
    -------
    plt.Axes
        The subplot object containing the map.

    Example
    -------
    """
    shp_path = pathlib.Path(__file__).parent / f'{file}'

    with zipfile.ZipFile(shp_path.with_suffix('.zip'), 'r') as archive:
        shp = archive.open(f'{file}.shp', "r")
        dbf = archive.open(f'{file}.dbf', "r")
        sf = shapefile.Reader(shp=shp, dbf=dbf)
        i=0  # I'm unsure what the other shapes are, but i=0 works.
        lats = np.array([point[0] for point in sf.shapes()[i].points])
        lons = np.array([point[1] for point in sf.shapes()[i].points])

    # Since the landmass shapes are represented continuously in (lats, lons), 
    # matplotlib draws straight (annoying) lines between them. This code uses 
    # the jumps bool array and masked_arrays to remove those lines.
    jumps = (
        (np.abs(lats[1:]-lats[:-1]) > 3) | 
        (np.abs(lons[1:]-lons[:-1]) > 3)
        )
    mlats = ma.masked_array(lats[:-1], mask=jumps)
    mlons = ma.masked_array(lons[:-1], mask=jumps)

    split_lats = _consecutive(lats, jumps)
    split_lons = _consecutive(lons, jumps)

    if ax is None:
        _, ax = plt.subplots()
    if ocean_color is not None:
        ax.set_facecolor(ocean_color)
    if coast_color is not None:
        ax.plot(mlats, mlons, coast_color)
    if landmass_color is not None:
        for split_lon, split_lat in zip(split_lons, split_lats):
            ax.fill(split_lat, split_lon, landmass_color)

    ax.set_aspect('equal', adjustable='box')
    return ax

def _consecutive(data, jump_bool):
    """
    Calculate where the array jumps.

    Taken from: https://stackoverflow.com/questions/7352684/
    how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    """
    return np.split(data, np.where(jump_bool)[0]+1)