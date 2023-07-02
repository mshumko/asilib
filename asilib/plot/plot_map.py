"""
This module contains functions to project the ASI images to a map.
"""
from typing import List, Union
import importlib
import pathlib
import zipfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import shapefile  # A pure python-library. Yay!

import asilib
from asilib.io import load


def plot_map(
    asi_array_code: str,
    location_code: str,
    time: asilib.io.utils._time_type,
    map_alt: float,
    time_thresh_s: float = 3,
    ax: plt.Axes = None,
    color_map: str = 'auto',
    min_elevation: float = 10,
    norm: bool = True,
    asi_label: bool = True,
    color_bounds: Union[List[float], None] = None,
    color_norm: str = 'log',
    pcolormesh_kwargs: dict = {},
    map_shapefile: Union[str, pathlib.Path] = 'ne_10m_land',
    coast_color: str = 'k',
    land_color: str = 'g',
    ocean_color: str = 'w',
    lon_bounds: tuple = (-140, -60),
    lat_bounds: tuple = (40, 82),
):
    """
    Projects the ASI images to a map at an altitude defined in the skymap calibration file.

    .. warning::
        Use :py:meth:`~asilib.imager.Imager.plot_map()` instead. This function will be
        removed in or after December 2023.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to download of the data. If str, ``time`` must be in the
        ISO 8601 standard.
    map_alt: float
        The altitude in kilometers to project to. Must be an altitude value
        in the skymap calibration.
    time_thresh_s: float
        The maximum allowable time difference between ``time`` and an ASI time stamp.
        This is relevant only when ``time`` is specified.
    ax: plt.Axes
        The subplot to plot the image on. If None, this function will
        create one.
    color_map: str
        The matplotlib colormap to use. If 'auto', will default to a
        black-red colormap for REGO and black-white colormap for THEMIS.
        For more information See https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    min_elevation: float
        Masks the pixels below min_elevation degrees.
    norm: bool
        If True, normalizes the image array to 0-1. This is useful when
        mapping images from multiple imagers.
    asi_label: bool
        Annotates the map with the ASI code in the center of the image.
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    pcolormesh_kwargs: dict
        A dictionary of keyword arguments (kwargs) to pass directly into
        plt.pcolormesh. One use of this parameter is to change the colormap. For example,
        pcolormesh_kwargs = {'cmap':'tu}
    map_shapefile: str or pathlib.Path
        The path to the shapefile zip archive. If str, it will try to load the
        shapefile in asilib/data/{file}.
    coast_color: str
        The coast color. If None will not draw it.
    land_color: str
        The land color. If None will not draw it.
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
    image_time: datetime.datetime
        The time of the current image.
    image: np.array
        The 2d ASI image corresponding to image_time.
    skyamp: dict
        The skymap calibration for that ASI.
    ax: plt.Axes
        The subplot object to modify the axis, labels, etc.
    p: plt.AxesImage
        The plt.pcolormesh image object. Common use for p is to add a colorbar.

    Example
    -------
    | from datetime import datetime
    |
    | import matplotlib.pyplot as plt
    |
    | import asilib
    |
    | asi_array_code = 'THEMIS'
    | location_code = 'ATHA'
    | time = datetime(2008, 3, 9, 9, 18, 0)
    | map_alt_km = 110
    | asilib.plot_map(asi_array_code, location_code, time, map_alt_km);
    | plt.show()
    """
    warnings.warn(
        "Use asilib.Imager.plot_map() instead. This function will be removed "
        "in or after December 2023.",
        DeprecationWarning,
    )

    image_time, image = load.load_image(
        asi_array_code, location_code, time=time, time_thresh_s=time_thresh_s
    )
    skymap = load.load_skymap(asi_array_code, location_code, time)

    # Check that the map_alt is in the skymap calibration data.
    assert (
        map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
    ), f'{map_alt} km is not in skymap calibration altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
    alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]

    image, lon_map, lat_map = _mask_low_horizon(
        image,
        skymap['FULL_MAP_LONGITUDE'][alt_index, :, :],
        skymap['FULL_MAP_LATITUDE'][alt_index, :, :],
        skymap['FULL_ELEVATION'],
        min_elevation,
    )

    if norm:
        image /= np.nanmax(image)

    # Set up the plot parameters
    if ax is None:
        ax = make_map(
            file=map_shapefile,
            coast_color=coast_color,
            land_color=land_color,
            ocean_color=ocean_color,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
        )

    if color_bounds is None:
        color_bounds = asilib.plot.utils.get_color_bounds(image)
    _color_map = asilib.plot.utils.get_color_map(asi_array_code, color_map)
    _norm = asilib.plot.utils.get_color_norm(color_norm, color_bounds)

    p = _pcolormesh_nan(
        lon_map,
        lat_map,
        image,
        ax,
        cmap=_color_map,
        norm=_norm,
        pcolormesh_kwargs=pcolormesh_kwargs,
    )

    if asi_label:
        ax.text(
            skymap['SITE_MAP_LONGITUDE'],
            skymap['SITE_MAP_LATITUDE'],
            location_code.upper(),
            color='r',
            va='center',
            ha='center',
        )
    return image_time, image, skymap, ax, p


def make_map(
    file: Union[str, pathlib.Path] = 'ne_10m_land',
    coast_color: str = 'k',
    land_color: str = 'g',
    ocean_color: str = 'w',
    ax: plt.Axes = None,
    lon_bounds: tuple = (-140, -60),
    lat_bounds: tuple = (40, 82),
) -> plt.Axes:
    """
    Makes a map using the mercator projection with a shapefile read in by the pyshp package.

    A good place to download shapefiles is
    https://www.naturalearthdata.com/downloads/10m-physical-vectors/.

    .. warning::
        Use :py:meth:`~asilib.map.create_map()` instead. This function will be
        removed in or after December 2023.

    Parameters
    ----------
    file: str or pathlib.Path
        The path to the shapefile zip archive. If str, it will try to load the
        shapefile in asilib/data/{file}.
    coast_color: str
        The coast color. If None will not draw it.
    land_color: str
        The land color. If None will not draw it.
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
    | import asilib
    |
    | ax = asilib.make_map(lon_bounds=(-127, -100), lat_bounds=(45, 65))
    """
    warnings.warn(
        "Use asilib.map.create_map() instead. This function will be removed "
        "in or after December 2023.",
        DeprecationWarning,
    )

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
    mlats = ma.masked_array(lats[:-1], mask=jumps)
    mlons = ma.masked_array(lons[:-1], mask=jumps)

    split_lats = _consecutive(lats, jumps)
    split_lons = _consecutive(lons, jumps)

    if ax is None:
        _, ax = plt.subplots()
    if ocean_color is not None:
        ax.set_facecolor(ocean_color)
        pass
    if coast_color is not None:
        ax.plot(mlats, mlons, coast_color)
    if land_color is not None:
        for split_lon, split_lat in zip(split_lons, split_lats):
            ax.fill(split_lat, split_lon, land_color, zorder=0)

    # ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)
    return ax


def _consecutive(data, jump_bool):
    """
    Calculate where the array jumps.

    Taken from: https://stackoverflow.com/questions/7352684/
    how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    """
    return np.split(data, np.where(jump_bool)[0] + 1)


def _pcolormesh_nan(
    x: np.ndarray, y: np.ndarray, c: np.ndarray, ax, cmap=None, norm=None, pcolormesh_kwargs={}
):
    """
    Since pcolormesh cant handle nan lat/lon grid values, we will compress them to the
    nearest valid lat/lon grid. There are two main steps:

    1) All nan values to the left of the first valid value are
    reassigned to the first valid value. Likewise, all nan values to the
    right of the last valid value are reassigned to it.

    2) All nan-filled rows above the first valid row are assigned to the
    maximum value in the first row, likewise for the bottom rows.

    Essentially this is a reassignment (or a compression) of all nan values in the periphery
    to the valid grid values in the center.

    Function taken from Michael, scivision @ GitHub.:
    https://github.com/scivision/python-matlab-examples/blob/0dd8129bda8f0ec2c46dae734d8e43628346388c/PlotPcolor/pcolormesh_NaN.py
    """
    # mask is True when lat and lon grid values are not nan.
    mask = np.isfinite(x) & np.isfinite(y)
    top = None
    bottom = None

    for i, m in enumerate(mask):
        # A common use for nonzero is to find the indices of
        # an array, where a condition is True (not nan or inf)
        good = m.nonzero()[0]

        if good.size == 0:  # Skip row is all columns are nans.
            continue
        # First row that has at least 1 valid value.
        elif top is None:
            top = i
        # Bottom row that has at least 1 value value. All indices in between top and bottom
        else:
            bottom = i

        # Reassign all lat/lon columns after good[-1] (all nans) to good[-1].
        x[i, good[-1] :] = x[i, good[-1]]
        y[i, good[-1] :] = y[i, good[-1]]
        # Reassign all lat/lon columns before good[0] (all nans) to good[0].
        x[i, : good[0]] = x[i, good[0]]
        y[i, : good[0]] = y[i, good[0]]

    # Reassign all of the fully invalid lat/lon rows above top to the the max lat/lon value.
    x[:top, :] = np.nanmax(x[top, :])
    y[:top, :] = np.nanmax(y[top, :])
    # Same, but for the rows below bottom.
    x[bottom:, :] = np.nanmax(x[bottom, :])
    y[bottom:, :] = np.nanmax(y[bottom, :])

    # old masked c code: np.ma.masked_where(~mask[:-1, :-1], c)[::-1, ::-1]
    p = ax.pcolormesh(
        x,
        y,
        c,
        cmap=cmap,
        shading='flat',
        norm=norm,
        **pcolormesh_kwargs,
    )
    return p


def _mask_low_horizon(image, lon_map, lat_map, el_map, min_elevation):
    """
    Mask the image, skymap['FULL_MAP_LONGITUDE'], skymap['FULL_MAP_LONGITUDE'] arrays
    with np.nans where the skymap['FULL_ELEVATION'] is nan or
    skymap['FULL_ELEVATION'] < min_elevation.
    """
    idh = np.where(np.isnan(el_map) | (el_map < min_elevation))
    # Copy variables to not modify original np.arrays.
    image_copy = image.copy()
    lon_map_copy = lon_map.copy()
    lat_map_copy = lat_map.copy()
    # Can't mask image unless it is a float array.
    image_copy = image_copy.astype(float)
    image_copy[idh] = np.nan
    lon_map_copy[idh] = np.nan
    lat_map_copy[idh] = np.nan

    # For some reason the lat/lon_map arrays are one size larger than el_map, so
    # here we mask the boundary indices in el_map by adding 1 to both the rows
    # and columns.
    idh_boundary_bottom = (idh[0] + 1, idh[1])  # idh is a tuple so we have to create a new one.
    idh_boundary_right = (idh[0], idh[1] + 1)
    lon_map_copy[idh_boundary_bottom] = np.nan
    lat_map_copy[idh_boundary_bottom] = np.nan
    lon_map_copy[idh_boundary_right] = np.nan
    lat_map_copy[idh_boundary_right] = np.nan
    return image_copy, lon_map_copy, lat_map_copy
