"""
This module contains functions to project the ASI images to a map.
"""
from typing import List, Union, Optional, Sequence, Tuple
from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs

from asilib.io.load import load_skymap, get_frame


def plot_map(time: Union[datetime, str], mission: str,
    station: str, map_alt: int, time_thresh_s: float = 3,
    ax: plt.subplot = None, color_map: str = 'auto',
    min_elevation=10,
    color_bounds: Union[List[float], None] = None,
    color_norm: str = 'lin', pcolormesh_kwargs={}):
    """
    Projects the ASI images to a map at an altitude in the calibration file.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    map_alt: int
        The altitude in kilometers to project to. Must be an altitude value 
        in the calibration.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the time argument in seconds. Will raise a ValueError if no
        image time stamp is within the threshold.
    ax: plt.subplot
        The subplot to plot the frame on. If None, this function will
        create one.
    color_map: str
        The matplotlib colormap to use. If 'auto', will default to a
        black-red colormap for REGO and black-white colormap for THEMIS.
        For more information See https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    min_elevation: float
        Masks the pixels below min_elevation degrees.
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

    Returns
    -------
    frame_time: datetime.datetime
        The time of the current frame.
    frame: np.array
        The 2d ASI image corresponding to frame_time.
    cal: dict
        The calibration data for that mission-station.
    ax: plt.Axes
        The subplot object to modify the axis, labels, etc.
    im: plt.AxesImage
        The plt.imshow image object. Common use for im is to add a colorbar.
        The image is oriented in the map orientation (north is up, south is down,
        east is right, and west is left), contrary to the camera orientation where
        the east/west directions are flipped. Set azel_contours=True to confirm.
    """
    warnings.warn(
        'plot_map() is an experimental function. It is not tested. Use at your own risk.'
    )
    # Load data
    frame_time, frame = get_frame(
        time, mission, station, time_thresh_s=time_thresh_s
    )
    skymap = load_skymap(mission, station, time)

    # Check that the map_alt is in the skymap calibration data.
    assert map_alt in skymap['FULL_MAP_ALTITUDE']/1000, \
            f'{map_alt} km is not in skymap calibration altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
    alt_index = np.where(skymap['FULL_MAP_ALTITUDE']/1000 == map_alt)[0][0] 

    # Mask the frame and lat/lon map values where elevation < min_elevation with nans
    idh = np.where(skymap['FULL_ELEVATION'] < min_elevation)
    # Can't mask frame unless it is a float array.
    frame = frame.astype(float)
    frame[idh] = 0
    lon_map = skymap['FULL_MAP_LONGITUDE'][alt_index, :, :]
    lat_map = skymap['FULL_MAP_LATITUDE'][alt_index, :, :]
    lon_map[idh] = np.nan
    lat_map[idh] = np.nan

    # Set up the plot parameters
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        projection = ccrs.NearsidePerspective(
            central_latitude=skymap['SITE_MAP_LATITUDE'], 
            central_longitude=skymap['SITE_MAP_LONGITUDE'], 
            satellite_height=10000*map_alt
            )
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.coastlines()

    if color_bounds is None:
        lower, upper = np.quantile(frame, (0.25, 0.98))
        color_bounds = [lower, np.min([upper, lower * 10])]

    if (color_map == 'auto') and (mission.lower() == 'themis'):
        color_map = 'Greys_r'
    elif (color_map == 'auto') and (mission.lower() == 'rego'):
        color_map = colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r'])
    else:
        raise NotImplementedError('color_map == "auto" but the mission is unsupported')

    if color_norm == 'log':
        norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
    elif color_norm == 'lin':
        norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    else:
        raise ValueError('color_norm must be either "log" or "lin".')

    pcolormesh_nan(lon_map, lat_map,
                frame, ax, cmap=color_map, norm=norm)
    return frame_time, frame, skymap, ax

def pcolormesh_nan(x: np.ndarray, y: np.ndarray, c: np.ndarray, 
                    ax, cmap=None, norm=None):
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

    Stolen from:
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
        x[i, good[-1]:] = x[i, good[-1]]
        y[i, good[-1]:] = y[i, good[-1]]
        # Reassign all lat/lon columns before good[0] (all nans) to good[0].
        x[i, : good[0]] = x[i, good[0]]
        y[i, : good[0]] = y[i, good[0]]

    # Reassign all of the fully invalid lat/lon rows above top to the the max lat/lon value.
    x[:top, :] = np.nanmax(x[top, :])
    y[:top, :] = np.nanmax(y[top, :])
    # Same, but for the rows below bottom.
    x[bottom:, :] = np.nanmax(x[bottom, :])
    y[bottom:, :] = np.nanmax(y[bottom, :])

    # TODO: skymap rotation.
    # old masked c code: np.ma.masked_where(~mask[:-1, :-1], c)[::-1, ::-1]
    ax.pcolormesh(x, y, c[::-1, ::-1], 
                cmap=cmap, shading='flat', transform=ccrs.PlateCarree(), 
                norm=norm)
    return

if __name__ == '__main__':
    # https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-020-79665-5/MediaObjects/41598_2020_79665_Fig1_HTML.jpg?as=webp
    # plot_map(datetime(2017, 9, 15, 2, 34, 0), 'THEMIS', 'RANK', 110)

    # https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2008GL033794
    # plot_map(datetime(2007, 3, 13, 5, 8, 45), 'THEMIS', 'TPAS', 110)

    # http://themis.igpp.ucla.edu/nuggets/nuggets_2018/Gallardo-Lacourt/fig2.jpg
    plot_map(datetime(2010, 4, 5, 6, 7, 0), 'THEMIS', 'ATHA', 110)

    # https://www.essoar.org/doi/abs/10.1002/essoar.10507288.1
    # plot_map(datetime(2008, 1, 16, 11, 0, 0), 'THEMIS', 'GILL', 110)

    # cal = load_cal('THEMIS', 'GILL')
    # fig = plt.figure(figsize=(5, 5))
    # projection = ccrs.NearsidePerspective(
    #     central_latitude=cal['SITE_MAP_LATITUDE'], 
    #     central_longitude=cal['SITE_MAP_LONGITUDE'], 
    #     satellite_height=10000*110
    #     )
    # ax = fig.add_subplot(1, 1, 1, projection=projection)
    # ax.coastlines()
    # plot_map(datetime(2007, 1, 20, 0, 39, 0), 'THEMIS', 'TPAS', 110, ax=ax)
    # plot_map(datetime(2007, 1, 20, 0, 39, 0), 'THEMIS', 'GILL', 110, ax=ax)

    plt.show()