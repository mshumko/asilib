"""
This module contains functions to project the ASI images to a map.
"""
from typing import List, Union, Optional, Sequence, Tuple
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from asilib import config
from asilib.io import load


def plot_map(time: Union[datetime, str], mission: str,
    station: str, map_alt: int, time_thresh_s: float = 3,
    ax: plt.subplot = None, color_map: str = 'auto',
    color_bounds: Union[List[float], None] = None,
    color_norm: str = 'log', pcolormesh_kwargs={}):
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
    # Load data
    frame_time, frame = load.get_frame(
        time, mission, station, time_thresh_s=time_thresh_s
    )
    cal = load.load_cal(mission, station)

    # Check that the map_alt is in the calibration data.
    assert map_alt in cal['FULL_MAP_ALTITUDE']/1000, \
            f'{map_alt} km is not in calibration altitudes: {cal["FULL_MAP_ALTITUDE"]/1000} km'
    alt_index = np.where(cal['FULL_MAP_ALTITUDE']/1000 == map_alt)[0][0] 

    # Set up the plot parameters
    if ax is None:
        _, ax = plt.subplots()

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

    # Make the plot
    im = ax.pcolormesh(cal['FULL_MAP_LONGITUDE'][alt_index, :, :], 
                  cal['FULL_MAP_LATITUDE'][alt_index, :, :],
                  frame, 
                  norm=norm, shading='flat', **pcolormesh_kwargs
                  )
    return frame_time, frame, cal, ax, im

if __name__ == '__main__':
    plot_map(datetime(2017, 9, 15, 2, 34, 0), 'THEMIS', 'RANK', 110)
    plt.show()