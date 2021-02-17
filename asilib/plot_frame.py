import pathlib
from datetime import datetime, timedelta
import dateutil.parser
from typing import List, Union, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cdflib

from asilib import config
from asilib import load


def plot_frame(
    time: Union[datetime, str],
    mission: str,
    station: str,
    force_download: bool = False,
    time_thresh_s: float = 3,
    ax: plt.subplot = None,
    add_label: bool = True,
    color_map: str = 'auto',
    color_bounds: Union[List[float], None] = None,
    color_norm: str = 'log',
    azel_contours: bool = False,
) -> Tuple[datetime, plt.Axes, matplotlib.image.AxesImage]:
    """
    Plots one ASI image frame given the mission (THEMIS or REGO), station, and
    the day date-time parameters. If a file does not locally exist, the load_img()
    function will attempt to download it.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the time argument in seconds. Will raise a ValueError if no
        image time stamp is within the threshold.
    ax: plt.subplot
        The subplot to plot the frame on. If None, this function will
        create one.
    add_label: bool
        Flag to add the "mission/station/frame_time" text to the plot.
    color_map: str
        The matplotlib colormap to use. If 'auto', will default to a
        black-red colormap for REGO and black-white colormap for THEMIS.
        For more information See
        https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    azel_contours: bool
        Switch azimuth and elevation contours on or off.

    Returns
    -------
    frame_time: datetime.datetime
        The time of the current frame.
    ax: plt.Axes
        The subplot object to modify the axis, labels, etc.
    im: plt.AxesImage
        The plt.imshow image object. Common use for im is to add a colorbar.
        The image is oriented in the map orientation (north is up, south is down,
        east is right, and west is left), contrary to the camera orientation where
        the east/west directions are flipped. Set azel_contours=True to confirm.

    Example
    -------
    from datetime import datetime

    import matplotlib.pyplot as plt

    import asilib

    # A bright auroral arc that was analyzed by Imajo et al., 2021 "Active
    # auroral arc powered by accelerated electrons from very high altitudes"
    frame_time, ax, im = asilib.plot_frame(datetime(2017, 9, 15, 2, 34, 0), 'THEMIS', 'RANK',
                        color_norm='log', force_download=False)
    plt.colorbar(im)
    ax.axis('off')
    plt.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    frame_time, frame = load.get_frame(
        time, mission, station, force_download=force_download, time_thresh_s=time_thresh_s
    )

    # Figure out the color_bounds from the frame data.
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

    im = ax.imshow(frame[:, :], cmap=color_map, norm=norm, origin="lower")
    ax.text(
        0,
        0,
        f"{mission}/{station}\n{frame_time.strftime('%Y-%m-%d %H:%M:%S')}",
        va='bottom',
        transform=ax.transAxes,
        color='white',
    )
    if azel_contours:
        cal_dict = load.load_cal_file(mission, station, force_download=force_download)

        az_contours = ax.contour(
            cal_dict['FULL_AZIMUTH'][::-1, ::-1],
            colors='yellow',
            linestyles='dotted',
            levels=np.arange(0, 361, 90),
            alpha=1,
        )
        el_contours = ax.contour(
            cal_dict['FULL_ELEVATION'][::-1, ::-1],
            colors='yellow',
            linestyles='dotted',
            levels=np.arange(0, 91, 30),
            alpha=1,
        )
        plt.clabel(az_contours, inline=True, fontsize=8)
        plt.clabel(el_contours, inline=True, fontsize=8, rightside_up=True)
    return frame_time, ax, im  # TODO: Add the image array return statement and update docs.


if __name__ == '__main__':
    # time, ax, im = plot_frame(datetime(2017, 9, 15, 2, 30, 0), 'THEMIS', 'RANK',
    #                     color_norm='log', force_download=False, azel_contours=True)
    time, ax, im = plot_frame(
        datetime(2017, 9, 15, 2, 36, 36),
        'THEMIS',
        'RANK',
        color_norm='log',
        force_download=False,
        azel_contours=True,
    )
    plt.colorbar(im)
    plt.axis('off')
    plt.show()
