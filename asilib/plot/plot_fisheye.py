from datetime import datetime
from typing import List, Union, Tuple
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from asilib.io import load
from asilib.io import utils


def plot_fisheye(
    asi_array_code: str,
    location_code: str,
    time: utils._time_type,
    redownload: bool = False,
    time_thresh_s: float = 3,
    ax: plt.Axes = None,
    label: bool = True,
    color_map: str = 'auto',
    color_bounds: List[float] = None,
    color_norm: str = 'log',
    azel_contours: bool = False,
) -> Tuple[datetime, np.array, plt.Axes, matplotlib.image.AxesImage]:
    """
    Plots one fisheye image, oriented with North on the top, and East on the right of the image.

    .. warning::
        Use :py:meth:`~asilib.imager.Imager.plot_fisheye()` instead. This function will be
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
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    redownload: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete and corrupted.
    time_thresh_s: float
        The maximum allowable time difference between ``time`` and an ASI time stamp.
    ax: plt.Axes
        The subplot to plot the image on. If None, this function will
        create one.
    label: bool
        Flag to add the "asi_array_code/location_code/image_time" text to the plot.
    color_map: str
        The matplotlib colormap to use. If 'auto', will default to a
        black-red colormap for REGO and black-white colormap for THEMIS.
        For more information See https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    color_bounds: List[float]
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile). This range works well
        for most cases.
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    azel_contours: bool
        Switch azimuth and elevation contours on or off.

    Returns
    -------
    image_time: datetime.datetime
        The time of the current image.
    image: np.array
        The 2d ASI image corresponding to image_time.
    ax: plt.Axes
        The subplot object to modify the axis, labels, etc.
    im: plt.AxesImage
        The plt.imshow image object. Common use for im is to add a colorbar.
        The image is oriented in the map orientation (north is up, south is down,
        west is right, and east is left), contrary to the camera orientation where
        the east/west directions are flipped. Set azel_contours=True to confirm.

    Raises
    ------
    NotImplementedError
        If the colormap is unspecified ('auto' by default) and the
        auto colormap is undefined for an ASI array.
    ValueError
        If the color_norm kwarg is not "log" or "lin".

    Example
    -------
    | from datetime import datetime
    |
    | import matplotlib.pyplot as plt
    |
    | import asilib
    |
    | # A bright auroral arc that was analyzed by Imajo et al., 2021 "Active
    | # auroral arc powered by accelerated electrons from very high altitudes"
    | time = datetime(2017, 9, 15, 2, 34, 0)
    | image_time, ax, im = asilib.plot_fisheye('THEMIS', 'RANK', time,
    |     color_norm='log', redownload=False)
    |
    | plt.colorbar(im)
    | ax.axis('off')
    | plt.show()
    """
    warnings.warn(
        "Use asilib.Imager.plot_fisheye() instead. This function will be removed "
        "in or after December 2023.",
        DeprecationWarning,
    )
    if ax is None:
        _, ax = plt.subplots()

    image_time, image = load.load_image(
        asi_array_code,
        location_code,
        time=time,
        redownload=redownload,
        time_thresh_s=time_thresh_s,
    )

    # Figure out the color_bounds from the image data.
    if color_bounds is None:
        lower, upper = np.quantile(image, (0.25, 0.98))
        color_bounds = [lower, np.min([upper, lower * 10])]

    if (color_map == 'auto') and (asi_array_code.lower() == 'themis'):
        color_map = 'Greys_r'
    elif (color_map == 'auto') and (asi_array_code.lower() == 'rego'):
        color_map = colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r'])
    else:
        raise NotImplementedError('color_map == "auto" but the ASI array is unsupported')

    if color_norm == 'log':
        norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
    elif color_norm == 'lin':
        norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    else:
        raise ValueError('color_norm must be either "log" or "lin".')

    im = ax.imshow(image[:, :], cmap=color_map, norm=norm, origin="lower")
    if label:
        ax.text(
            0,
            0,
            f"{asi_array_code.upper()}/{location_code.upper()}\n{image_time.strftime('%Y-%m-%d %H:%M:%S')}",
            va='bottom',
            transform=ax.transAxes,
            color='white',
        )
    if azel_contours:
        skymap_dict = load.load_skymap(
            asi_array_code, location_code, image_time, redownload=redownload
        )

        az_contours = ax.contour(
            skymap_dict['FULL_AZIMUTH'],
            colors='yellow',
            linestyles='dotted',
            levels=np.arange(0, 361, 90),
            alpha=1,
        )
        el_contours = ax.contour(
            skymap_dict['FULL_ELEVATION'],
            colors='yellow',
            linestyles='dotted',
            levels=np.arange(0, 91, 30),
            alpha=1,
        )
        plt.clabel(az_contours, inline=True, fontsize=8)
        plt.clabel(el_contours, inline=True, fontsize=8, rightside_up=True)
    return image_time, image, ax, im
