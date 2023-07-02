from typing import List
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import asilib.io.utils as utils
from asilib.analysis.keogram import keogram


def plot_keogram(
    asi_array_code: str,
    location_code: str,
    time_range: utils._time_range_type,
    map_alt: float = None,
    path: np.array = None,
    aacgm: bool = False,
    ax: plt.Axes = None,
    color_bounds: List[float] = None,
    color_norm: str = 'lin',
    title: bool = True,
    pcolormesh_kwargs: dict = {},
):
    """
    Makes a keogram along the central meridian.

    .. warning::
        Use :py:meth:`~asilib.imager.Imager.plot_keogram()` instead. This function will be
        removed in or after December 2023.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    map_alt: int
        The mapping altitude, in kilometers, used to index the mapped latitude in the
        skymap calibration data. If None, will plot pixel index for the y-axis.
    path: array
        Make a keogram along a custom path. Path shape must be (n, 2) and contain the
        lat/lon coordinates that are mapped to map_alt. If the map_alt kwarg is
        unspecified, this function will raise a ValueError.
    aacgm: bool
        Map the keogram latitudes to Altitude Adjusted Corrected Geogmagnetic Coordinates
        (aacgmv2) derived by Shepherd, S. G. (2014), Altitude-adjusted corrected geomagnetic
        coordinates: Definition and functional approximations, Journal of Geophysical
        Research: Space Physics, 119, 7501-7521, doi:10.1002/2014JA020264.
    ax: plt.Axes
        The subplot to plot the image on. If None, this function will
        create one.
    color_bounds: List[float]
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    color_norm: str
        Sets the linear ('lin') or logarithmic ('log') color normalization.
    title: bool
        Toggles a default plot title with the format "date ASI_array_code-location_code keogram".
    pcolormesh_kwargs: dict
        A dictionary of keyword arguments (kwargs) to pass directly into
        plt.pcolormesh. One use of this parameter is to change the colormap. For example,
        pcolormesh_kwargs = {'cmap':'tu'}

    Returns
    -------
    ax: plt.Axes
        The subplot object to modify the axis, labels, etc.
    im: plt.AxesImage
        The plt.pcolormesh image object. Common use for im is to add a colorbar.

    Raises
    ------
    AssertionError
        If len(time_range) != 2. Also if map_alt does not equal the mapped
        altitudes in the skymap mapped values.

    Example
    -------
    | import matplotlib.pyplot as plt
    |
    | import asilib
    |
    | asi_array_code='REGO'
    | location_code='LUCK'
    | time_range=['2017-09-27T07', '2017-09-27T09']
    |
    | fig, ax = plt.subplots(figsize=(8, 6))
    | ax, im = asilib.plot_keogram(asi_array_code, location_code, time_range,
    |                ax=ax, map_alt=230, color_bounds=(300, 800), pcolormesh_kwargs={'cmap':'turbo'})
    |
    | plt.colorbar(im)
    | plt.tight_layout()
    | plt.show()
    """
    warnings.warn(
        "Use asilib.Imager.plot_keogram() instead. This function will be removed "
        "in or after December 2023.",
        DeprecationWarning,
    )
    time_range = utils._validate_time_range(time_range)
    keo_df = keogram(asi_array_code, location_code, time_range, map_alt, path=path, aacgm=aacgm)

    if ax is None:
        _, ax = plt.subplots()

    # Figure out the color_bounds from the image data.
    if color_bounds is None:
        lower, upper = np.quantile(keo_df, (0.25, 0.98))
        color_bounds = [lower, np.min([upper, lower * 10])]

    if color_norm == 'log':
        norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
    elif color_norm == 'lin':
        norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    else:
        raise ValueError('color_norm must be either "log" or "lin".')

    im = ax.pcolormesh(
        keo_df.index,
        keo_df.columns,
        keo_df.to_numpy()[:-1, :-1].T,
        norm=norm,
        shading='flat',
        **pcolormesh_kwargs,
    )

    if title:
        ax.set_title(
            f'{time_range[0].date()} | {asi_array_code.upper()}-{location_code.upper()} keogram'
        )
    return ax, im
