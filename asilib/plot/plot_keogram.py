import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from asilib.io.load import _validate_time_range
from asilib.analysis.keogram import keogram


def plot_keogram(
    time_range,
    mission,
    station,
    map_alt=None,
    ax=None,
    color_bounds=None,
    color_norm='lin',
    title=True,
    pcolormesh_kwargs={},
):
    """
    Makes a keogram along the central meridian.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        images. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    map_alt: int, optional
        The mapping altitude, in kilometers, used to index the mapped latitude in the
        skymap calibration data. If None, will plot pixel index for the y-axis.
    ax: plt.subplot
        The subplot to plot the image on. If None, this function will
        create one.
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    title: bool
        Toggles a default plot title with the format "date mission-station keogram".
    pcolormesh_kwargs: dict
        A dictionary of keyword arguments (kwargs) to pass directly into
        plt.pcolormesh. One use of this parameter is to change the colormap. For example,
        pcolormesh_kwargs = {'cmap':'tu}

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
    | mission='REGO'
    | station='LUCK'
    |
    | fig, ax = plt.subplots(figsize=(8, 6))
    | ax, im = asilib.plot_keogram(['2017-09-27T07', '2017-09-27T09'], mission, station,
    |                ax=ax, map_alt=230, color_bounds=(300, 800), pcolormesh_kwargs={'cmap':'turbo'})
    |
    | plt.colorbar(im)
    | plt.tight_layout()
    | plt.show()
    """
    time_range = _validate_time_range(time_range)
    keo_df = keogram(time_range, mission, station, map_alt)

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
        ax.set_title(f'{time_range[0].date()} | {mission.upper()}-{station.upper()} keogram')
    return ax, im
