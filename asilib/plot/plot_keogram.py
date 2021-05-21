import dateutil.parser

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from asilib.io.load import get_frames


def keogram(time_range, mission, station, ax=None, color_bounds=None, color_norm='lin'):
    """
    Makes a keogram along the central meridian.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        frames. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    ax: plt.subplot
        The subplot to plot the frame on. If None, this function will
        create one.
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    """

    # Run a few checks to make sure that the time_range parameter has length 2 and
    # convert time_range to datetime objects if it is a string.
    assert len(time_range) == 2, f'len(time_range) = {len(time_range)} is not 2.'
    for i, t_i in enumerate(time_range):
        if isinstance(t_i, str):
            time_range[i] = dateutil.parser.parse(t_i)

    frame_times, frames = get_frames(time_range, mission, station)

    # Find the pixel at the center of the camera.
    center_pixel = int(frames.shape[1]/2)

    keo = np.nan*np.zeros((frames.shape[0], frames.shape[2]))

    for i, frame in enumerate(frames):
        keo[i, :] = frame[center_pixel, :]

    if ax is None:
        _, ax = plt.subplots()

    # Figure out the color_bounds from the frame data.
    if color_bounds is None:
        lower, upper = np.quantile(keo, (0.25, 0.98))
        color_bounds = [lower, np.min([upper, lower * 10])]

    if color_norm == 'log':
        norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
    elif color_norm == 'lin':
        norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    else:
        raise ValueError('color_norm must be either "log" or "lin".')

    im = ax.pcolormesh(keo.T, norm=norm)
    plt.colorbar(im)
    return


if __name__ == '__main__':
    keogram(['2017-09-27T07:00:00', '2017-09-27T10:00:00'], 'REGO', 'LUCK')
    plt.show()