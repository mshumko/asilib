import pathlib
from typing import List, Union, Optional, Sequence
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from asi.load import get_frames
import asi.config as config


def plot_movie(time_range: Sequence[Union[datetime, str]], mission: str, station: str, 
            force_download: bool=False, add_label: bool=True, color_map: str='hot',
            color_bounds: Union[List[float], None]=None, color_norm: str='log'):
    """
    Make a series of ASI plots that can be later combined into a movie using
    QuickTime or mencoder. For reference, see 
    https://matplotlib.org/gallery/animation/movie_demo_sgskip.html

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
    force_download: bool (optional)
        If True, download the file even if it already exists.
    add_label: bool
        Flag to add the "mission/station/frame_time" text to the plot.
    color_map: str
        The matplotlib colormap to use. See 
        https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will 
        automatically set it to low=1st_quartile and 
        high=min(3rd_quartile, 10*1st_quartile)
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.

    Returns
    -------
    """
    frame_times, frames = get_frames(time_range, mission, station, 
                                    force_download=force_download)
    _, ax = plt.subplots()

    # Create the movie directory inside config.ASI_DATA_DIR if it does 
    # not exist.
    save_dir = config.ASI_DATA_DIR / 'movies'
    if not save_dir.is_dir():
        save_dir.mkdir()
        print(f'Created a {save_dir} directory')

    for frame_time, frame in zip(frame_times, frames):
        ax.clear()
        plt.axis('off')
        # Figure out the color_bounds from the frame data.
        if color_bounds is None:
            lower, upper = np.quantile(frame, (0.25, 0.98))
            color_bounds = [lower, np.min([upper, lower*10])]

        if color_norm == 'log':
            norm=colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
        elif color_norm == 'lin':
            norm=colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
        else:
            raise ValueError('color_norm must be either "log" or "lin".')

        im = ax.imshow(frame, cmap=color_map, norm=norm)
        ax.text(0, 0, f"{mission}/{station}\n{frame_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                va='bottom', transform=ax.transAxes, color='white')
        yield frame_time, im, ax
        # Save the file and clear the subplot for next frame.
        save_name = (f'{mission.lower()}_{station.lower()}_'
                     f'{frame_time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(save_dir / save_name)
    
if __name__ == "__main__":
    plot_movie((datetime(2016, 10, 16, 5, 43, 0), datetime(2016, 10, 16, 5, 44, 0)), 
                'THEMIS', 'RANK', color_norm='log')