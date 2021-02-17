import pathlib
from typing import List, Union, Optional, Sequence, Generator
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ffmpeg

import asilib.load as load
import asilib.config as config


def plot_movie(time_range: Sequence[Union[datetime, str]], mission: str, station: str, **kwargs):
    """
    A warpper for plot_movie_generator() generator function. This function calls
    plot_movie_generator() in a for loop, nothing more. The two function's arguments
    and keyword arguments are identical.

    To make movies, you'll need to install ffmpeg in your operating system.

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
        The matplotlib colormap to use. If 'auto', will default to a
        black-red colormap for REGO and black-white colormap for THEMIS.
        For more information See
        https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    ax: plt.subplot()
        The optional subplot that will be drawn on.
    azel_contours: bool
        Switch azimuth and elevation contours on or off.
    movie_format: str
        The movie format: mp4 has better compression but avi can be
        opened by the VLC player.
    frame_rate: int
        The movie frame rate.
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    delete_pngs: bool
        Remove the intermediate png files created for the ffmpeg library.
        Be careful that if you call this function multiple times without
        removing the png files, the files used in the movies come from the
        search pattern mission_station. So if there are multiple times
        from the same mission/station, those pngs will all be written to one
        movie.

    Example
    -------
    from datetime import datetime

    import asilib

    time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
    asilib.plot_movie(time_range, 'THEMIS', 'FSMI')

    Return
    -------
    None
    """
    movie_generator = plot_movie_generator(time_range, mission, station, **kwargs)

    for frame_time, im, ax in movie_generator:
        pass
    return


def plot_movie_generator(
    time_range: Sequence[Union[datetime, str]],
    mission: str,
    station: str,
    force_download: bool = False,
    add_label: bool = True,
    color_map: str = 'auto',
    color_bounds: Union[List[float], None] = None,
    color_norm: str = 'log',
    azel_contours: bool = False,
    ax: plt.subplot = None,
    movie_format: str = 'mp4',
    frame_rate=10,
    overwrite_output: bool = False,
    delete_pngs: bool = True,
) -> Generator[datetime, plt.Axes, matplotlib.image.AxesImage]:
    """
    A generator function that loads the ASI data and then yields individual ASI images,
    frame by frame. This allows the user to add content to each frame, such as the
    spacecraft position, and that will convert it to a movie. If you just want to make
    an ASI movie, use the wrapper for this function, plot_movie().

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
        The matplotlib colormap to use. If 'auto', will default to a
        black-red colormap for REGO and black-white colormap for THEMIS.
        For more information See
        https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    ax: plt.subplot()
        The optional subplot that will be drawn on.
    movie_format: str
        The movie format: mp4 has better compression but avi can be
        opened by the VLC player.
    frame_rate: int
        The movie frame rate.
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    azel_contours: bool
        Switch azimuth and elevation contours on or off.
    overwrite_output: bool
        If true, the output will be overwritten automatically. If false it will
        prompt the user to answer y/n.
    delete_pngs: bool
        Remove the intermediate png files created for the ffmpeg library.
        Be careful that if you call this function multiple times without
        removing the png files, the files used in the movies come from the
        search pattern mission_station. So if there are multiple times
        from the same mission/station, those pngs will all be written to one
        movie.

    Yields
    ------
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

    import asilib

    time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
    movie_generator = asilib.plot_movie_generator(time_range, 'THEMIS, 'FSMI')

    for frame_time, im, ax in movie_generator:
        # The code that modifies each frame here.
        pass
    """
    try:
        frame_times, frames = load.get_frames(
            time_range, mission, station, force_download=force_download
        )
    except AssertionError as err:
        if '0 number of time stamps were found in time_range' in str(err):
            print(f'The file exists for {mission}/{station}, but no data ' f'between {time_range}.')
            raise
        else:
            raise
    if ax is None:
        _, ax = plt.subplots()

    # Create the movie directory inside config.ASI_DATA_DIR if it does
    # not exist.
    save_dir = config.ASI_DATA_DIR / 'movies' / 'temp'
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)
        print(f'Created a {save_dir} directory')

    if (color_map == 'auto') and (mission.lower() == 'themis'):
        color_map = 'Greys_r'
    elif (color_map == 'auto') and (mission.lower() == 'rego'):
        color_map = colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r'])
    else:
        raise NotImplementedError('color_map == "auto" but the mission is unsupported')

    save_paths = []

    for frame_time, frame in zip(frame_times, frames):
        # If the frame is all 0s we have a bad frame and we need to skip it.
        if np.all(frame == 0):
            continue
        ax.clear()
        plt.axis('off')
        # Figure out the color_bounds from the frame data.
        if color_bounds is None:
            lower, upper = np.quantile(frame, (0.25, 0.98))
            color_bounds = [lower, np.min([upper, lower * 10])]

        if color_norm == 'log':
            norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
        elif color_norm == 'lin':
            norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
        else:
            raise ValueError('color_norm must be either "log" or "lin".')

        im = ax.imshow(frame, cmap=color_map, norm=norm, origin='lower')
        if add_label:
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

        # Give the user the control of the subplot, image object, and return the frame time
        # so that the user can manipulate the image to add, for example, the satellite track.
        yield frame_time, ax, im  # TODO: Add the image array return statement and update docs.

        # Save the file and clear the subplot for next frame.
        save_name = (
            f'{frame_time.strftime("%Y%m%d_%H%M%S")}_{mission.lower()}_' f'{station.lower()}.png'
        )
        plt.savefig(save_dir / save_name)
        save_paths.append(save_dir / save_name)

    # Make the movie
    # TODO: Add a temp/ subfolder with the start time to save the png files.
    movie_file_name = (
        f'{frame_times[0].strftime("%Y%m%d_%H%M%S")}_'
        f'{frame_times[-1].strftime("%H%M%S")}_'
        f'{mission.lower()}_{station.lower()}.{movie_format}'
    )
    movie_obj = ffmpeg.input(
        str(save_dir) + f'/*{mission.lower()}_{station.lower()}.png',
        pattern_type='glob',
        framerate=frame_rate,
    )
    movie_obj.output(str(save_dir.parent / movie_file_name)).run(overwrite_output=overwrite_output)
    # Clean up.
    if delete_pngs:
        for path in save_paths:
            path.unlink()
    return


if __name__ == "__main__":
    plot_movie(
        (datetime(2017, 9, 15, 2, 34, 0), datetime(2017, 9, 15, 2, 36, 0)),
        'THEMIS',
        'RANK',
        color_norm='log',
        azel_contours=True,
    )
