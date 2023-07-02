import shutil
import importlib
import pathlib
from datetime import datetime
from typing import List, Union, Generator, Tuple
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import asilib
import asilib.plot.utils
from asilib.io.load import load_image
from asilib.io.load import load_skymap
from asilib.plot.plot_map import make_map
from asilib.plot.plot_map import _pcolormesh_nan
from asilib.analysis.start_generator import start_generator
from asilib.plot.animate_fisheye import _write_movie
from asilib.plot.animate_fisheye import Images


def animate_map(
    asi_array_code: str,
    location_code: str,
    time_range: asilib.io.utils._time_range_type,
    map_alt: float,
    **kwargs,
):
    """
    Projects a series of THEMIS or REGO images on a map at map_alt altitude in kilometers and
    animates them.

    This function basically runs ``animate_map_generator()`` in a for loop. The two function's
    arguments and keyword arguments are identical, so see ``animate_map_generator()`` docs for
    the full argument list.

    Note: To make animations, you'll need to install ``ffmpeg`` in your operating system.

    .. warning::
        Use :py:meth:`~asilib.imager.Imager.animate_map()` instead. This function will be
        removed in or after December 2023.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.

    Returns
    -------
    None

    Raises
    ------
    NotImplementedError
        If the colormap is unspecified ('auto' by default) and the
        auto colormap is undefined for an ASI array.
    ValueError
        If the color_norm kwarg is not "log" or "lin".
    AssertionError
        If the ASI data exists for that time period, but without time stamps
        inside time_range.

    Example
    -------
    | from datetime import datetime
    |
    | import asilib
    |
    | map_alt=110  # km
    | time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
    | asilib.animate_map('THEMIS', 'FSMI', time_range, map_alt=map_alt)
    | print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "animations"}')
    """
    warnings.warn(
        "Use asilib.Imager.animate_map() instead. This function will be removed "
        "in or after December 2023.",
        DeprecationWarning,
    )

    map_generator = animate_map_generator(
        asi_array_code, location_code, time_range, map_alt, **kwargs
    )

    for _ in map_generator:
        pass
    return


@start_generator
def animate_map_generator(
    asi_array_code: str,
    location_code: str,
    time_range: asilib.io.utils._time_range_type,
    map_alt: float,
    min_elevation: float = 10,
    overwrite: bool = False,
    color_map: str = 'auto',
    color_bounds: Union[List[float], None] = None,
    color_norm: str = 'log',
    ax: plt.Axes = None,
    map_shapefile: Union[str, pathlib.Path] = 'ne_10m_land',
    coast_color: str = 'k',
    land_color: str = 'g',
    ocean_color: str = 'w',
    lon_bounds: tuple = (-140, -60),
    lat_bounds: tuple = (40, 82),
    label: bool = True,
    movie_container: str = 'mp4',
    ffmpeg_output_params={},
    pcolormesh_kwargs: dict = {},
) -> Generator[Tuple[datetime, np.ndarray, plt.Axes, matplotlib.image.AxesImage], None, None]:
    """
    Projects a series of THEMIS or REGO images on a map at map_alt altitude in kilometers and
    animates them. This generator function is useful if you need to superpose other data onto a
    map in the movie.

    Once this generator is initiated with the name `gen`, for example, but **before**
    the for loop, you can get the ASI images and times by calling `gen.send('data')`.
    This will yield a collections.namedtuple with `time` and `images` attributes.

    .. warning::
        Use :py:meth:`~asilib.imager.Imager.animate_map_gen()` instead. This function will be
        removed in or after December 2023.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    map_alt: float
        The altitude in kilometers to project to. Must be an altitude value
        in the skymap calibration.
    min_elevation: float
        Masks the pixels below min_elevation degrees.
    overwrite: bool
        If True, the animation will be overwritten. Otherwise it will prompt
        the user to answer y/n.
    color_map: str
        The matplotlib colormap to use. If 'auto', will default to a
        black-red colormap for REGO and black-white colormap for THEMIS.
        For more information See
        https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will
        automatically set it to low=1st_quartile and
        high=min(3rd_quartile, 10*1st_quartile)
    ax: plt.Axes
        The optional subplot that will be drawn on.
    map_shapefile: str or pathlib.Path
        The path to the shapefile zip archive. If str, it will try to load the
        shapefile in asilib/data/{file}.
    coast_color: str
        The coast color. If None will not draw it.
    land_color: str
        The land color. If None will not draw it.
    ocean_color: str
        The ocean color. If None will not draw it.
    lon_bounds: tuple
        The map's longitude bounds.
    lat_bounds: tuple
        The map's latitude bounds.
    label: bool
        Annotates the map with the ASI code in the center of the image.
    movie_container: str
        The movie container: mp4 has better compression but avi was determined
        to be the official container for preserving digital video by the
        National Archives and Records Administration.
    ffmpeg_output_params: dict
        The additional/overwitten ffmpeg output parameters. The default parameters are:
        framerate=10, crf=25, vcodec=libx264, pix_fmt=yuv420p, preset=slower.
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    pcolormesh_kwargs: dict
        A dictionary of keyword arguments (kwargs) to pass directly into
        plt.pcolormesh. One use of this parameter is to change the colormap. For example,
        pcolormesh_kwargs = {'cmap':'tu}

    Yields
    ------
    image_time: datetime.datetime
        The time of the current image.
    image: np.ndarray
        A 2d image array of the image corresponding to image_time
    ax: plt.Axes
        The subplot object to modify the axis, labels, etc.
    im: plt.AxesImage
        The plt.pcolormesh object.

    Raises
    ------
    NotImplementedError
        If the colormap is unspecified ('auto' by default) and the
        auto colormap is undefined for an ASI array.
    ValueError
        If the color_norm kwarg is not "log" or "lin".
    AssertionError
        If the ASI data exists for that time period, but without time stamps
        inside time_range.

    Example
    -------
    | from datetime import datetime
    |
    | import asilib
    |
    | map_alt=110
    | time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
    | map_generator = asilib.animate_map_generator('THEMIS', 'FSMI', time_range, map_alt=map_alt,
    |     lon_bounds=(-125, -100), lat_bounds=(55, 70))
    |
    | for (image_time1, image, ax, p) in map_generator:
    |       # The code that modifies each image here.
    |       pass
    |
    | print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "animations"}')
    """
    warnings.warn(
        "Use asilib.Imager.animate_fisheye_gen() instead. This function will be removed "
        "in or after December 2023.",
        DeprecationWarning,
    )
    try:
        image_times, images = load_image(asi_array_code, location_code, time_range=time_range)
    except AssertionError as err:
        if '0 number of time stamps were found in time_range' in str(err):
            print(
                f'The file exists for {asi_array_code}/{location_code}, but no data '
                f'between {time_range}.'
            )
            raise
        else:
            raise
    skymap = load_skymap(asi_array_code, location_code, time_range[0])

    # Create the movie directory inside asilib.config['ASI_DATA_DIR'] if it does
    # not exist.
    image_save_dir = pathlib.Path(
        asilib.config['ASI_DATA_DIR'],
        'animations',
        'images',
        f'{image_times[0].strftime("%Y%m%d_%H%M%S")}_{asi_array_code.lower()}_'
        f'{location_code.lower()}_map',
    )
    # If the image directory exists we need to first remove all of the images to avoid
    # animating images from different calls.
    if image_save_dir.is_dir():
        shutil.rmtree(image_save_dir)
    image_save_dir.mkdir(parents=True)
    print(f'Created a {image_save_dir} directory')

    # Check that the map_alt is in the skymap calibration data.
    assert (
        map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
    ), f'{map_alt} km is not in skymap calibration altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
    alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]
    image, lon_map, lat_map = _mask_low_horizon(
        images,
        skymap['FULL_MAP_LONGITUDE'][alt_index, :, :],
        skymap['FULL_MAP_LATITUDE'][alt_index, :, :],
        skymap['FULL_ELEVATION'],
        min_elevation,
    )

    if ax is None:
        ax = make_map(
            file=map_shapefile,
            coast_color=coast_color,
            land_color=land_color,
            ocean_color=ocean_color,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
        )

    color_map = asilib.plot.utils.get_color_map(asi_array_code, color_map)

    # With the @start_generator decorator, when this generator first gets called, it
    # will halt here. This way the errors due to missing data will be raised up front.
    # user_input can be used to get the image_times and images out of the generator.
    user_input = yield
    if isinstance(user_input, str) and 'data' in user_input.lower():
        yield Images(image_times, images)

    if label:
        ax.text(
            skymap['SITE_MAP_LONGITUDE'],
            skymap['SITE_MAP_LATITUDE'],
            location_code.upper(),
            color='r',
            va='center',
            ha='center',
        )

    image_paths = []
    for i, (image_time, image) in enumerate(zip(image_times, images)):
        if 'p' in locals():
            p.remove()  # noqa

        # if-else statement is to recalculate color_bounds for every image
        # and set it to _color_bounds. If _color_bounds did not exist,
        # color_bounds will be overwritten after the first iteration which will
        # disable the dynamic color bounds for each image.
        if color_bounds is None:
            _color_bounds = asilib.plot.utils.get_color_bounds(image)
        else:
            _color_bounds = color_bounds

        norm = asilib.plot.utils.get_color_norm(color_norm, _color_bounds)

        p = _pcolormesh_nan(
            lon_map,
            lat_map,
            image,
            ax,
            cmap=color_map,
            norm=norm,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )

        # Give the user the control of the subplot, image object, and return the image time
        # so that the user can manipulate the image to add, for example, the satellite track.
        yield image_time, image, ax, p

        # Save the plot before the next iteration.
        save_name = f'{str(i).zfill(5)}.png'
        plt.savefig(image_save_dir / save_name)
        image_paths.append(image_save_dir / save_name)

    # Make the movie
    movie_save_name = (
        f'{image_times[0].strftime("%Y%m%d_%H%M%S")}_'
        f'{image_times[-1].strftime("%H%M%S")}_'
        f'{asi_array_code.lower()}_{location_code.lower()}_map.{movie_container}'
    )
    movie_save_path = image_save_dir.parents[1] / movie_save_name
    _write_movie(image_paths, movie_save_path, ffmpeg_output_params, overwrite)
    return


def _mask_low_horizon(images, lon_map, lat_map, el_map, min_elevation):
    """
    Mask the images, skymap['FULL_MAP_LONGITUDE'], skymap['FULL_MAP_LONGITUDE'] arrays
    with np.nans where the skymap['FULL_ELEVATION'] is nan or
    skymap['FULL_ELEVATION'] < min_elevation.
    """
    idh = np.where(np.isnan(el_map) | (el_map < min_elevation))
    # Copy variables to not modify original np.arrays.
    images_copy = images.copy()
    lon_map_copy = lon_map.copy()
    lat_map_copy = lat_map.copy()
    # Can't mask image unless it is a float array.
    image_copy = images_copy.astype(float)
    image_copy[:, idh] = np.nan
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
    return images_copy, lon_map_copy, lat_map_copy
