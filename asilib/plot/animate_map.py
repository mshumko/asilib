from datetime import datetime
from typing import List, Union, Generator, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ffmpeg

import asilib
from asilib.io import utils
from asilib.io.load import load_image, load_skymap
from asilib.analysis.start_generator import start_generator

@start_generator
def animate_map_generator(
    asi_array_code: str,
    location_code: str,
    time_range: utils._time_range_type,
    map_alt: float,
    min_elevation: float = 10,
    force_download: bool = False,
    label: bool = True,
    color_map: str = 'auto',
    color_bounds: Union[List[float], None] = None,
    color_norm: str = 'log',
    azel_contours: bool = False,
    ax: plt.Axes = None,
    map_style: str = 'green',
    movie_container: str = 'mp4',
    ffmpeg_output_params={},
    overwrite: bool = False,
) -> Generator[Tuple[datetime, np.ndarray, plt.Axes, matplotlib.image.AxesImage], None, None]:
    """
    TODO: Update the doc string.
    Projects the fisheye images into the ionosphere at map_alt (altitude in kilometers) and 
    animates them using ffmpeg. 

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
    force_download: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    label: bool
        Flag to add the "asi_array_code/location_code/image_time" text to the plot.
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
    map_style: str
        If ax is None, this kwarg toggles between two predefined map styles:
        'green' map has blue oceans and green land, while the `white` map
        has white oceans and land with black coastlines.
    movie_container: str
        The movie container: mp4 has better compression but avi was determined
        to be the official container for preserving digital video by the
        National Archives and Records Administration.
    ffmpeg_output_params: dict
        The additional/overwitten ffmpeg output parameters. The default parameters are:
        framerate=10, crf=25, vcodec=libx264, pix_fmt=yuv420p, preset=slower.
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.
    azel_contours: bool
        Switch azimuth and elevation contours on or off.
    overwrite: bool
        If true, the output will be overwritten automatically. If false it will
        prompt the user to answer y/n.

    Yields
    ------
    image_time: datetime.datetime
        The time of the current image.
    image: np.ndarray
        A 2d image array of the image corresponding to image_time
    ax: plt.Axes
        The subplot object to modify the axis, labels, etc.
    im: plt.AxesImage
        The plt.imshow image object. Common use for im is to add a colorbar.
        The image is oriented in the map orientation (north is up, south is down,
        east is right, and west is left), contrary to the camera orientation where
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
    | import asilib
    |
    | time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
    | movie_generator = asilib.animate_fisheye_generator('THEMIS', 'FSMI', time_range)
    |
    | for image_time, image, im, ax in movie_generator:
    |       # The code that modifies each image here.
    |       pass
    |
    | print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')
    """

    return