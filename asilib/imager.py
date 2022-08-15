import datetime
import dateutil.parser
import pathlib
import numbers
import inspect
import importlib
import shutil
import copy
from typing import List, Tuple, Generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import ffmpeg

import asilib
import asilib.utils as utils


class Imager:
    """
    Manage and analyze the ASI images and time stamps, as well as the associated metadata
    and skymaps.

    Parameters
    ----------
    data: dict
        Dictionary containing image data in two main formats (sets of keys): the first is with
        two keys ``time`` and ``image`` for a single image; or ``start_time``, ``end_time``,
        ``path``, and ``loader`` keys.
    meta: dict
        The ASI metadata that must have the following keys ``array``, ``location``, ``lat``,
        ``lon``, ``alt``, and ``cadence``. The ``cadence`` units are seconds and ``alt`` is
        kilometers.
    skymap: dict
        The data to map each pixel to azimuth and elevation (``az``, ``el``) and latitude,
        longitude, altitude (``lat``, ``lon``, ``alt``) coordinates.
    plot_settings: dict
        An optional dictionary containing  ```color_bounds```, ```color_map```,
        and ```color_norm``` keys. The ```color_bounds``` can be either a function takes in
        an image and returns the lower and upper bound numbers, or a len 2 tuple or list.
        The ```color_map``` key must be a valid matplotlib colormap. And lastly, ```color_norm```
        must be either ```lin``` for linear or ```log``` for logarithmic color scale.
    """

    def __init__(self, data: dict, meta: dict, skymap: dict, plot_settings: dict = {}) -> None:
        self._data = {k.lower(): v for k, v in data.items()}
        self.meta = {k.lower(): v for k, v in meta.items()}
        self.skymap = {k.lower(): v for k, v in skymap.items()}
        self.plot_settings = {k.lower(): v for k, v in plot_settings.items()}
        # self._validate_inputs()  # TODO-Validation: Add a small-scale validations to each method.
        return

    def plot_fisheye(
        self,
        time: utils._time_type = None,
        ax: plt.Axes = None,
        label: bool = True,
        color_map: str = None,
        color_bounds: List[float] = None,
        color_norm: str = None,
        azel_contours: bool = False,
        azel_contour_color: str = 'yellow',
    ) -> Tuple[plt.Axes, matplotlib.image.AxesImage]:
        """
        Plots one fisheye image, oriented with North on the top, and East on the right of the image.

        Parameters
        ----------
        time: datetime.datetime or str
            The date and time to download of the data. If str, ``time`` must be in the
            ISO 8601 standard.
        ax: plt.Axes
            The subplot to plot the image on. If None this method will create one.
        label: bool
            Flag to add the "asi_array_code/location_code/image_time" text to the plot.
        color_map: str
            The matplotlib colormap to use. By default will use a black-white colormap.
            For more information See https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
        color_bounds: List[float]
            The lower and upper values of the color scale. The default is: low=1st_quartile and
            high=min(3rd_quartile, 10*1st_quartile). This range works well for most cases.
        color_norm: str
            Set the 'lin' linear or 'log' logarithmic color normalization.
        azel_contours: bool
            Superpose azimuth and elevation contours on or off.
        azel_contour_color: str
            The color of the azimuth and elevation contours.

        Returns
        -------
        ax: plt.Axes
            The subplot object to modify the axis, labels, etc.
        im: plt.AxesImage
            The plt.imshow image object. Common use for im is to add a colorbar.
            The image is oriented in the map orientation (north is up, south is down,
            west is right, and east is left). Set azel_contours=True to confirm.

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
        | ax, im = asilib.plot_fisheye('THEMIS', 'RANK', time,
        |     color_norm='log')
        |
        | plt.colorbar(im)
        | ax.axis('off')
        | plt.show()
        """
        if ax is None:
            _, ax = plt.subplots()

        if time is not None:
            self = self.__getitem__(time)
            time, image = self.data()
        elif 'time' in self._data.keys():
            time, image = self.data()
        else:
            raise ValueError('I am not supposed to get here. Congrats! You found a bug!')

        color_bounds, color_map, color_norm = self._plot_params(
            image, color_bounds, color_map, color_norm
        )

        im = ax.imshow(image[:, :], cmap=color_map, norm=color_norm, origin="lower")
        if label:
            self._add_label(time, ax)
        if azel_contours:
            self._add_azel_contours(ax, color=azel_contour_color)
        return ax, im

    def animate_fisheye(self, **kwargs) -> None:
        """
        A wrapper for the ```animate_fisheye_gen()``` method that animates a series of
        fisheye images. Any kwargs are passed directly into ```animate_fisheye_gen()```.

        Parameters
        ----------
        ax: plt.Axes
            The optional subplot that will be drawn on.
        label: bool
            Flag to add the "asi_array_code/location_code/image_time" text to the plot.
        color_map: str
            The matplotlib colormap to use. By default will use a black-white colormap.
            For more information See https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
        color_bounds: List[float]
            The lower and upper values of the color scale. The default is: low=1st_quartile and
            high=min(3rd_quartile, 10*1st_quartile). This range works well for most cases.
        color_norm: str
            Set the 'lin' linear or 'log' logarithmic color normalization.
        azel_contours: bool
            Superpose azimuth and elevation contours on or off.
        azel_contour_color: str
            The color of the azimuth and elevation contours.
        movie_container: str
            The movie container: mp4 has better compression but avi was determined
            to be the official container for preserving digital video by the
            National Archives and Records Administration.
        ffmpeg_params: dict
            The additional/overwitten ffmpeg output parameters. The default parameters are:
            framerate=10, crf=25, vcodec=libx264, pix_fmt=yuv420p, preset=slower.
        overwrite: bool
            If true, the output will be overwritten automatically. If false it will
            prompt the user to answer y/n.

        Returns
        -------
        Add

        Raises
        ------
        Add

        Example
        -------
        Add
        """

        # ax = kwargs.get('ax', None)
        # if ax is None:
        #     _, ax = plt.subplots(figsize=(6, 6))
        #     kwargs['ax'] = ax
        #     plt.tight_layout()

        movie_generator = self.animate_fisheye_gen(**kwargs)

        for _ in movie_generator:
            pass
        return

    def animate_fisheye_gen(
        self,
        ax: plt.Axes = None,
        label: bool = True,
        color_map: str = None,
        color_bounds: List[float] = None,
        color_norm: str = None,
        azel_contours: bool = False,
        azel_contour_color: str = 'yellow',
        movie_container: str = 'mp4',
        ffmpeg_params={},
        overwrite: bool = False,
    ) -> Generator[
        Tuple[datetime.datetime, np.ndarray, plt.Axes, matplotlib.image.AxesImage], None, None
    ]:
        """
        A generator method that animates a series of fisheye images.

        A generator behaves like an iterator in that it plots one fisheye image
        at a time and yields (similar to returns) the image. You can modify, or add
        content to the image (such as a spacecraft position). Then, once the iteration
        is complete, this method stitches the images into an animation. See the examples
        below and in the examples page for use cases. The ```animate_fisheye()``` method
        takes care of the iteration.

        Parameters
        ----------
        ax: plt.Axes
            The optional subplot that will be drawn on.
        label: bool
            Flag to add the "asi_array_code/location_code/image_time" text to the plot.
        color_map: str
            The matplotlib colormap to use. By default will use a black-white colormap.
            For more information See https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
        color_bounds: List[float]
            The lower and upper values of the color scale. The default is: low=1st_quartile and
            high=min(3rd_quartile, 10*1st_quartile). This range works well for most cases.
        color_norm: str
            Set the 'lin' linear or 'log' logarithmic color normalization.
        azel_contours: bool
            Superpose azimuth and elevation contours on or off.
        azel_contour_color: str
            The color of the azimuth and elevation contours.
        movie_container: str
            The movie container: mp4 has better compression but avi was determined
            to be the official container for preserving digital video by the
            National Archives and Records Administration.
        ffmpeg_params: dict
            The additional/overwitten ffmpeg output parameters. The default parameters are:
            framerate=10, crf=25, vcodec=libx264, pix_fmt=yuv420p, preset=slower.
        overwrite: bool
            If true, the data will be downloaded again and the output animation overwritten.
            If false it will prompt the user to answer y/n.

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
            west is right, and east is left). Set azel_contours=True to confirm.

        Raises
        ------
        NotImplementedError
            If the colormap is unspecified ('auto' by default) and the
            auto colormap is undefined for an ASI array.
        ValueError
            If the color_norm kwarg is not "log" or "lin".

        Example
        -------
        TODO: REDO
        | from datetime import datetime
        |
        | import asilib
        |
        | time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
        | movie_generator = asilib.animate_fisheye_generator('THEMIS', 'FSMI', time_range)
        |
        | for image_time, image, ax, im in movie_generator:
        |       # The code that modifies each image here.
        |       pass
        |
        | print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')
        """
        if ax is None:
            _, ax = plt.subplots()

        # Create the movie directory inside asilib.config['ASI_DATA_DIR'] if it does
        # not exist.
        image_save_dir = pathlib.Path(
            asilib.config['ASI_DATA_DIR'],
            'animations',
            'images',
            f'{self._data["time_range"][0].strftime("%Y%m%d_%H%M%S")}_{self.meta["array"].lower()}_'
            f'{self.meta["location"].lower()}_fisheye',
        )
        movie_save_name = (
            f'{self._data["time_range"][0].strftime("%Y%m%d_%H%M%S")}_'
            f'{self._data["time_range"][-1].strftime("%H%M%S")}_'
            f'{self.meta["array"].lower()}_{self.meta["location"].lower()}_fisheye.{movie_container}'
        )
        movie_save_path = image_save_dir.parents[1] / movie_save_name
        # If the image directory exists we need to first remove all of the images to avoid
        # animating images from different method calls.
        if image_save_dir.is_dir():
            shutil.rmtree(image_save_dir)
        image_save_dir.mkdir(parents=True)

        image_paths = []

        for i, (image_time, image) in utils.progressbar(
            enumerate(self.__iter__()), self._estimate_n_times(), movie_save_name
        ):
            # # If the image is all 0s we have a bad image and we need to skip it.
            # if np.all(image == 0):
            #     continue
            ax.clear()
            ax.axis('off')
            # Use an underscore so the original method parameters are not overwritten.
            _, _color_map, _color_norm = self._plot_params(
                image, color_bounds, color_map, color_norm
            )

            im = ax.imshow(image, cmap=_color_map, norm=_color_norm, origin='lower')
            if label:
                self._add_label(image_time, ax)

            if azel_contours:
                self._add_azel_contours(ax, color=azel_contour_color)

            # Give the user the control of the subplot, image object, and return the image time
            # so that they can manipulate the image to add, for example, the satellite track.
            yield image_time, image, ax, im

            # Save the plot before the next iteration.
            save_name = f'{str(i).zfill(6)}.png'
            plt.savefig(image_save_dir / save_name)
            image_paths.append(image_save_dir / save_name)

        self._create_animation(image_paths, movie_save_path, ffmpeg_params, overwrite)
        return

    # def plot_map(
    #     self,
    #     time: utils._time_type=None,
    #     ax: plt.Axes = None,
    #     map_style: str = 'green',
    #     color_map: str = 'auto',
    #     min_elevation: float = 10,
    #     norm: bool = True,
    #     asi_label: bool = True,
    #     color_bounds: List[float] = None,
    #     color_norm: str = 'log',
    #     pcolormesh_kwargs: dict = {},
    #     ) -> plt.Axes:
    #     """
    #     Projects the ASI images to a map at an altitude defined in the skymap calibration file.

    #     Parameters
    #     ----------
    #     asi_array_code: str
    #         The imager array name, i.e. ``THEMIS`` or ``REGO``.
    #     location_code: str
    #         The ASI station code, i.e. ``ATHA``
    #     time: datetime.datetime or str
    #         The date and time to download of the data. If str, ``time`` must be in the
    #         ISO 8601 standard.
    #     map_alt: float
    #         The altitude in kilometers to project to. Must be an altitude value
    #         in the skymap calibration.
    #     time_thresh_s: float
    #         The maximum allowable time difference between ``time`` and an ASI time stamp.
    #         This is relevant only when ``time`` is specified.
    #     ax: plt.Axes
    #         The subplot to plot the image on. If None, this function will
    #         create one.
    #     map_style: str
    #         If ax is None, this kwarg toggles between two predefined map styles:
    #         'green' map has blue oceans and green land, while the `white` map
    #         has white oceans and land with black coastlines.
    #     color_map: str
    #         The matplotlib colormap to use. If 'auto', will default to a
    #         black-red colormap for REGO and black-white colormap for THEMIS.
    #         For more information See https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    #     min_elevation: float
    #         Masks the pixels below min_elevation degrees.
    #     norm: bool
    #         If True, normalizes the image array to 0-1. This is useful when
    #         mapping images from multiple imagers.
    #     asi_label: bool
    #         Annotates the map with the ASI code in the center of the image.
    #     color_bounds: List[float] or None
    #         The lower and upper values of the color scale. If None, will
    #         automatically set it to low=1st_quartile and
    #         high=min(3rd_quartile, 10*1st_quartile)
    #     color_norm: str
    #         Sets the 'lin' linear or 'log' logarithmic color normalization.
    #     pcolormesh_kwargs: dict
    #         A dictionary of keyword arguments (kwargs) to pass directly into
    #         plt.pcolormesh. One use of this parameter is to change the colormap. For example,
    #         pcolormesh_kwargs = {'cmap':'tu}

    #     Returns
    #     -------
    #     image_time: datetime.datetime
    #         The time of the current image.
    #     image: np.array
    #         The 2d ASI image corresponding to image_time.
    #     skyamp: dict
    #         The skymap calibration for that ASI.
    #     ax: plt.Axes
    #         The subplot object to modify the axis, labels, etc.
    #     p: plt.AxesImage
    #         The plt.pcolormesh image object. Common use for p is to add a colorbar.
    #     """
    #     # Halt here if cartopy is not installed.
    #     if importlib.util.find_spec("cartopy") is None:
    #         raise ImportError(
    #             "cartopy can't be imported. This is a required dependency for asilib.plot_map()"
    #             " that must be installed separately. See https://scitools.org.uk/cartopy/docs/latest/installing.html"
    #             " and https://aurora-asi-lib.readthedocs.io/en/latest/installation.html."
    #         )

    #     image_time, image = load.load_image(
    #         asi_array_code, location_code, time=time, time_thresh_s=time_thresh_s
    #     )
    #     skymap = load.load_skymap(asi_array_code, location_code, time)

    #     # Check that the map_alt is in the skymap calibration data.
    #     assert (
    #         map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
    #     ), f'{map_alt} km is not in skymap calibration altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
    #     alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]

    #     lon_map, lat_map, image = self._mask_low_horizon(
    #         skymap['FULL_MAP_LONGITUDE'][alt_index, :, :],
    #         skymap['FULL_MAP_LATITUDE'][alt_index, :, :],
    #         skymap['FULL_ELEVATION'],
    #         min_elevation,
    #         image=image
    #     )

    #     if norm:
    #         image /= np.nanmax(image)

    #     # Set up the plot parameters
    #     if ax is None:
    #         ax = self.create_cartopy_map(map_style=map_style)

    #     if color_bounds is None:
    #         color_bounds = asilib.plot.utils.get_color_bounds(image)
    #     _color_map = asilib.plot.utils.get_color_map(asi_array_code, color_map)
    #     _norm = asilib.plot.utils.get_color_norm(color_norm, color_bounds)

    #     p = self._pcolormesh_nan(
    #         lon_map,
    #         lat_map,
    #         image,
    #         ax,
    #         cmap=_color_map,
    #         norm=_norm,
    #         pcolormesh_kwargs=pcolormesh_kwargs,
    #     )

    #     # if asi_label:
    #     #     ax.text(
    #     #         skymap['SITE_MAP_LONGITUDE'],
    #     #         skymap['SITE_MAP_LATITUDE'],
    #     #         location_code.upper(),
    #     #         color='r',
    #     #         transform=ccrs.PlateCarree(),
    #     #         va='center',
    #     #         ha='center',
    #     #     )
    #     return ax, p

    def _validate_inputs(self):
        """
        Check that the correct keys were passed for either
        single or multiple images, or None.
        """
        single_image_keys = ['time', 'image']
        multiple_images_keys = ['start_time', 'end_time', 'path', 'loader', 'time_range']

        if all([key in self._data.keys() for key in single_image_keys]):
            self._data['time'] = utils.validate_time(self._data['time'])

            # image is None means that were in a conjunction finding mode when Conjunction
            # only needs the data in the skymap and the meta dictionaries.
            if self._data['image'] is not None:
                self._data['image'] = np.array(self._data['image'])

                if len(self._data['image'].shape) != 2:
                    raise ValueError(
                        f'The image shape must be 2. Got {len(self._data["image"].shape)}'
                    )

        elif all([key in self._data.keys() for key in multiple_images_keys]):
            self._data['start_time'] = np.array(
                [utils.validate_time(t_i) for t_i in self._data['start_time']]
            )
            self._data['end_time'] = np.array(
                [utils.validate_time(t_i) for t_i in self._data['end_time']]
            )
            self._data['time_range'] = np.array(
                [utils.validate_time(t_i) for t_i in self._data['time_range']]
            )
        elif len(self._data.keys()) == 0:
            pass  # The case when the Image instance is only used for conjunctions.
        else:
            raise AttributeError(
                'The Imager "data" dictionary did not contain either of the two sets of '
                f'acceptable keys: {single_image_keys} or {multiple_images_keys}. Got '
                f'{self._data.keys()}.'
            )

        # TODO: Add the meta and skymap checks
        # Check that the skyamp lons are -180 - 180.
        return

    def __getitem__(self, _slice):
        """
        A generator to return images and time stamps one by one.

        Parameters
        ----------
        time_range: list of datetime or string-formatted times.
            The bounding time range.

        Yields
        ------
        asilib.Imager:
            A sliced version of the Imager.
        """
        # Deal with the [time_i:time_f] slice logic.
        if isinstance(_slice, slice):
            if isinstance(_slice.start, str):
                start_time = dateutil.parser.parse(_slice.start)
            elif isinstance(_slice.start, (datetime.datetime, pd.Timestamp)):
                start_time = _slice.start
            elif _slice.start is None:
                start_time = self._data['time_range'][0]
            else:
                raise ValueError(
                    f'The start index can only be a time object, string, or None. '
                    f'{_slice.start} is unsupported'
                )

            if isinstance(_slice.stop, str):
                end_time = dateutil.parser.parse(_slice.stop)
            elif isinstance(_slice.stop, (datetime.datetime, pd.Timestamp)):
                end_time = _slice.stop
            elif _slice.stop is None:
                end_time = self._data['time_range'][1]
            else:
                raise ValueError(
                    f'The start index can only be a time object, string, or None. '
                    f'{_slice.stop} is unsupported'
                )

            if _slice.step is not None:
                raise NotImplementedError

            # TODO: Fix what files are loaded.
            # start_file_i = np.where(start_time >= np.array(self._data['start_time']))[0][-1]
            # end_file_i = np.where(end_time <= np.array(self._data['end_time']))[0][0]

            new_data = copy.copy(self._data)
            new_data['time_range'] = [start_time, end_time]
            # new_data['start_time'] = new_data['start_time'][start_file_i:end_file_i]
            # new_data['end_time'] = new_data['end_time'][start_file_i:end_file_i]
            # new_data['path'] = new_data['path'][start_file_i:end_file_i]
            files = np.where(
                (start_time <= np.array(self._data['end_time']))
                & (end_time >= np.array(self._data['start_time']))
            )[0]
            new_data['start_time'] = np.array(new_data['start_time'])[files]
            new_data['end_time'] = np.array(new_data['end_time'])[files]
            new_data['path'] = np.array(new_data['path'])[files]
            new_meta = copy.copy(self.meta)
            new_skymap = copy.copy(self.skymap)
            new_plot_settings = copy.copy(self.plot_settings)

            cls = type(self)
            return cls(new_data, new_meta, new_skymap, plot_settings=new_plot_settings)

        # Deal with the [time] slice logic.
        elif isinstance(_slice, (str, datetime.datetime, pd.Timestamp)):
            if isinstance(_slice, str):
                slice_time = dateutil.parser.parse(_slice)
            else:
                slice_time = _slice

            if 'start_time' in self._data.keys():  # First find the correct file
                file_index = np.where(
                    (slice_time >= np.array(self._data['start_time']))
                    & (slice_time < np.array(self._data['end_time']))
                )[0]
                assert len(file_index) == 1
                file_index = file_index[0]
                file_path = self._data['path'][file_index]
                file_times, file_images = self._data['loader'](file_path)

                # Then find the correct time stamp
                image_index = np.argmin(
                    np.abs([(slice_time - t_i).total_seconds() for t_i in file_times])
                )
                if (
                    np.abs((slice_time - file_times[image_index]).total_seconds())
                    > self.meta['cadence']
                ):
                    raise IndexError(
                        f'Cannot find a time stamp within of {self.meta["cadence"]} s of '
                        f'{slice_time}. Closest time stamp is {file_times[image_index]}.'
                    )
                new_data = {
                    'time': file_times[image_index],
                    'image': file_images[
                        image_index, ...
                    ],  # Ellipsis to return all other dimensions.
                }
            # Edge case when [time] is within the imager cadence of self._image['time']
            elif 'time' in self._data.keys():
                if np.abs((slice_time - self._data['time']).total_seconds()) > self.meta['cadence']:
                    raise ValueError(
                        f'Imager contains only one image at time={self._data["time"]} '
                        f'but was sliced with time={slice_time}.'
                    )
                new_data = copy.copy(self._data)

            new_meta = copy.copy(self.meta)
            new_skymap = copy.copy(self.skymap)
            new_plot_settings = copy.copy(self.plot_settings)

            cls = type(self)
            return cls(new_data, new_meta, new_skymap, plot_settings=new_plot_settings)

        elif isinstance(_slice, tuple):
            raise NotImplementedError(
                'At this time Imager does not support ' 'multi-dimensional indexing.'
            )
        return

    def __iter__(self):
        """
        Iterate over individual timestamps and images.

        Parameters
        ----------
        None

        Yields
        ------
        datetime.datetime
            timestamp
        np.array
            image.
        """
        self._loader_is_gen = inspect.isgeneratorfunction(self._data['loader'])

        if 'time_range' not in self._data.keys():
            raise KeyError('Imager was not instantiated with a time_range.')

        for path in self._data['path']:
            # Check is the loader function is a generator. If not, asilib
            # will load one image file at a time and assume that opening one file
            # won't overwhelm the PC's memory. If loader is a generator,
            # on the other hand, we need to loop over every chunk of data
            # yielded by the generator and over the timestamps in each chunk.
            if not self._loader_is_gen:
                times, images = self._data['loader'](path)

                for time, image in zip(times, images):
                    if (time < self._data['time_range'][0]) or (time > self._data['time_range'][1]):
                        continue
                    yield time, image
            else:
                gen = self._data['loader'](path)

                for time_chunk, image_chunk in gen:
                    for time, image in zip(time_chunk, image_chunk):
                        if (time < self._data['time_range'][0]) or (
                            time > self._data['time_range'][1]
                        ):
                            continue
                        yield time, image

    def _estimate_n_times(self):
        """
        Estimate the maximum number of time stamps for the Imager's time range..
        """
        n_sec = (self._data['time_range'][1] - self._data['time_range'][0]).total_seconds()
        # +2 is for when time_range is a bit bigger than an integer number of samples
        return int(n_sec / self.meta['cadence']) + 2

    def data(self, which='time'):
        """
        Returns the full data in the specified time_range.

        Warning: this does not respect how much memory you have avaliable; a long enough
        time_range can completely use up your memory and then your computer will become
        unresponsive or sluggish (if you have swap memory).

        Parameters
        ----------
        which: str
            Which data to return. Can be ```time``` to return only time stamps, ```image```
            to return only images, and ```both``` to return both time stamps and images.
            This kwarg is only used in multi-image Imager objects.
        """
        if 'time' in which.lower():
            times_flag = True
            images_flag = False
        elif 'image' in which.lower():
            times_flag = False
            images_flag = True
        elif 'both' in which.lower():
            times_flag = True
            images_flag = True
        else:
            raise ValueError("'which' kwarg must be 'time', 'image', or 'both'.")

        if ('time_range' in self._data.keys()) and (self._data['time_range'] is not None):
            if times_flag:
                _times = np.nan * np.zeros(self._estimate_n_times(), dtype=object)
            else:
                _times = None
            if images_flag:
                _images = np.nan * np.zeros((self._estimate_n_times(), *self.meta['resolution']))
            else:
                _images = None

            for i, (time, image) in enumerate(self.__iter__()):
                if times_flag:
                    _times[i] = time
                if images_flag:
                    _images[i, ...] = image

            # Filter out missing data
            if times_flag:
                # np.nan does not work with an array of datetime objects.
                nonempty_times = [isinstance(t_i, datetime.datetime) for t_i in _times]
                _times = _times[nonempty_times]
            if images_flag:
                nonempty_images = np.where(~np.isnan(_images[:, 0, 0]))[0]
                _images = _images[nonempty_images, ...]
            return _times, _images

        elif ('time' in self._data.keys()) and (self._data['time'] is not None):
            return self._data['time'], self._data['image']

    def __str__(self) -> str:
        if ('time' in self._data.keys()) and (self._data['time'] is not None):
            s = (
                f'A {self.meta["array"]}-{self.meta["location"]} Imager. '
                f'time={self._data["time"]}'
            )
        elif ('time_range' in self._data.keys()) and (self._data['time_range'] is not None):
            s = (
                f'A {self.meta["array"]}-{self.meta["location"]} Imager. '
                f'time_range={self._data["time_range"]}'
            )
        return s

    def _plot_params(self, image, color_bounds, color_map, color_norm):
        """
        Sets the plot color bounds, map style, and normalization values. In order of precedence,
        these kwargs are checked first, and then self.plot_settings. If these values are
        unset in either place, defaults are returned.
        """
        # Check the plot_settings dict and fall back to a default if user did not specify
        # color_bounds in the method call.
        if color_bounds is None:
            if 'color_bounds' in self.plot_settings.keys():
                # Is it a function?
                if callable(self.plot_settings['color_bounds']):
                    color_bounds = self.plot_settings['color_bounds'](image)
                else:
                    color_bounds = self.plot_settings['color_bounds']
            else:
                lower, upper = np.quantile(image, (0.25, 0.98))
                color_bounds = [lower, np.min([upper, lower * 10])]
        else:  # color_bounds is specified in the method call.
            if callable(color_bounds):
                color_bounds = color_bounds(image)
            else:
                color_bounds = color_bounds

        if color_map is None:
            if 'color_map' in self.plot_settings.keys():
                color_map = self.plot_settings['color_map']
            else:
                color_map = 'Greys_r'
        else:
            color_map = color_map

        if color_norm is None:
            if 'color_norm' in self.plot_settings.keys():
                color_norm = self.plot_settings['color_norm']
            else:
                color_norm = 'log'
        else:
            color_norm = color_norm

        if color_norm == 'log':
            color_norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
        elif color_norm == 'lin':
            color_norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
        else:
            raise ValueError('color_norm must be either "log" or "lin".')

        return color_bounds, color_map, color_norm

    def _add_azel_contours(self, ax: plt.Axes, color: str = 'yellow') -> None:
        """
        Adds contours of azimuth and elevation to the movie image.

        Parameters
        ----------
        ax: plt.Axes
            The subplot to draw on.
        color: str (optional)
            The contour color.
        """
        az_contours = ax.contour(
            self.skymap['az'],
            colors=color,
            linestyles='dotted',
            levels=np.arange(0, 361, 90),
            alpha=1,
        )
        el_contours = ax.contour(
            self.skymap['el'],
            colors=color,
            linestyles='dotted',
            levels=np.arange(1, 91, 30),
            alpha=1,
        )
        plt.clabel(az_contours, inline=True, fontsize=12)
        plt.clabel(el_contours, inline=True, fontsize=12, rightside_up=True)
        return

    def _add_label(self, image_time, ax: plt.Axes):
        """
        Adds a ASI array-location-time label to the image
        """
        if self.meta['cadence'] > 1:
            time_label = image_time.strftime('%Y-%m-%d %T')
        else:
            time_label = image_time.strftime('%Y-%m-%d %T:%f')
        ax.text(
            0,
            0,
            (f"{self.meta['array'].upper()}/{self.meta['location'].upper()}\n" f"{time_label}"),
            va='bottom',
            transform=ax.transAxes,
            color='white',
        )
        return

    def _mask_low_horizon(self, lon_map, lat_map, el_map, min_elevation, image=None):
        """
        Mask the image, skymap['lon'], skymap['lat'] arrays with np.nans
        where the skymap['el'] < min_elevation or is nan.
        """
        idh = np.where(np.isnan(el_map) | (el_map < min_elevation))
        # Copy variables to not modify original np.arrays.
        lon_map_copy = lon_map.copy()
        lat_map_copy = lat_map.copy()
        lon_map_copy[idh] = np.nan
        lat_map_copy[idh] = np.nan

        if image is not None:
            image_copy = image.copy()
            image_copy = image_copy.astype(float)  # Can't mask image unless it is a float array.
            image_copy[idh] = np.nan
        else:
            image_copy = None

        if (lon_map.shape[0] == el_map.shape[0] + 1) and (lon_map.shape[1] == el_map.shape[1] + 1):
            # TODO: This is REGO/THEMIS specific. Remove here and add this to the themis() function?
            # For some reason the THEMIS & REGO lat/lon_map arrays are one size larger than el_map, so
            # here we mask the boundary indices in el_map by adding 1 to both the rows
            # and columns.
            idh_boundary_bottom = (
                idh[0] + 1,
                idh[1],
            )  # idh is a tuple so we have to create a new one.
            idh_boundary_right = (idh[0], idh[1] + 1)
            lon_map_copy[idh_boundary_bottom] = np.nan
            lat_map_copy[idh_boundary_bottom] = np.nan
            lon_map_copy[idh_boundary_right] = np.nan
            lat_map_copy[idh_boundary_right] = np.nan
        return lon_map_copy, lat_map_copy, image_copy

    def _create_animation(
        self,
        image_paths: List[pathlib.Path],
        movie_save_path: pathlib.Path,
        ffmpeg_params: dict,
        overwrite: bool,
    ):
        """
        Helper function to stich together images into an animation using ffmpeg.

        Parameters
        ----------
        image_save_dir: pathlib.Path
            The directory where the individual images are saved.
        ffmpeg_params: dict
            The additional/overwitten ffmpeg output parameters. The default parameters are:
            framerate=10, crf=25, vcodec=libx264, pix_fmt=yuv420p, preset=slower.
        movie_file_name: str
            The movie file name.
        overwrite: bool
            Overwrite the movie.
        """
        _ffmpeg_params = {
            'framerate': 10,
            'crf': 25,
            'vcodec': 'libx264',
            'pix_fmt': 'yuv420p',
            'preset': 'slower',
        }
        # Add or change the ffmpeg_params's key:values with ffmpeg_params
        _ffmpeg_params.update(ffmpeg_params)

        try:
            movie_obj = ffmpeg.input(
                str(image_paths[0].parent / "%06d.png"),
                pattern_type='sequence',
                # Pop so it won't be passed into movie_obj.output().
                framerate=_ffmpeg_params.pop('framerate'),
            )
            movie_obj.output(str(movie_save_path), **_ffmpeg_params).run(overwrite_output=overwrite)
            print(f'{movie_save_path.name} animation saved to {movie_save_path.parent}')
        except FileNotFoundError as err:
            if '[WinError 2] The system cannot find the file specified' in str(err):
                raise FileNotFoundError("Windows doesn't have ffmpeg installed.") from err
        return

    # def __repr__(self):
    #     params = (f'{self.asi_array}, {self.asi_location}, '
    #               f'{self.time}, {self.image}, skymap={self.skymap}')
    #     return f'{self.__class__.__qualname__}(' + params + ')'

    def _pcolormesh_nan(
        self,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        ax,
        cmap=None,
        norm=None,
        pcolormesh_kwargs={},
    ):
        """
        # TODO: Remove when done (move into Imager)
        Since pcolormesh cant handle nan lat/lon grid values, we will compress them to the
        nearest valid lat/lon grid. There are two main steps:

        1) All nan values to the left of the first valid value are
        reassigned to the first valid value. Likewise, all nan values to the
        right of the last valid value are reassigned to it.

        2) All nan-filled rows above the first valid row are assigned to the
        maximum value in the first row, likewise for the bottom rows.

        Essentially this is a reassignment (or a compression) of all nan values in the periphery
        to the valid grid values in the center.

        Function taken from Michael, scivision @ GitHub.:
        https://github.com/scivision/python-matlab-examples/blob/0dd8129bda8f0ec2c46dae734d8e43628346388c/PlotPcolor/pcolormesh_NaN.py
        """
        # mask is True when lat and lon grid values are not nan.
        mask = np.isfinite(x) & np.isfinite(y)
        top = None
        bottom = None

        for i, m in enumerate(mask):
            # A common use for nonzero is to find the indices of
            # an array, where a condition is True (not nan or inf)
            good = m.nonzero()[0]

            if good.size == 0:  # Skip row is all columns are nans.
                continue
            # First row that has at least 1 valid value.
            elif top is None:
                top = i
            # Bottom row that has at least 1 value value. All indices in between top and bottom
            else:
                bottom = i

            # Reassign all lat/lon columns after good[-1] (all nans) to good[-1].
            x[i, good[-1] :] = x[i, good[-1]]
            y[i, good[-1] :] = y[i, good[-1]]
            # Reassign all lat/lon columns before good[0] (all nans) to good[0].
            x[i, : good[0]] = x[i, good[0]]
            y[i, : good[0]] = y[i, good[0]]

        # Reassign all of the fully invalid lat/lon rows above top to the the max lat/lon value.
        x[:top, :] = np.nanmax(x[top, :])
        y[:top, :] = np.nanmax(y[top, :])
        # Same, but for the rows below bottom.
        x[bottom:, :] = np.nanmax(x[bottom, :])
        y[bottom:, :] = np.nanmax(y[bottom, :])

        # old masked c code: np.ma.masked_where(~mask[:-1, :-1], c)[::-1, ::-1]
        p = ax.pcolormesh(
            x,
            y,
            c,
            cmap=cmap,
            shading='flat',
            norm=norm,
            **pcolormesh_kwargs,
        )
        return p
