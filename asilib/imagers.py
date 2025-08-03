"""
The Imagers class handles multiple Imager objects and coordinates plotting and animating multiple
fisheye lens and mapped images. The mapped images are also called mosaics.
"""

from typing import Tuple, List, Union, Generator, Callable
from collections import namedtuple
import pathlib
from datetime import datetime, timedelta
import shutil
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.collections
import matplotlib.patches
import scipy.interpolate
from scipy.spatial import cKDTree
try:
    import IRBEM
    _irbem_imported = True
except ImportError:
    _irbem_imported = False

from asilib.imager import Imager, _haversine, Skymap_Cleaner
import asilib.map
import asilib.utils


class Imagers:
    """
    Plot and animate multiple :py:meth:`~asilib.imager.Imager` objects.

    .. warning::

        This class is in development and not all methods are implemented/matured.

    Parameters
    ----------
    imagers: Tuple
        The Imager objects to plot and animate. 
    iter_tol: float
        The allowable time tolerance, in units of time_tol*imager_cadence, for imagers to be 
        considered synchronized. Adjusting this kwarg is useful if the imager has missing 
        data and you need to animate a mosaic.
    """
    def __init__(self, imagers:Tuple[Imager], iter_tol:float=2) -> None:
        self.imagers = imagers
        # Wrap self.imagers in a tuple if the user passes in a single Imager object.
        if isinstance(self.imagers, Imager):
            self.imagers = (self.imagers, )
        self.iter_tol = iter_tol
        return
    
    def plot_fisheye(self, ax:Tuple[plt.Axes], **kwargs):
        """
        Plots one fisheye image in each subplot, oriented with North on the top, and East on the 
        left of the image.

        Parameters
        ----------
        ax: Tuple[plt.Axes]
            Subplots corresponding to each fisheye lens image.
        kwargs: dict
            Keyword arguments directly passed into each :py:meth:`~asilib.Imager.plot_fisheye`
            method.

        Example
        -------
        >>> from datetime import datetime
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> import asilib
        >>> import asilib.asi
        >>> 
        >>> time = datetime(2007, 3, 13, 5, 8, 45)
        >>> location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
        >>> 
        >>> fig, ax = plt.subplots(1, len(location_codes), figsize=(12, 3.5))
        >>> 
        >>> _imagers = []
        >>> 
        >>> for location_code in location_codes:
        >>>     _imagers.append(asilib.asi.themis(location_code, time=time))
        >>> 
        >>> for ax_i in ax:
        >>>     ax_i.axis('off')
        >>> 
        >>> asis = asilib.Imagers(_imagers)
        >>> asis.plot_fisheye(ax=ax)
        >>> 
        >>> plt.suptitle('Donovan et al. 2008 | First breakup of an auroral arc')
        >>> plt.tight_layout()
        >>> plt.show()
        """
        if not isinstance(ax, (list, tuple, np.ndarray)):
            ax = (ax,)
        assert len(ax) == len(self.imagers), 'Number of subplots must equal the number of imagers.'

        for ax_i, imager_i in zip(ax, self.imagers):
            imager_i.plot_fisheye(ax=ax_i, **kwargs)
        return
    
    def plot_map(self, overlap=False, **kwargs):
        """
        Projects multiple ASI images onto a map at an altitude that is defined in the skymap 
        calibration file.

        Parameters
        ----------
        overlap: bool
            If True, pixels that overlap between imager FOV's are overplotted such that only the 
            final imager's pixels are shown.
        kwargs: dict
            Keyword arguments directly passed into each :py:meth:`~asilib.Imager.plot_map`
            method.

        Example
        -------
        >>> from datetime import datetime
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> import asilib
        >>> import asilib.map
        >>> import asilib.asi
        >>> 
        >>> time = datetime(2007, 3, 13, 5, 8, 45)
        >>> location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
        >>> map_alt = 110
        >>> min_elevation = 2
        >>> 
        >>> ax = asilib.map.create_map(lon_bounds=(-140, -60), lat_bounds=(40, 82))
        >>> 
        >>> _imagers = []
        >>> 
        >>> for location_code in location_codes:
        >>>     _imagers.append(asilib.asi.themis(location_code, time=time, alt=map_alt))
        >>> 
        >>> asis = asilib.Imagers(_imagers)
        >>> asis.plot_map(ax=ax, overlap=False, min_elevation=min_elevation)
        >>> 
        >>> ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
        >>> plt.show()
        """
        if 'ax' not in kwargs:
            kwargs['ax'] = asilib.map.create_map()

        # If the kwarg is not specified, or is None, then it should just be None for all imagers.
        # Below will be True if the kwarg['color_bounds'] is set as None, or if the key is missing.
        if kwargs.get('color_bounds', None) is None:
            color_bounds = np.full(len(self.imagers), None)
        elif len(np.array(kwargs['color_bounds']).shape) == 1:  # Same color_bounds for all imagers.
            color_bounds = np.repeat(
                np.array(kwargs['color_bounds'])[:, np.newaxis], 
                len(self.imagers), 
                axis=-1
                ).T
        elif len(np.array(kwargs['color_bounds']).shape) == 2:
            assert np.array(kwargs['color_bounds']).shape == (len(self.imagers), 2), (
                "A unique color_bounds must be specified for each imager."
                )
            color_bounds = kwargs['color_bounds']
        else:
            raise ValueError(
                f'The shape of color_bounds must be (1, 2), or '
                f'({len(self.imagers)}, 2), not {np.array(kwargs["color_bounds"]).shape}'
                )
        # Must remove the color_bounds variable, if the kwargs dict contains it, 
        # so the call to imager.plot_map() doesn't raise an error because of
        # duplicate keyword arguments. 
        kwargs.pop('color_bounds', None)

        self._skymaps = {
            _imager.meta['location']:{
                'lon':_imager.skymap['lon'].copy(), 
                'lat':_imager.skymap['lat'].copy(),
                'el':_imager.skymap['el'].copy()
                } for _imager in self.imagers
            }
        # If overlap=False, each skymap first has nans in the imager overlapping region,
        # and then the skymap cleaner nans everything below min_elevation, which is then
        # immediately followed by the reassignment of all nans to the nearest valid value.
        if not overlap:
            self._skymaps = self._calc_overlap_mask(self._skymaps, shrink_distance_to_self=0.96)

        for imager, _color_bounds in zip(self.imagers, color_bounds):
            _skymap_cleaner = Skymap_Cleaner(
                self._skymaps[imager.meta['location']]['lon'], 
                self._skymaps[imager.meta['location']]['lat'], 
                self._skymaps[imager.meta['location']]['el'], 
            )
            _skymap_cleaner.mask_elevation(kwargs.get('min_elevation', 5))
            _cleaned_lon_grid, _cleaned_lat_grid = _skymap_cleaner.remove_nans()
            imager.plot_map(
                **kwargs, 
                color_bounds=_color_bounds, 
                lon_grid=_cleaned_lon_grid, 
                lat_grid=_cleaned_lat_grid
                )
        return
    
    # def animate_fisheye(self):
    #     raise NotImplementedError
    
    # def animate_fisheye_gen(self):
    #     raise NotImplementedError
    
    def animate_map(self, **kwargs):
        """
        Animate an ASI mosaic. It is a wrapper for the 
        :py:meth:`~asilib.Imagers.animate_map_gen` method.

        See :py:meth:`~asilib.Imagers.animate_map_gen` documentation for the complete 
        list of kwargs.

        Example
        -------
        .. code-block:: python

            >>> import asilib
            >>> import asilib.asi
            >>> 
            >>> time_range = ('2021-11-04T06:55', '2021-11-04T07:05')
            >>> asis = asilib.Imagers(
            >>>     [asilib.asi.trex_rgb(location_code, time_range=time_range) 
            >>>     for location_code in ['LUCK', 'PINA', 'GILL', 'RABB']]
            >>>     )
            >>> asis.animate_map(lon_bounds=(-115, -85), lat_bounds=(43, 63), overwrite=True)
        """
        for _ in self.animate_map_gen(**kwargs):
            pass
        return
    
    def animate_map_gen(
        self,
        overlap=False,
        lon_bounds: tuple = (-160, -50),
        lat_bounds: tuple = (40, 82),
        ax: Union[plt.Axes, tuple] = None,
        coast_color: str = 'k',
        land_color: str = 'g',
        ocean_color: str = 'w',
        color_map: str = None,
        color_bounds: List[float] = None,
        color_norm: str = None,
        min_elevation: float = 10,
        pcolormesh_kwargs: dict = {},
        asi_label: bool = True,
        movie_container: str = 'mp4',
        animation_save_dir: Union[pathlib.Path, str]=None,
        ffmpeg_params={},
        overwrite: bool = False,
        ) -> Generator[
            Tuple[datetime, list[datetime], list[np.ndarray], plt.Axes], None, None
        ]:
        """
        Animate an ASI mosaic.

        Parameters
        ----------
        overlap: bool
            If True, pixels that overlap between imager FOV's are overplotted such that only the 
            final imager's pixels are shown.
        lon_bounds: tuple
            The map's longitude bounds.
        lat_bounds: tuple
            The map's latitude bounds.
        ax: plt.Axes, tuple
            The subplot to put the map on. If cartopy is installed, ```ax``` must be
            a two element tuple specifying the ``plt.Figure`` object and subplot position
            passed directly as ``args`` into
            `fig.add_subplot() <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.add_subplot>`_.
        coast_color: str
            The coast color. If None will not draw it.
        land_color: str
            The land color. If None will not draw it.
        ocean_color: str
            The ocean color. If None will not draw it.
        color_map: str
            The matplotlib colormap to use. See `matplotlib colormaps <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
            for supported colormaps.
        color_bounds: List[float]
            The lower and upper values of the color scale. The default is: low=1st_quartile and
            high=min(3rd_quartile, 10*1st_quartile). This range works well for most cases.
        color_norm: str
            Set the 'lin' (linear) or 'log' (logarithmic) color normalization. If color_norm=None,
            the color normalization will be taken from the ASI array (if specified), and if not
            specified it will default to logarithmic.
        min_elevation: float
            Masks the pixels below min_elevation degrees.
        pcolormesh_kwargs: dict
            A dictionary of keyword arguments (kwargs) to pass directly into
            plt.pcolormesh.
        asi_label: bool
            Annotates the map with the ASI code in the center of the mapped image.
        movie_container: str
            The movie container: mp4 has better compression but avi was determined
            to be the official container for preserving digital video by the
            National Archives and Records Administration.
        ffmpeg_params: dict
            The additional/overwitten ffmpeg output parameters. The default parameters are:
            framerate=10, crf=25, vcodec=libx264, pix_fmt=yuv420p, preset=slower.
        overwrite: bool
            Overwrite the animation. If False, ffmpeg will prompt you to answer y/n if the
            animation already exists.

        Yields
        ------
        datetime.datetime
            The guide time used to keep the images synchronized.
        List[datetime.datetime]
            Nearest imager time stamps to the guide time. If the difference between
            the imager time and the guide time is greater than time_tol*imager_cadence, 
            or the imager is off, the imager is considered unsynchronized and the 
            returned time is datetime.min.
        List[np.ndarray]
            The images corresponding to the times returned above. If the that imager is
            unsynchronized, the corresponding image value is None.
        plt.Axes
            The subplot object.

        Example
        -------
        .. code-block:: python

            >>> # Animate a TREX-RGB mosaic and print the individual time stamps
            >>> # to confirm that the imagers are synchronized.
            >>> import asilib
            >>> import asilib.asi
            >>> 
            >>> time_range = ('2021-11-04T06:55', '2021-11-04T07:05')
            >>> asis = asilib.Imagers(
            >>>     [asilib.asi.trex_rgb(location_code, time_range=time_range) 
            >>>     for location_code in ['LUCK', 'PINA', 'GILL', 'RABB']]
            >>>     )
            >>> gen = asis.animate_map_gen(
            >>>     lon_bounds=(-115, -85), lat_bounds=(43, 63), overwrite=True
            >>>     )
            >>> for guide_time, asi_times, asi_images, ax in gen:
            >>>     if '_text_obj' in locals():
            >>>         _text_obj.remove()
            >>>     info_str = f'Guide: {guide_time: %Y:%m:%d %H:%M:%S}\\n'
            >>>     # The below for loop is possible because the imagers and 
            >>>     # asi_times can be indexed together.
            >>>     for _imager, _imager_time in zip(asis.imagers, asi_times):
            >>>         info_str += f'{_imager.meta["location"]}: {_imager_time: %Y:%m:%d %H:%M:%S}\\n'
            >>>     info_str = info_str[:-1]  # Remove the training newline
            >>> 
            >>>     _text_obj = ax.text(
            >>>         0.01, 0.99, info_str, va='top', transform=ax.transAxes, 
            >>>         bbox=dict(facecolor='grey', edgecolor='black'))
            
        """
        if ax is None:
            ax = asilib.map.create_map(
                lon_bounds=lon_bounds,
                lat_bounds=lat_bounds,
                coast_color=coast_color,
                land_color=land_color,
                ocean_color=ocean_color,
            )

        # Handle manual color_bounds for none, all, or each imager.
        if color_bounds is None:
            color_bounds = np.full(len(self.imagers), None)
        elif len(np.array(color_bounds).shape) == 1:  # Same color_bounds for all imagers.
            color_bounds = np.repeat(
                np.array(color_bounds)[:, np.newaxis], 
                len(self.imagers), 
                axis=-1
                ).T
        elif len(np.array(color_bounds).shape) == 2:
            assert np.array(color_bounds).shape == (len(self.imagers), 2), (
                "A unique color_bounds must be specified for each imager."
                )
        else:
            raise ValueError(
                f'The shape of color_bounds must be (1, 2), or '
                f'({len(self.imagers)}, 2), not {np.array(color_bounds).shape}'
                )

        # Create the animation directory inside asilib.config['ASI_DATA_DIR'] if it does
        # not exist.
        if animation_save_dir is None:
            _path = asilib.config['ASI_DATA_DIR']
        else:
            _path = animation_save_dir
        image_save_dir = pathlib.Path(
            _path,
            'animations',
            'images',
            f'{self.imagers[0].file_info["time_range"][0].strftime("%Y%m%d_%H%M%S")}_mosaic',
        )

        self.animation_name = (
            f'{self.imagers[0].file_info["time_range"][0].strftime("%Y%m%d_%H%M%S")}_'
            f'{self.imagers[0].file_info["time_range"][-1].strftime("%H%M%S")}_mosaic.'
            f'{movie_container}'
        )
        movie_save_path = image_save_dir.parents[1] / self.animation_name
        # If the image directory exists we need to first remove all of the images to avoid
        # animating images produced by different method calls.
        if image_save_dir.is_dir():
            shutil.rmtree(image_save_dir)
        image_save_dir.mkdir(parents=True)

        image_paths = []
        _progressbar = asilib.utils.progressbar(
            enumerate(self.__iter__()),
            iter_length=self.imagers[0]._estimate_n_times(),
            text=self.animation_name,
        )
        operating_asis = np.array([])

        for i, (_guide_time, _asi_times, _asi_images) in _progressbar:
            asi_labels = len(self.imagers)*[None]
            pcolormesh_objs = len(self.imagers)*[None]

            # Determine if an imager turned off/on and we need to recalculate overlapping skymaps.
            currently_on_asis = np.where(np.array(_asi_times) != datetime.min)[0].astype(int)
            if (currently_on_asis.shape != operating_asis.shape) or np.any(currently_on_asis != operating_asis):
                _skymaps = {}
                for j in currently_on_asis:
                    imager = self.imagers[j]
                    _skymaps[imager.meta['location']] = {
                        'lon':imager.skymap['lon'].copy(), 
                        'lat':imager.skymap['lat'].copy()
                    }

                if not overlap:
                    _skymaps = self._calc_overlap_mask(_skymaps, idx=currently_on_asis)

                for j in currently_on_asis:
                    imager = self.imagers[j]
                    _skymap_cleaner = Skymap_Cleaner(
                        _skymaps[imager.meta['location']]['lon'], 
                        _skymaps[imager.meta['location']]['lat'], 
                        imager.skymap['el'],
                    )
                    _skymap_cleaner.mask_elevation(min_elevation=min_elevation)
                    _cleaned_lon_grid, _cleaned_lat_grid = _skymap_cleaner.remove_nans()
                    _skymaps[imager.meta['location']]['lon'] = _cleaned_lon_grid
                    _skymaps[imager.meta['location']]['lat'] = _cleaned_lat_grid
                operating_asis = currently_on_asis

            for j, (_asi_time, _asi_image) in enumerate(zip(_asi_times, _asi_images)):
                if _asi_time == datetime.min:
                    continue
                _color_map, _color_norm = self.imagers[j]._plot_params(
                    _asi_image, color_bounds[j], color_map, color_norm
                    )

                ax, pcolormesh_objs[j], asi_labels[j] = self.imagers[j]._plot_mapped_image(
                    ax, 
                    _asi_image, 
                    min_elevation, 
                    _color_map, 
                    _color_norm, 
                    asi_label, 
                    pcolormesh_kwargs, 
                    lon_grid=_skymaps[self.imagers[j].meta['location']]['lon'], 
                    lat_grid=_skymaps[self.imagers[j].meta['location']]['lat']
                )

            # Give the user the control of the subplot, image object, and return the image time
            # so that they can manipulate the image to add, for example, the satellite track.
            yield _guide_time, _asi_times, _asi_images, ax

            # Save the plot before the next iteration.
            save_name = f'{str(i).zfill(6)}.png'
            plt.savefig(image_save_dir / save_name)
            image_paths.append(image_save_dir / save_name)

            # Clean up the objects that this method generated.
            for _asi_label in asi_labels:
                if _asi_label is not None:
                    _asi_label.remove()
            for pcolormesh_obj in pcolormesh_objs:
                if pcolormesh_obj is not None:
                    pcolormesh_obj.remove()
        
        self.imagers[0]._create_animation(image_paths, movie_save_path, ffmpeg_params, overwrite)
        return
    
    def map_eq(self, b_model: Callable='IGRF') -> Tuple[np.ndarray, np.ndarray]:
        """
        Map an auroral image to the magnetic equator using IGRF or a user-defined magnetic field model.

        Parameters
        ----------
        b_model: Callable
            A function with the (time, lla) arguments where time is a single time stamp, 
            and lla is one (lat, lon, alt) point. If ``b_model='IGRF'``, IRBEM's IGRF model is used
            in the default GSM coordinates.

        Returns
        -------
        numpy.ndarray:
            Mapped locations in the Geocentric Solar Magnetospheric (GSM) coordinates.
        numpy.ndarray:
            Auroral pixel intensities.
        """
        warnings.warn(
            'asilib.Imagers.map_eq() is an experimental method. Its functionality or interface '
            'stability is not guaranteed.'
            )
        if b_model == 'IGRF':
            b_model = self._igrf_model_wrapper
            if not _irbem_imported:
                raise ImportError(
                    'IRBEM is not installed. Please install it to use the IGRF model, '
                    'or supply a custom b_model.'
                    )
            
        lats_lons, intensities = self.get_points()
        equator_sm = np.zeros((lats_lons[:, 0].shape[0], 3))

        _progressbar = asilib.utils.progressbar(
            enumerate(zip(lats_lons[:, 0], lats_lons[:, 1])),
            iter_length=lats_lons[:, 0].shape[0],
            text='Mapping to magnetic equator'
            )

        for i, (lat, lon) in _progressbar:
            equator_sm[i, :] = b_model(self.imagers[0].file_info['time'], (lat, lon, self.imagers[0].skymap['alt']))
        return equator_sm, intensities
    
    def plot_map_eq(
            self, 
            ax:plt.Axes=None, 
            b_model:Callable="IGRF",
            max_valid_grid_distance:float=0.03,
            x_grid:np.ndarray=None, 
            y_grid:np.ndarray=None, 
            color_bounds: List[float] = None,
            color_norm: str = None,
            color_map: str = None,
            pcolormesh_kwargs:dict={},
            ) -> Tuple[plt.Axes, matplotlib.collections.QuadMesh]:
        """
        Plot the equatorial projection of an auroral image, mapped using IGRF or a user-defined 
        magnetic field model.

        Parameters
        ----------
        ax: plt.Axes
            The subplot to put the map on. If None, one will be created.
        b_model: Callable
            A function with the (time, lla) arguments where time is a single time stamp, 
            and lla is one (lat, lon, alt) point. If ``b_model='IGRF'``, IRBEM's IGRF model is used
            in the default GSM coordinates.
        max_valid_grid_distance: float
            The maximum distance from the mapped point to the grid point. If the distance is greater
            than this value, the mapped grid point will be masked as NaN. The distance is in units of Re.
        x_grid, x_grid: np.ndarray
            The x and y grid points to plot the mapped image. If None, a grid of 1000x1001 points spanning 
            -12 < x < 1.1 and -5 < y < 5 is generated.
        color_bounds: List[float]
            The ASI image color bounds.
        color_map: str
            The matplotlib colormap to use. See `matplotlib colormaps <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
            for supported colormaps.
        color_norm: str
            Set the 'lin' (linear) or 'log' (logarithmic) color normalization. If color_norm=None,
            the color normalization will be taken from the ASI array (log for black & white images 
            and linear for RGB images).
        pcolormesh_kwargs: dict
            Keyword arguments directly passed into plt.pcolormesh.
        """
        if ax is None:
            _, ax = plt.subplots()
        if (x_grid is None) and (y_grid is None):
            x_grid, y_grid = np.meshgrid(np.linspace(-12, 1.1, num=1000), np.linspace(-5, 5, num=1001))

        equator_sm, intensities = self.map_eq(b_model=b_model)

        gridded_eq_data = scipy.interpolate.griddata(
            equator_sm[:, :2], 
            intensities, 
            (x_grid, y_grid), 
            method='nearest'
            )
        # https://stackoverflow.com/a/31189177
        tree = cKDTree(equator_sm[:, :2])
        dists, _ = tree.query(np.stack((x_grid, y_grid), axis=-1))
        gridded_eq_data[dists > max_valid_grid_distance, :] = np.nan  # Mask any gridded point > 0.1 Re from the mapped point as NaN

        color_map, color_norm = self.imagers[0]._plot_params(gridded_eq_data, color_bounds, color_map, color_norm)

        vmin, vmax = self.imagers[0].plot_settings['color_bounds']
        if len(self.imagers[0].meta['resolution']) == 3:
            vmin, vmax = self.imagers[0].get_color_bounds()
            gridded_eq_data = self.imagers[0]._rgb_replacer(gridded_eq_data)
            gridded_eq_data = asilib.utils.stretch_contrast(gridded_eq_data, vmin, vmax)

            # Add a transprancy channel so that the NaN RGB values show up as transparent instead of black.
            gridded_eq_data = np.concatenate(
                (gridded_eq_data, np.ones(gridded_eq_data.shape[:2]).reshape(*gridded_eq_data.shape[:2], 1))
                , axis=-1
                )
            gridded_eq_data[np.isnan(gridded_eq_data.sum(axis=-1)), -1] = 0

        pcolormesh_kwargs['norm'] = color_norm
        pcolormesh_kwargs['cmap'] = color_map
        gridded_eq_data = np.ma.array(gridded_eq_data, mask=np.isnan(gridded_eq_data))
        p = ax.pcolormesh(x_grid, y_grid, gridded_eq_data, **pcolormesh_kwargs)

        # Draw Earth circle
        ax.add_artist(matplotlib.patches.Wedge((0, 0), 1, -90, 90, fc='w', ec='k'))
        ax.add_artist(matplotlib.patches.Wedge((0, 0), 1, 90, 270, fc='k', ec='k'))
        return ax, p

    def _igrf_model_wrapper(self, time, lla):
        if not hasattr(self, '_irbem_obj'):
            self._irbem_obj = IRBEM.MagFields(kext=0)
            self._coords_obj = IRBEM.Coords()
        output_dictionary = self._irbem_obj.find_magequator(
            {'datetime':time, 'x1':lla[2], 'x2':lla[0], 'x3':lla[1]},
            {}
            )
        equator_gsm = self._coords_obj.transform(time, output_dictionary['XGEO'], 1, 2)  # Convert to GSM coordinates
        return equator_gsm
    
    def __iter__(self) -> Generator[datetime, List, List]:
        """
        Generate a list of time stamps and images for all synchronized imagers, one time stamp 
        at a time.

        Yields
        ------
        datetime.datetime
            The guide time used to synchronize the imagers. If the difference between the imager
            time and the guide time is greater than time_tol*imager_cadence, or the imager is off,
            the imager is considered unsynchronized.
        list
            The time stamps from each imager. If the imager is unsynchronized the yielded 
            time stamp is ``datetime.min``.
        list 
            The images from each imager. If the imager is unsynchronized the yielded 
            image is ``None``.
        """
        t0 = self.imagers[0].file_info['time_range'][0]
        # TODO: Check all imagers for the quickest cadence to allow for multi-ASI array mosaics.
        times = np.array(
            [t0+timedelta(seconds=i*self.imagers[0].meta['cadence'])
            for i in range(self.imagers[0]._estimate_n_times())]
            )
        # asi_iterators keeps track of all ASIs in Imagers, with same order as passed into
        # __init__(). 
        # 
        # We must also keep track of ASIs whose next time stamp beyond the allowable tolerance 
        # of guide_time, or is off. future_iterator holds the imager (time, image) data until 
        # it becomes synchronized with the guide_time again.
        asi_iterators = {
            f'{_imager.meta["array"]}-{_imager.meta["location"]}':iter(_imager) 
            for _imager in self.imagers
            }
        future_iterators = {}
        stopped_iterators = []

        for guide_time in times:
            _asi_times = []
            _asi_images = []
            for _name, _iterator in asi_iterators.items():
                # We have three cases to address. If the iterator is synchronized, the
                # future_iterators will not have the _name key. This will trigger next() 
                # on that iterator. In either case, this will return a time stamp and 
                # an image. The one exception is when the _iterator is exhausted we fill 
                # in dummy values for the time and image.
                try:
                    _asi_time, _asi_image = future_iterators.get(_name, next(_iterator))
                except StopIteration:
                    _asi_times.append(datetime.min)
                    _asi_images.append(None)

                    if _name not in stopped_iterators:
                        stopped_iterators.append(_name)
                    if len(asi_iterators) == len(stopped_iterators):
                        # Stop once all iterators are exhausted.
                        return
                    continue
                abs_dt = np.abs((guide_time-_asi_time).total_seconds())
                synchronized = abs_dt < self.imagers[0].meta['cadence']*self.iter_tol

                # We must always append a time stamp and image, even if a dummy variable
                # to preserve the Imager order.
                if synchronized:
                    _asi_times.append(_asi_time)
                    _asi_images.append(_asi_image)
                else:
                    future_iterators[_name] = (_asi_time, _asi_image)
                    _asi_times.append(datetime.min)
                    _asi_images.append(None)

                if synchronized and (_name in future_iterators):
                    future_iterators.pop(_name)
            yield guide_time, _asi_times, _asi_images
        return

    def get_points(self, min_elevation:float=10)->Tuple[np.ndarray, np.ndarray]:
        """
        Get pixel intensities in each (lat, lon) grid point.

        Parameters
        ----------
        min_elevation: float
            Only return pixel intensities above min_elevation.

        Returns
        -------
        np.ndarray
            An (n, 2) array with each row corresponding to a (lat, lon) point.
        np.ndarray
            Pixel intensities with shape (n) for white-light images, and (n, 3) for RGB images.

        Examples
        --------
        >>> from datetime import datetime
        >>> 
        >>> import asilib
        >>> import asilib.asi
        >>> 
        >>> time = datetime(2007, 3, 13, 5, 8, 45)
        >>> location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
        >>> map_alt = 110
        >>> min_elevation = 2
        >>> 
        >>> _imagers = [asilib.asi.themis(location_code, time=time, alt=map_alt) 
        >>>             for location_code in location_codes]
        >>> asis = asilib.Imagers(_imagers)
        >>> lat_lon_points, intensities = asis.get_points(min_elevation=min_elevation)

        >>> # A comprehensive example showing how Imagers.get_points() can closely reproduce 
        >>> # Imagers.plot_map()
        >>> from datetime import datetime
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib.colors
        >>> 
        >>> import asilib
        >>> import asilib.map
        >>> import asilib.asi
        >>> 
        >>> time = datetime(2007, 3, 13, 5, 8, 45)
        >>> location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
        >>> map_alt = 110
        >>> min_elevation = 2
        >>> 
        >>> _imagers = [asilib.asi.themis(location_code, time=time, alt=map_alt) 
        >>>             for location_code in location_codes]
        >>> asis = asilib.Imagers(_imagers)
        >>> lat_lon_points, intensities = asis.get_points(min_elevation=min_elevation)
        >>> 
        >>> fig = plt.figure(figsize=(12,5))
        >>> ax = asilib.map.create_simple_map(
        >>>     lon_bounds=(-140, -60), lat_bounds=(40, 82), fig_ax=(fig, 121)
        >>>     )
        >>> bx = asilib.map.create_simple_map(
        >>>     lon_bounds=(-140, -60), lat_bounds=(40, 82), fig_ax=(fig, 122)
        >>>     )
        >>> asis.plot_map(ax=ax, overlap=False, min_elevation=min_elevation)
        >>> bx.scatter(lat_lon_points[:, 1], lat_lon_points[:, 0], c=intensities, 
        >>>         norm=matplotlib.colors.LogNorm())
        >>> ax.text(0.01, 0.99, f'(A) Mosaic using Imagers.plot_map()', transform=ax.transAxes, 
        >>>         va='top', fontweight='bold', color='red')
        >>> bx.text(0.01, 0.99, f'(B) Mosaic from Imagers.get_points() scatter', transform=bx.transAxes,
        >>>         va='top', fontweight='bold', color='red')
        >>> fig.suptitle('Donovan et al. 2008 | First breakup of an auroral arc')
        >>> plt.tight_layout()
        >>> plt.show()
        """
        lat_lon_points = np.zeros((0, 2), dtype=float)
        if len(self.imagers[0].meta['resolution']) == 3: # RGB
            intensities  = np.zeros((0, self.imagers[0].meta['resolution'][-1]), dtype=float)
        else:  # single-color (or white light)
            intensities  = np.zeros(0, dtype=float)

        _skymaps = {}
        for imager in self.imagers:
            _skymaps[imager.meta['location']] = {
                'lon':imager.skymap['lon'].copy(), 
                'lat':imager.skymap['lat'].copy()
            }
        _skymaps = self._calc_overlap_mask(_skymaps)

        for _imager in self.imagers:
            assert 'time' in _imager.file_info.keys(), (
                f'Imagers.get_points() only works with single images.'
                )
            _skymap_cleaner = Skymap_Cleaner(
                _skymaps[_imager.meta['location']]['lon'], 
                _skymaps[_imager.meta['location']]['lat'], 
                _imager.skymap['el'],
            )
            _masked_lon_map, _masked_lat_map, _ = _skymap_cleaner.mask_elevation(
                min_elevation=min_elevation
                )

            if _imager.skymap['lon'].shape[0] == _imager.data.image.shape[0] + 1:
                # Skymap defines vertices. We look for NaNs at either of the pixel edges.
                _valid_idx = np.where(
                    ~np.isnan(_masked_lon_map[1:, 1:]-_masked_lon_map[:-1, :-1])
                    )
            elif _imager.skymap['lon'].shape[0] == _imager.data.image.shape[0]:
                # Skymap defines pixel centers
                _valid_idx = np.where(~np.isnan(_masked_lon_map))
            else:
                raise ValueError(f'The skymap shape: {_imager.skymap["lon"].shape} and image '
                                 f'shape: {_imager.data.image.shape} are incompatible.')

            lat_grid = _masked_lat_map[_valid_idx[0], _valid_idx[1]]
            lon_grid = _masked_lon_map[_valid_idx[0], _valid_idx[1]]
            intensity = _imager.data.image[_valid_idx[0], _valid_idx[1], ...]

            # Concatenate joins arrays along an existing axis, while stack joins arrays
            # along a new axis. 
            intensities = np.concatenate((intensities, intensity))
            lat_lon_points = np.concatenate((
                lat_lon_points, 
                np.stack((lat_grid, lon_grid)).T
                ))
        return lat_lon_points, intensities
    
    def _calc_overlap_mask(self, _skymaps, idx=None, shrink_distance_to_self=0.98):
        """
        Calculate which pixels to plot for overlapping imagers by the criteria that the ith 
        imager's pixel must be closest to that imager (and not a neighboring one).

        Algorithm:
        1. Loop over ith imager
        2. Loop over jth imager within 500 km distance of the ith imager
        3. Mask low-elevations with np.nan
        4. Create a distance array with shape (resolution[0], resolution[1], j_total)
        5. For all pixels in ith imager, calculate the haversine distance to jth imager and 
        assign it to distance[..., j].
        6. For all pixels calculate the nearest imager out of all j's.
        7. If the minimum j is not the ith imager, mask the imager.skymap['lat'] and 
        imager.skymap['lon'] as np.nan.
        """
        # This variable keeps track of all imagers in case their field of view is clipped
        # by the CCD (TREx-RGB, for example has this). If at least one imager's FOV is 
        # clipped, it will run an additional step to check if some of the pixels to be removed
        # from one imager are not in the clipped region of the other imager.
        clipped_fov=False

        if idx is None:
            _imagers = self.imagers
        else:
            _imagers = [self.imagers[i] for i in idx]

        for imager in _imagers:
            merged_edges = np.concatenate((
                _skymaps[imager.meta['location']]['lat'][0, :], 
                _skymaps[imager.meta['location']]['lat'][-1, :], 
                _skymaps[imager.meta['location']]['lat'][:, 0], 
                _skymaps[imager.meta['location']]['lat'][:, -1]
                ))
            if not np.all(np.isnan(merged_edges)):
                clipped_fov=True
                # Save the clipped (lats, lon) to compare with here.

        for i, imager in enumerate(_imagers):
            _distances = np.nan*np.ones(
                (*_skymaps[imager.meta['location']]['lat'].shape, len(_imagers))
                )
            for j, other_imager in enumerate(_imagers):
                # Calculate the distance between all imager pixels and every other imager 
                # location (including itself).
                _other_lon = np.broadcast_to(
                    other_imager.meta['lon'], 
                    _skymaps[imager.meta['location']]['lat'].shape
                    )
                _other_lat = np.broadcast_to(
                    other_imager.meta['lat'], 
                    _skymaps[imager.meta['location']]['lat'].shape
                    )

                _distances[:, :, j] = _haversine(
                    _skymaps[imager.meta['location']]['lat'], 
                    _skymaps[imager.meta['location']]['lon'],
                    _other_lat, 
                    _other_lon
                    )
                
                # _distances[:, :, i] = 0
            # Without this small reduction in the distance of pixels to its own imager,
            # there are gaps between the imager boundaries. In other words, this scaling
            # slightly biases to plotting pixels nearest to the imager. 
            _distances[:, :, i] *= shrink_distance_to_self
            # TODO: Set the _distances[:, :, i] values to 0 (not masked out) if they are 
            # #closer to the other imager but the other imager's pixels are nans.
            # Need a masked array so that np.nanargmin correctly handles all NaN slices.
            _distances = np.ma.masked_array(_distances, np.isnan(_distances))
            # For each pixel, calculate the nearest imager. If the pixel is not closest to 
            # the imager that it's from, mask it as np.nan.
            min_distances = np.argmin(_distances, axis=2)
            far_pixels = np.where(min_distances != i)

            # if clipped_fov and (far_pixels[0].shape[0] > 0):
            #     far_pixels = self._valid_far_pixels(i, _skymaps, far_pixels)

            _skymaps[imager.meta['location']]['lat'][far_pixels] = np.nan
            _skymaps[imager.meta['location']]['lon'][far_pixels] = np.nan

            # TODO: Add an option to return the skymaps with just the overlapping part.
        return _skymaps
    
    def _valid_far_pixels(self, imager_id, _skymaps, far_pixels):
        """
        Sometimes for imagers with clipped FOV the far pixel from the main imager falls inside 
        the clipped region. This loop checks that these far pixels are not in the clipped FOV.
        """
        imager = self.imagers[imager_id]
        valid_far_pixels = []

        _progressbar = asilib.utils.progressbar(
            enumerate(zip(far_pixels[0], far_pixels[1])),
            iter_length=far_pixels[0].shape[0],
            text='Processing clipped FOV pixels',
        )
        for i, far_pixel in _progressbar:
            _lon = _skymaps[imager.meta['location']]['lon'][far_pixel]
            _lat = _skymaps[imager.meta['location']]['lat'][far_pixel]
            
            for k, other_imager in enumerate(self.imagers):
                if k==i:
                    continue
                _lon_arr = np.broadcast_to(
                    _lon, 
                    _skymaps[other_imager.meta['location']]['lon'].shape
                )
                _lat_arr = np.broadcast_to(
                    _lat, 
                    _skymaps[other_imager.meta['location']]['lat'].shape
                )
                _other_lon_arr = _skymaps[other_imager.meta['location']]['lon']
                _other_lat_arr = _skymaps[other_imager.meta['location']]['lat']
                _distances_other_imager = _haversine(
                    _other_lon_arr,
                    _other_lat_arr,
                    _lat_arr,
                    _lon_arr
                    )
                idx_other = np.unravel_index(
                    np.argmin(_distances_other_imager), 
                    _skymaps[other_imager.meta['location']]['lon'].shape
                    )
                if _distances_other_imager[idx_other] < 100:
                    valid_far_pixels.append(far_pixel)
        return np.array(valid_far_pixels).astype(int)
    
    def __str__(self):
        names = [f'{_img.meta["array"]}-{_img.meta["location"]}' for _img in self.imagers]
        names = 'asilib.Imagers initiated with:\n' + ', '.join(names)

        if (
                ('time' in self.imagers[0].file_info.keys()) and 
                (self.imagers[0].file_info['time'] is not None)
            ):
            return (names + f'\ntime={self.imagers[0].file_info["time"]}')
        elif (
                ('time_range' in self.imagers[0].file_info.keys()) and 
                (self.imagers[0].file_info['time_range'] is not None)
            ):
            return (names + f'\ntime_range={self.imagers[0].file_info["time_range"]}')
        else:
            raise ValueError(
                'The 0th imager object does not have a "time" or a "time_range" variable.'
                )
