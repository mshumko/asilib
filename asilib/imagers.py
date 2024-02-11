"""
The Imagers class handles multiple Imager objects and coordinates plotting and animating multiple
fisheye lens and mapped images. The mapped images are also called mosaics.
"""

from typing import Tuple, List, Union
from collections import namedtuple
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

from asilib.imager import Imager, _haversine
import asilib.map


class Imagers:
    """
    Plot and animate multiple :py:meth:`~asilib.imager.Imager` objects.

    .. warning::

        This class is in development and not all methods are implemented/matured.

    Parameters
    ----------
    imagers: Tuple
        The Imager objects to plot and animate. 
    """
    def __init__(self, imagers:Tuple[Imager]) -> None:
        self.imagers = imagers
        # Wrap self.imagers in a tuple if the user passes in a single Imager object.
        if isinstance(self.imagers, Imager):
            self.imagers = (self.imagers, )
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
            Keyword arguments directly passed into each :py:meth:`~asilib.imager.Imager.plot_fisheye()`
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
            Keyword arguments directly passed into each :py:meth:`~asilib.imager.Imager.plot_map()`
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
        if not overlap:
            self._calc_overlap_mask()

        for imager in self.imagers:
            imager.plot_map(**kwargs)
        return
    
    # def animate_fisheye(self):
    #     raise NotImplementedError
    
    # def animate_fisheye_gen(self):
    #     raise NotImplementedError
    
    def animate_map(
            self, 
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
            ffmpeg_params={},
            overwrite: bool = False,
            overlap=False
            ):
        """
        Animate an ASI mosaic.

        Parameters
        ----------
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
        """
        if ax is None:
            ax = asilib.map.create_map(
                lon_bounds=lon_bounds,
                lat_bounds=lat_bounds,
                coast_color=coast_color,
                land_color=land_color,
                ocean_color=ocean_color,
            )
        for _ in self.animate_map_gen(ax=ax, overlap=overlap):
            pass
        return
    
    def animate_map_gen(self, ax=None, overlap=False):
        """
        A generator to animate the ASI mosaic.

        Parameters
        ----------
        TODO: Finish docs
        """
        if overlap:
            self._calc_overlap_mask()

        for _asi_times, _asi_images in self.__iter__():
            yield _asi_times, _asi_images, # ax, pcolormesh_obj

        return
    
    def __iter__(self, time_tol=2):
        """
        Generate a set of time stamps and images for all imagers, one time stamp at a time.

        The algorithm is to loop over all time stamps within the first imager's time_range,
        with an inner loop that loops over all of the imagers to get the next time stamp.
        As long as the all ASIs next time stamps are within the time_tol of the current
        time stamp, we then yield all of the time stamps and images.

        Parameters
        ----------
        time_tol: float
            The allowable time tolerance, in units of time_tol*imager_cadence, to check the 
            imagers' time stamps for alignment (so the correct images from all imagers are 
            animated).

        Raises
        ------
        ValueError
            If the time stamps are misaligned.
        """
        t0 = self.imagers[0].file_info['time_range'][0]

        times = np.array(
            [t0+timedelta(seconds=i*self.imagers[0].meta['cadence']) 
            for i in range(self.imagers[0]._estimate_n_times())]
            )
        # asi_iterators keeps track of all ASIs in Imagers, with same order as passed into
        # __init__(). 
        # 
        # We must also keep track of ASIs whose next time stamp is in the future, or is off.
        # future_iterator is for an ASI that has the next time stamp ahead of _time and
        asi_iterators = {
            f'{_imager.meta["array"]}-{_imager.meta["location"]}':iter(_imager) 
            for _imager in self.imagers
            }
        future_iterators = {}

        for _time in times:
            _asi_times = []
            _asi_images = []
            for i, (_name, _iterator) in enumerate(asi_iterators.items()):

                if _name not in future_iterators:
                    # An imager synchronized in time or is off.
                    try:
                        _asi_time, _asi_image = next(_iterator)
                    except StopIteration:
                        _asi_times.append(datetime.min)
                        _asi_images.append(None)
                        continue
                    abs_dt = np.abs((_time-_asi_times[-1]).total_seconds())

                else:
                    # An imager unsynchronized in time
                    if np.abs((_time-future_iterators[_name][0]).total_seconds()) < self.imagers[0].meta['cadence']*time_tol:
                        _asi_times.append(_asi_time)
                        _asi_images.append(_asi_image)

                    future_iterators.pop(_name)

                if np.abs((_time-_asi_time).total_seconds()) < self.imagers[0].meta['cadence']*time_tol:
                    _asi_times.append(_asi_time)
                    _asi_images.append(_asi_image)
                else:  # Save the imager state to future_iterators if it is far in the future
                    _asi_times.append(None)
                    _asi_images.append(None)
                    future_iterators[_name] = (_asi_time, _asi_image)

            # Replace an image with None when the time stamp for it becomes misaligned.
            dt = np.array([(_time-ti).total_seconds() for ti in _asi_times])
            outside_tol = np.where(np.abs(dt) > self.imagers[0].meta['cadence']*time_tol)[0]
            for i in outside_tol:  # Usually skip for loop if all imagers are in sync.
                _asi_images[i] = None            
            yield _asi_times, _asi_images
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

        self._calc_overlap_mask()

        for _imager in self.imagers:
            assert 'time' in _imager.file_info.keys(), (
                f'Imagers.get_points() only works with single images.'
                )
            image = _imager.data.image
            _masked_lon_map, _masked_lat_map, _masked_image = _imager._mask_low_horizon(
                _imager.skymap['lon'], _imager.skymap['lat'], _imager.skymap['el'], min_elevation, 
                image=image
            )
            if _imager.skymap['lon'].shape[0] == image.shape[0] + 1:
                # Skymap defines vertices. We look for NaNs at either of the pixel edges.
                _valid_idx = np.where(
                    ~np.isnan(_masked_lon_map[1:, 1:]-_masked_lon_map[:-1, :-1])
                    )
            elif _imager.skymap['lon'].shape[0] == image.shape[0]:
                # Skymap defines pixel centers
                _valid_idx = np.where(~np.isnan(_masked_lon_map))
            else:
                raise ValueError(f'The skymap shape: {_imager.skymap["lon"].shape} and image '
                                 f'shape: {image.shape} are incompatible.')

            lat_grid = _masked_lat_map[_valid_idx[0], _valid_idx[1]]
            lon_grid = _masked_lon_map[_valid_idx[0], _valid_idx[1]]
            intensity = _masked_image[_valid_idx[0], _valid_idx[1], ...]

            # Concatenate joins arrays along an existing axis, while stack joins arrays
            # along a new axis. 
            intensities = np.concatenate((intensities, intensity))
            lat_lon_points = np.concatenate((
                lat_lon_points, 
                np.stack((lat_grid, lon_grid)).T
                ))
        return lat_lon_points, intensities
    
    def _calc_overlap_mask(self):
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
        if hasattr(self, '_masked'):
            return

        for i, imager in enumerate(self.imagers):
            _distances = np.nan*np.ones((*imager.skymap['lat'].shape, len(self.imagers)))
            for j, other_imager in enumerate(self.imagers):
                # Calculate the distance between all imager pixels and every other imager 
                # location (including itself).
                _distances[:, :, j] = _haversine(
                    imager.skymap['lat'], imager.skymap['lon'],
                    np.broadcast_to(other_imager.meta['lat'], imager.skymap['lat'].shape), 
                    np.broadcast_to(other_imager.meta['lon'], imager.skymap['lat'].shape)
                    )
            # Without this small reduction in the distance of pixels to its own imager,
            # there are gaps between the imager boundaries. In other words, this scaling
            # slightly biases to plotting pixels nearest to the imager. 
            _distances[:, :, i] *= 0.95
            # Need a masked array so that np.nanargmin correctly handles all NaN slices.
            _distances = np.ma.masked_array(_distances, np.isnan(_distances))
            # For each pixel, calculate the nearest imager. If the pixel is not closest to 
            # the imager that it's from, mask it as np.nan. Then the Imager._pcolormesh_nan() 
            # method then won't plot that pixel.
            min_distances = np.argmin(_distances, axis=2)
            far_pixels = np.where(min_distances != i)
            imager.skymap['lat'][far_pixels] = np.nan
            imager.skymap['lon'][far_pixels] = np.nan
        self._masked = True  # A flag to not run again.
        return