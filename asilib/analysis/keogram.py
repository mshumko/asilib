import numpy as np
import pandas as pd
import warnings
import aacgmv2

from asilib.io import utils
from asilib.io.load import (
    load_image_generator,
    load_skymap,
    _create_empty_data_arrays,
)


def keogram(
    asi_array_code: str,
    location_code: str,
    time_range: utils._time_range_type,
    map_alt: int = None,
    path: np.array = None,
    aacgm: bool = False,
):
    """
    Makes a keogram pd.DataFrame along the central meridian.

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
        skymap data. If None, will plot pixel index for the y-axis.
    path: array
        Make a keogram along a custom path. Path shape must be (n, 2) and contain the
        lat/lon coordinates that are mapped to map_alt. If the map_alt kwarg is
        unspecified, this function will raise a ValueError.
    aacgm: bool
        Map the keogram latitudes to Altitude Adjusted Corrected Geogmagnetic Coordinates
        (aacgmv2) derived by Shepherd, S. G. (2014), Altitude-adjusted corrected geomagnetic
        coordinates: Definition and functional approximations, Journal of Geophysical
        Research: Space Physics, 119, 7501-7521, doi:10.1002/2014JA020264.

    Returns
    -------
    keo: pd.DataFrame
        The 2d keogram with the time index. The columns are the geographic latitude
        if map_alt != None, otherwise it is the image pixel values (0-265) or (0-512).

    Raises
    ------
    AssertionError
        If map_alt does not equal the mapped altitudes in the skymap mapped values.
    ValueError
        If no images are in ``time_range``.
    ValueError
        If a custom path is provided but not map_alt.
    """
    keo = Keogram(asi_array_code, location_code, time_range)
    return keo.keogram(map_alt, path, aacgm)


def ewogram(
    asi_array_code: str, location_code: str, time_range: utils._time_range_type, map_alt: int = None
):
    """
    Makes a East-West ewogram.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    map_alt: int
        The mapping altitude, in kilometers, used to index the mapped longitude in the
        skymap data. If None, will plot pixel index for the y-axis.

    Returns
    -------
    ewo: pd.DataFrame
        The 2d ewogram with the time index. The columns are the geographic longitude
        if map_alt != None, otherwise it is the image pixel indices.

    Raises
    ------
    AssertionError
        If map_alt does not equal the mapped altitudes in the skymap mapped values.
    ValueError
        If no imager data is found in ``time_range``.
    """
    # TODO: Add tests for an ewogram.
    image_generator = load_image_generator(asi_array_code, location_code, time_range)
    ewo_times, ewo = _create_empty_data_arrays(asi_array_code, time_range, 'keogram')

    start_time_index = 0
    for file_image_times, file_images in image_generator:
        end_time_index = start_time_index + file_images.shape[0]
        ewo[start_time_index:end_time_index, :] = file_images[:, ewo.shape[1] // 2, :]

        ewo_times[start_time_index:end_time_index] = file_image_times
        start_time_index += file_images.shape[0]

    # This code block removes any filler nan values if the ASI images were not sampled at the instrument
    # cadence throughout time_range.
    i_valid = np.where(~np.isnan(ewo[:, 0]))[0]
    ewo = ewo[i_valid, :]
    ewo_times = ewo_times[i_valid]

    if not ewo.shape[0]:
        raise ValueError(
            f'The keogram is empty for {asi_array_code}/{location_code} '
            f'during {time_range}. The image data probably does not exist '
            f'in this time interval'
        )

    if map_alt is None:
        ewo_longitude = np.arange(ewo.shape[0])  # Dummy index values for longitudes.
    else:
        skymap = load_skymap(asi_array_code, location_code, time_range[0])
        assert (
            map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
        ), f'{map_alt} km is not in skymap altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
        alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]
        ewo_longitude = skymap['FULL_MAP_LONGITUDE'][alt_index, ewo.shape[0] // 2, :]

        # ewo_longitude array are at the pixel edges. Remap it to the centers
        dl = ewo_longitude[1:] - ewo_longitude[:-1]
        ewo_longitude = ewo_longitude[0:-1] + dl / 2

        # Since keogram_latitude values are NaNs near the image edges, we want to filter
        # out those indices from keogram_latitude and keo.
        valid_lons = np.where(~np.isnan(ewo_longitude))[0]
        ewo_longitude = ewo_longitude[valid_lons]
        keo = ewo[:, valid_lons]
    return pd.DataFrame(data=keo, index=ewo_times, columns=ewo_longitude)


class Keogram:
    # TODO: Change the asi args to asi_info dictionary that is created by asilib.themis() or asilib.rego() functions.
    def __init__(
        self, asi_array_code: str, location_code: str, time_range: utils._time_range_type
    ) -> None:
        """
        Keogram class that makes keograms and ewograms along the meridian or a custom path.

        Parameters
        ----------
        asi_array_code: str
            The imager array name, i.e. ``THEMIS`` or ``REGO``.
        location_code: str
            The ASI station code, i.e. ``ATHA``
        time_range: list of datetime.datetimes or stings
            Defined the duration of data to download. Must be of length 2.
        """
        # __init__ saves the variables needed to get the images.
        self.asi_array_code = asi_array_code
        self.location_code = location_code
        self.time_range = utils._validate_time_range(time_range)

        # In the full implementation we won't need these if-else statements.
        if self.asi_array_code.lower() == 'themis':
            self.img_size = 256
            self.asi_cadence_s = 3
        elif self.asi_array_code.lower() == 'rego':
            self.img_size = 512
            self.asi_cadence_s = 3
        else:
            raise NotImplementedError
        return

    def keogram(self, map_alt: int = None, path: np.array = None, aacgm=False):
        """
        Creates a keogram along the meridian or a custom path.

        Parameters
        ----------
        map_alt: int
            The mapping altitude, in kilometers, used to index the mapped latitude in the
            skymap data. If None, will plot pixel index for the y-axis.
        path: array
            Make a keogram along a custom path. Path shape must be (n, 2) and contain the
            lat/lon coordinates that are mapped to map_alt. If the map_alt kwarg is
            unspecified, this function will raise a ValueError.
        aacgm: bool
            Map the keogram latitudes to Altitude Adjusted Corrected Geogmagnetic Coordinates
            (aacgmv2) derived by Shepherd, S. G. (2014), Altitude-adjusted corrected geomagnetic
            coordinates: Definition and functional approximations, Journal of Geophysical
            Research: Space Physics, 119, 7501-7521, doi:10.1002/2014JA020264.

        Returns
        -------
        keo: pd.DataFrame
            The 2d keogram with the time index. The columns are the (geographic or magnetic)
            latitudes if map_alt is specified, otherwise it is the image pixel indices.

        Raises
        ------
        AssertionError
            If map_alt does not equal the mapped altitudes in the skymap mapped values.
        ValueError
            If no images are in ``time_range``.
        ValueError
            If a custom path is provided but not map_alt.
        """
        self.map_alt = map_alt
        self.aacgm = aacgm
        self.path = path
        keo_times, self._keo = self._empty_keogram(self.time_range)
        self.skymap = load_skymap(self.asi_array_code, self.location_code, self.time_range[0])

        # Determine what pixels to slice
        self._keogram_pixels()
        # Not all of the pixels are valid (e.g. below the horizon)
        self._keo = self._keo[:, : self._pixels.shape[0]]

        self._keogram_latitude()

        # Load and slice images.
        image_generator = load_image_generator(
            self.asi_array_code, self.location_code, self.time_range
        )
        start_time_index = 0
        for file_image_times, file_images in image_generator:
            # Check if the image time stamps are out of order. Otherwise this will cause
            # a broadcast numpy error.
            dt = np.array(
                [
                    (f - i).total_seconds()
                    for i, f in zip(file_image_times[:-1], file_image_times[1:])
                ]
            )
            if not np.all(dt > 0):
                raise ValueError(
                    f'{self.asi_array_code}-{self.location_code} time stamps are out of order.'
                )
            end_time_index = start_time_index + file_images.shape[0]
            self._keo[start_time_index:end_time_index, :] = file_images[
                :, self._pixels[:, 0], self._pixels[:, 1]
            ]
            keo_times[start_time_index:end_time_index] = file_image_times
            start_time_index += file_images.shape[0]

        # Remove NaN times.
        i_valid = np.where(~np.isnan(self._keo[:, 0]))[0]
        keo = self._keo[i_valid, :]
        keo_times = keo_times[i_valid]

        if not keo.shape[0]:
            raise ValueError(
                f'The keogram is empty for {self.asi_array_code}/{self.location_code} '
                f'during {self.time_range}. The image data probably does not exist '
                f'in this time interval'
            )
        self.keo = pd.DataFrame(data=keo, index=keo_times, columns=self.keogram_latitude)
        return self.keo

    def _empty_keogram(self, time_range):
        """
        Creates an empty 2D keogram and time arrays.
        """
        time_range = utils._validate_time_range(time_range)
        # + 1 just in case. The THEMIS and REGO cadence is not
        # always exactly 3 seconds, so in certain circumstances
        # there may be one or two extra data points.
        max_n_timestamps = (
            int((time_range[1] - time_range[0]).total_seconds() / self.asi_cadence_s) + 1
        )
        data_shape = (max_n_timestamps, self.img_size)

        # object is the only dtype that can contain datetime objects
        times = np.nan * np.zeros(max_n_timestamps, dtype=object)
        data = np.nan * np.zeros(data_shape)
        return times, data

    def _keogram_pixels(self, minimum_elevation=0):
        """
        Find what pixels to index and reshape the keogram.
        """
        # CASE 1: No path provided. Output self._pixels that slice the meridian.
        if self.path is None:
            self._pixels = np.column_stack(
                (
                    np.arange(self._keo.shape[1]),
                    self._keo.shape[1] * np.ones(self._keo.shape[1]) // 2,
                )
            ).astype(int)

        # CASE 2: A path is provided so now we need to calculate the custom path
        # on the lat/lon skymap.
        else:
            if (self.map_alt is None) or (
                self.map_alt not in self.skymap['FULL_MAP_ALTITUDE'] / 1000
            ):
                raise ValueError(
                    f'{self.map_alt} km is not in skymap altitudes: {self.skymap["FULL_MAP_ALTITUDE"]/1000} km'
                )
            alt_index = np.where(self.skymap['FULL_MAP_ALTITUDE'] / 1000 == self.map_alt)[0][0]
            self._pixels = self._path_to_pixels(self.path, alt_index)

        above_elevation = np.where(
            self.skymap['FULL_ELEVATION'][self._pixels[:, 0], self._pixels[:, 1]]
            >= minimum_elevation
        )[0]
        self._pixels = self._pixels[above_elevation]
        return

    def _path_to_pixels(self, path, alt_index, threshold=1):
        """
        Convert the lat/lon path that is mapped to map_alt in kilometers to
        the x- and y-pixels in the skymap lat/lon mapped file.

        Parameters
        ----------
        path: np.array
            The lat/lon array of shape (n, 2).
        alt_index: int
            Determines what altitude index to use.
        threshold: float
            The maximum distance threshold, in degrees, between the path (lat, lon)
            and the skymap (lat, lon)

        Returns
        -------
        np.array
            The valid x pixels corresponding to the path.
        np.array
            The valid y pixels corresponding to the path.
        np.array
            Path indices corresponding to rows with a pixel within threshold
            degrees distance.
        """
        if np.array(path).shape[0] == 0:
            raise ValueError('The path is empty.')
        if np.any(np.isnan(path)):
            raise ValueError("The lat/lon path can't contain NaNs.")
        if np.any(np.max(path) > 180) or np.any(np.min(path) < -180):
            raise ValueError("The lat/lon values must be in the range -180 to 180.")

        nearest_pixels = np.nan * np.zeros_like(path)

        for i, (lat, lon) in enumerate(path):
            distances = np.sqrt(
                (self.skymap['FULL_MAP_LATITUDE'][alt_index, :, :] - lat) ** 2
                + (self.skymap['FULL_MAP_LONGITUDE'][alt_index, :, :] - lon) ** 2
            )
            idx = np.where(distances == np.nanmin(distances))

            # Keep NaNs if distanace is larger than threshold.
            if distances[idx][0] > threshold:
                warnings.warn(
                    f'Some of the keogram path coordinates are outside of the '
                    f'maximum {threshold} degrees distance from the nearest '
                    f'skymap map coordinate.'
                )
                continue
            nearest_pixels[i, :] = [idx[0][0], idx[1][0]]

        valid_pixels = np.where(np.isfinite(nearest_pixels[:, 0]))[0]
        if valid_pixels.shape[0] == 0:
            raise ValueError('The keogram path is completely outside of the skymap.')
        return nearest_pixels[valid_pixels, :].astype(int)

    def _keogram_latitude(self):
        """
        Keogram's vertical axis: geographic latitude, magnetic latitude, or pixel index.
        """
        if self.map_alt is None:
            self.keogram_latitude = np.arange(self._keo.shape[1])
        else:
            alt_indices = np.where(self.skymap['FULL_MAP_ALTITUDE'] / 1000 == self.map_alt)[0]
            assert len(alt_indices) == 1, (
                f"{self.map_alt} is not in the skymap altitudes: "
                f"{self.skymap['FULL_MAP_ALTITUDE'] / 1000}"
            )
            self.keogram_latitude = self.skymap['FULL_MAP_LATITUDE'][
                alt_indices[0], self._pixels[:, 0], self._pixels[:, 1]
            ]
            if self.aacgm:
                longitudes = self.skymap['FULL_MAP_LONGITUDE'][
                    alt_indices[0], self._pixels[:, 0], self._pixels[:, 1]
                ]
                self.keogram_latitude = aacgmv2.convert_latlon_arr(
                    self.keogram_latitude,
                    longitudes,
                    self.map_alt,
                    self.time_range[0],
                    method_code="G2A",
                )[0]
        return
