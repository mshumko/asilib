import importlib
from typing import Tuple, Union
import warnings

import numpy as np
import pandas as pd
from matplotlib.dates import date2num
import matplotlib.pyplot as plt  # TODO: Remove when done testing.
import pymap3d
import scipy

try:
    import IRBEM
except ImportError:
    pass  # make sure that asilb.__init__ fully loads and crashes if the user calls asilib.lla2footprint()

import asilib

earth_radius_km = 6371


class Conjunction:
    def __init__(self, imager: asilib.Imager, satellite: Union[tuple, pd.DataFrame]) -> None:
        """
        Calculates conjunctions between an imager and a satellite.

        Parameters
        ----------
        imager: asilib.Imager
            An instance of the imager class.
        satellite: tuple or pd.Dataframe
            Satellite time stamps and position in the latitude, longitude, altitude (LLA) coordinates.

            If a tuple, the first element must be a np.array of shape (n) filled with
            datetime.datetime() timestamps. The second tuple element must be a np.array
            of shape (n, 3) with the satellite positions with the columns mapping to
            (latitude, longitude, altitude), in that order. This tuple is converted to a
            pd.Dataframe described below.

            If a pd.Dataframe, the index must be pd.Timestamp() and the columns correspond to
            (latitude, longitude, altitude). The columns do not need to be in that order, but are
            automatically distinguished by searching for columns that match "lat", "lon", and "alt"
            (case insensitive).
        """
        self.imager = imager
        assert hasattr(imager, 'skymap'), 'imager does not contain a skymap.'

        assert isinstance(
            satellite, (tuple, pd.DataFrame)
        ), 'satellite must be a tuple or pd.Dataframe.'
        if isinstance(satellite, tuple):
            assert len(satellite) == 2, 'Satellite tuple must have length of 2'
            assert isinstance(
                satellite[0], np.ndarray
            ), 'satellite tuple must contain two np.arrays'
            assert isinstance(
                satellite[1], np.ndarray
            ), 'satellite tuple must contain two np.arrays'
            assert (
                satellite[1].shape[1] == 3
            ), '2nd element in the satellite tuple must have 3 columns.'
            self.sat = pd.DataFrame(
                index=satellite[0],
                data={
                    'lat': satellite[1][:, 0],
                    'lon': satellite[1][:, 1],
                    'alt': satellite[1][:, 2],
                },
            )
        else:  # the pd.Dataframe case.
            self.sat = self._rename_satellite_df_columns(satellite)
        if np.nanmax(self.sat['lon']) > 180:
            raise ValueError('Satellite longitude must be in the range -180 to 180 degrees.')
        return

    def _rename_satellite_df_columns(self, df: pd.DataFrame):
        """
        Detect and rename the satellite df columns to lat, lon, and alt.
        """
        lon_keys = [key for key in df.columns if 'lon' in key.lower()]
        lat_keys = [key for key in df.columns if 'lat' in key.lower()]
        alt_keys = [key for key in df.columns if 'alt' in key.lower()]
        assert len(lon_keys) == 1, f'{len(lon_keys)} longitude columns found from {df.columns}.'
        assert len(lat_keys) == 1, f'{len(lat_keys)} latitude columns found from {df.columns}.'
        assert len(alt_keys) == 1, f'{len(alt_keys)} altitude columns found from {df.columns}.'
        df = df.rename(columns={lon_keys[0]: 'lon', lat_keys[0]: 'lat', alt_keys[0]: 'alt'})
        return df

    def find(self, min_el=20, time_gap_s=60):
        """
        Finds the start and end times of conjunctions defined by a minimum elevation.

        Parameters
        ----------
        min_el: float
            The minimum elevation of the conjunction.
        """
        assert 0 < min_el < 90, "The minimum elevation must be between 0 and 90 degrees."

        # # Filter the elevation map to values > min_el
        self._lon_map, self._lat_map, _ = self.imager._mask_low_horizon(
            self.imager.skymap['lon'],
            self.imager.skymap['lat'],
            self.imager.skymap['el'],
            min_elevation=min_el,
        )

        # Search LLA for times when it was inside the map box
        conjunction_idx = np.where(
            (self.sat['lat'] > np.nanmin(self._lat_map))
            & (self.sat['lat'] < np.nanmax(self._lat_map))
            & (self.sat['lon'] > np.nanmin(self._lon_map))
            & (self.sat['lon'] < np.nanmax(self._lon_map))
        )[0]
        if conjunction_idx.shape[0] == 0:
            return pd.DataFrame(columns=['start_time', 'end_time', 'start_index', 'end_index'])

        start, end = self._conjunction_intervals(self.sat.index[conjunction_idx], min_dt=time_gap_s)

        df = pd.DataFrame(
            data={
                'start_time': self.sat.index[conjunction_idx][start],
                'end_time': self.sat.index[conjunction_idx][end],
                'start_index': conjunction_idx[start],
                'end_index': conjunction_idx[end],
            }
        )
        return df

    def interp_sat(self):
        """
        Interpolate the satellite timestamps and LLA at the imager timestamps.
        """
        imager_times, _ = self.imager.data
        numeric_imager_times = date2num(imager_times)
        numeric_sat_times = date2num(self.sat.index)
        if len(imager_times) == 0:
            raise ValueError('No imager time stamps to interpolate over.')

        assert np.all(
            np.diff(numeric_imager_times) > 0
        ), 'Imager times are not strictly increasing.'
        assert np.all(
            np.diff(numeric_sat_times) > 0
        ), 'satellite times are not strictly increasing.'

        interpolated_lla = {}

        # TODO: Detect when longitudes cross the 180-meridian and
        # correctly interpolate. Use the FIREBIRD data_processing code.
        for key in ['lat', 'lon', 'alt']:
            interpolated_lla[key] = np.interp(
                numeric_imager_times,
                numeric_sat_times,
                self.sat.loc[:, key],
                left=np.nan,
                right=np.nan,
            )
        if np.nanmax(np.abs(np.diff(self.sat.loc[:, 'lon']))) > 200:
            warnings.warn(
                'The asilib.Conjunction.interp_sat() does not yet correctly interpolate'
                'longitudes across the 180-meridian.',
                UserWarning,
            )
        self.sat = pd.DataFrame(index=imager_times, data=interpolated_lla)
        return self.sat

    def map_lla_footprint(
        self,
        map_alt: float,
        b_model: str = 'OPQ77',
        maginput: dict = None,
        hemisphere: int = 0,
    ) -> np.ndarray:
        """
        Map the spacecraft's position to ``map_alt`` along the magnetic field line.
        The mapping is implemeneted in ``IRBEM`` and by default it maps to the same
        hemisphere.

        Parameters
        ----------
        map_alt: float
            The altitude to map to, in km, in the same hemisphere.
        b_model: str
            The magnetic field model to use, by default the model is Olson-Pfitzer
            1974. This parameter is passed directly into IRBEM.MagFields as the
            'kext' parameter.
        maginput: dict
            If you use a differnet b_model that requires time-dependent parameters,
            supply the appropriate values to the maginput dictionary. It is directly
            passed into IRBEM.MagFields so refer to IRBEM on the proper format.
        hemisphere: int
            The hemisphere to map to. This kwarg is passed to IRBEM and can be one of
            these four values:
            0    = same magnetic hemisphere as starting point
            +1   = northern magnetic hemisphere
            -1   = southern magnetic hemisphere
            +2   = opposite magnetic hemisphere as starting point

        Returns
        -------
        magnetic_footprint: np.ndarray
            A numpy.array with size (n_times, 3) with lat, lon, alt
            columns representing the magnetic footprint coordinates.

        Raises
        ------
        ImportError
            If IRBEM can't be imported.
        """
        if importlib.util.find_spec('IRBEM') is None:
            raise ImportError(
                "IRBEM can't be imported. This is a required dependency for asilib.lla2footprint()"
                " that must be installed separately. See https://github.com/PRBEM/IRBEM"
                " and https://aurora-asi-lib.readthedocs.io/en/latest/installation.html."
            )

        magnetic_footprint = np.nan * np.zeros((self.sat.shape[0], 3))

        m = IRBEM.MagFields(kext=b_model)  # Initialize the IRBEM model.

        # Loop over every set of satellite coordinates.
        for i, (time, (lat, lon, alt)) in enumerate(self.sat.iterrows()):
            X = {'datetime': time, 'x1': alt, 'x2': lat, 'x3': lon}
            m_output = m.find_foot_point(X, maginput, map_alt, hemisphere)
            magnetic_footprint[i, :] = m_output['XFOOT']
        # Map from IRBEM's (alt, lat, lon) -> (lat, lon, alt)
        # magnetic_footprint[:, [2, 0, 1]] = magnetic_footprint[:, [0, 1, 2]]
        self.sat.loc[:, ['alt', 'lat', 'lon']] = magnetic_footprint
        return

    def map_azel(self, min_el=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps a satellite's location to the ASI's azimuth and elevation (azel)
        coordinates and image pixel index.

        Parameters
        ----------
        min_el: float
            The minimum elevation in degrees for which return valid values for.
            The satellite's azel values and pixel indices are NaN below min_el.

        Returns
        -------
        np.ndarray
            An array with shape (nPosition, 2) of the satellite's azimuth and
            elevation coordinates.
        np.ndarray
            An array with shape (nPosition, 2) of the x- and y-axis pixel
            indices for the ASI image.

        Raises
        ------
        AssertionError
            If the sat_lla argument does not have exactly 3 columns (1st dimension).

        Example
        -------
        """
        azel = np.nan * np.zeros((self.sat.shape[0], 2))

        for i, (time, (lat_i, lon_i, alt_km_i)) in enumerate(self.sat.iterrows()):
            any_nan = np.any(np.isnan([lat_i, lon_i, alt_km_i]))
            any_neg = np.any([lat_i, lon_i, alt_km_i] == -1e31)  # IRBEM error value.
            if any_nan or any_neg:
                continue

            az, el, _ = pymap3d.geodetic2aer(
                lat_i,
                lon_i,
                1e3 * alt_km_i,  # Meters
                self.imager.meta['lat'],
                self.imager.meta['lon'],
                1e3 * self.imager.meta['alt'],  # Meters
            )
            azel[i, :] = [az, el]

        # Now find the corresponding x- and y-axis pixel indices.
        azel_pixels = self._nearest_pixel(azel)

        # Mask elevations below min_el as NaN. This is necessary because
        # _nearest_pixel will find the nearest pixel even if the delta
        # az and delta el is large.
        invalid_el = np.where(azel[:, 1] < min_el)[0]
        azel[invalid_el, :] = np.nan
        azel_pixels[invalid_el, :] = np.nan
        return azel, azel_pixels

    def equal_area(self, box=(5, 5)):
        """
        Calculate a ``box_km`` area mask at the aurora emission altitude.

        Parameters
        ----------
        box: tuple
            Bounds the emission box dimensions in longitude and latitude.
            Units are kilometers.

        Returns
        -------
        pixel_mask: np.ndarray
            An array with (n_time, n_x_pixels, n_y_pixels) dimensions with
            dimensions n_x_pixels and n_y_pixels dimensions the size of each
            image. Values inside the area are 1 and outside are np.nan.
        """
        n_max = 5000
        if self.sat.shape[0] > n_max:
            warnings.warn(
                f'The {self.sat.shape[0]} footprints is larger than {n_max}. '
                f'Use the equal_area_gen() method instead.'
            )
        assert len(box) == 2, 'The box_km parameter must have a length of 2.'

        pixel_mask = np.nan * np.zeros((self.sat.shape[0], *self.imager.meta['resolution']))

        gen = self.equal_area_gen(box=box)

        for i, mask in enumerate(gen):
            # Check that the lat/lon values are inside the skymap coordinates.
            pixel_mask[i, :, :] = mask
        return pixel_mask

    def equal_area_gen(self, box=(5, 5)):
        """
        A generator function to calculate a ``box_km`` area mask at the
        aurora emission altitude for a large number of footprints.

        Parameters
        ----------
        box: tuple
            Bounds the emission box dimensions in longitude and latitude.
            Units are kilometers.

        Returns
        -------
        pixel_mask: np.ndarray
            An array with (n_time, n_x_pixels, n_y_pixels) dimensions with
            dimensions n_x_pixels and n_y_pixels dimensions the size of each
            image. Values inside the area are 1 and outside are np.nan.
        """
        assert len(box) == 2, 'The box_km parameter must have a length of 2.'

        lat_map = self.imager.skymap['lat']
        lon_map = self.imager.skymap['lon']

        for lat, lon, alt in self.sat.to_numpy():
            mask = np.nan * np.zeros(self.imager.meta['resolution'])
            dlat = self._dlat(box[1], alt)
            dlon = self._dlon(box[0], alt, lat)
            # Check that the lat/lon values are inside the skymap coordinates.
            if (
                (lat > np.nanmax(lat_map))
                or (lat < np.nanmin(lat_map))
                or (lon > np.nanmax(lon_map))
                or (lon < np.nanmin(lon_map))
                or np.isnan(lat)
                or np.isnan(lon)
                or np.isnan(alt)
            ):
                warnings.warn(
                    'Some latitude or longitude values are outside of the skymap '
                    'lat/lon arrays or are invalid. The equal area mask will be '
                    'all NaNs.'
                )
                yield mask
                continue

            # Find the indices of the box. If none were found (pixel smaller than
            # the box_size_km) then increase the box size until one pixel is found.
            masked_box_len = 0
            multiplier = 1
            step = 0.1

            while masked_box_len == 0:
                idx_box = np.where(
                    (lat_map >= lat - multiplier * dlat / 2)
                    & (lat_map <= lat + multiplier * dlat / 2)
                    & (lon_map >= lon - multiplier * dlon / 2)
                    & (lon_map <= lon + multiplier * dlon / 2)
                )

                masked_box_len = len(idx_box[0])
                if masked_box_len:
                    mask[idx_box[0], idx_box[1]] = 1
                else:
                    multiplier += step
            yield mask

    def _dlat(self, d, alt):
        """
        Calculate the change in latitude that correpsponds to arc length distance d at
        alt altitude. Units are kilometers. Both d and alt must be the same length.

        Parameters
        ----------
        d: float or np.ndarray
            A float, 1d list, or 1d np.array of arc length.
        alt: float or np.ndarray
            A float, 1d list, or 1d np.array of satellite altitudes.

        Returns
        -------
        dlat: float or np.ndarray
            A float or an array of the corresponding latitude differences in degrees.
        """
        if isinstance(alt, list):  # Don't need to cast d since it is in np.divide().
            alt = np.array(alt)
        return np.rad2deg(np.divide(d, (earth_radius_km + alt)))

    def _dlon(self, d, alt, lat):
        """
        Calculate the change in longitude that corresponds to arc length distance d at
        a lat latitude, and alt altitude.

        Parameters
        ----------
        d: float or np.ndarray
            A float or an array of arc length in kilometers.
        alt: float or np.ndarray
            A float or an array of satellite altitudes in kilometers.
        lat: float
            The latitude to evaluate the change in longitude in degrees.

        Returns
        -------
        dlon: float or np.ndarray
            A float or an array of the corresponding longitude differences in degrees.
        """
        if isinstance(alt, list):  # Don't need to cast other variables.
            alt = np.array(alt)

        numerator = np.sin(d / (2 * (earth_radius_km + alt)))
        denominator = np.cos(np.deg2rad(lat))
        dlon_rads = 2 * np.arcsin(numerator / denominator)
        return np.rad2deg(dlon_rads)

    def _nearest_pixel(self, azel, n_max_vectorize=10_000) -> np.ndarray:
        """
        Locate the nearest ASI skymap (x, y) pixel indices.

        Parameters
        ----------
        azel: array
            A 1d or 2d array of satellite azimuth and elevation points.
            If 2d, the rows corresponds to time.
        n_max_vectorize: int
            The maximum number of azel values before switching from the vectorized
            mode to the for-loop mode.

        Returns
        -------
        np.ndarray
            An array with the same shape as azel, but representing the
            x- and y-axis pixel indices in the ASI image.
        """
        pixel_index = np.nan * np.ones_like(azel)

        if azel.shape[0] < n_max_vectorize:
            skymap_shape = self.imager.skymap['el'].shape
            # Broadcast the skymaps such that the skymaps are copies along dim 0.
            # I need to read up on how broadcasting works in numpy, but it should
            # be very memory efficient.
            asi_el = np.broadcast_to(
                self.imager.skymap['el'][np.newaxis, :, :],
                (azel.shape[0], skymap_shape[0], skymap_shape[1]),
            )
            asi_az = np.broadcast_to(
                self.imager.skymap['az'][np.newaxis, :, :],
                (azel.shape[0], skymap_shape[0], skymap_shape[1]),
            )
            sat_az = azel[:, 0]
            sat_az = np.broadcast_to(sat_az[:, np.newaxis, np.newaxis], asi_el.shape)
            sat_el = azel[:, 1]
            sat_el = np.broadcast_to(sat_el[:, np.newaxis, np.newaxis], asi_el.shape)
            distances = self._haversine_distance(asi_el, asi_az, sat_el, sat_az)
            # This should be vectorized too, but np.nanargmin() can't find a minimum
            # along the 1st and 2nd dims.
            for i, distance in enumerate(distances):
                if np.all(np.isnan(distance)):
                    continue
                pixel_index[i, :] = np.unravel_index(np.nanargmin(distance), distance.shape)
        else:
            # Calculate one at a time to save memory.
            for i, (az, el) in enumerate(azel):
                el = el * np.ones_like(self.imager.skymap['el'])
                az = el * np.ones_like(self.imager.skymap['az'])
                distances = self._haversine_distance(
                    self.imager.skymap['el'], self.imager.skymap['az'], el, az
                )
                pixel_index[i, :] = np.unravel_index(np.nanargmin(distances), distances.shape)
        return pixel_index

    def _haversine_distance(
        self, lat1: np.array, lon1: np.array, lat2: np.array, lon2: np.array, r: float = 1
    ) -> np.array:
        """
        Calculate the distance between two points specified by latitude and longitude. All inputs
        must be the same shape.

        Parameters
        ----------
        lat1, lat2: np.array
            The latitude of points 1 and 2m in units of degrees. Can be n-dimensional.
        lon1, lon2: np.array
            The longitude of points 1 and 2m in units of degrees. Can be n-dimensional.
        r: float
            The sphere radius.
        """
        assert lat1.shape == lon1.shape == lat2.shape == lon2.shape, (
            'All input arrays' ' must have the same shape.'
        )
        lat1_rad = np.deg2rad(lat1)
        lat2_rad = np.deg2rad(lat2)
        lon1_rad = np.deg2rad(lon1)
        lon2_rad = np.deg2rad(lon2)

        d = (
            2
            * r
            * np.arcsin(
                np.sqrt(
                    np.sin((lat2_rad - lat1_rad) / 2) ** 2
                    + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin((lon2_rad - lon1_rad) / 2)
                )
            )
        )
        return d

    def _conjunction_intervals(self, times: np.array, min_dt: float):
        """
        Given an array of times that contain continuously incrementing spans of
        time with gaps between them, calculate the start and end indices of each
        time span.

        Parameters
        ----------
        times: np.array
            The array of continuos times with gaps.
        min_dt: float
            The minimum time gap threshold.

        Returns
        -------
        np.array
            An array of start indices for times
        np.array
            An array of end indices for times
        """
        dt = (times[1:] - times[:-1]).total_seconds()
        dIntervals = dt / min_dt
        # Calculate the start and end indices excluding the first and last index
        # dIntervals > 1 means tha a time gap exceeded the threshold.
        break_indices = np.where(dIntervals > 1)[0]

        # Add in the first and last index.
        start = np.insert(break_indices + 1, 0, 0)
        end = np.append(break_indices, len(dIntervals))
        return start, end
