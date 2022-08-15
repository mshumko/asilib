import importlib
from typing import Tuple
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

earth_radius_km = 6371


class Conjunction:
    def __init__(self, imager, sat_time, sat_loc) -> None:
        """
        Calculates conjunctions between an imager and a satellite.

        Parameters
        ----------
        imager: asilib.Imager
            An instance of the imager class.
        sat_time: list or np.array
            An array of satellite time stamps.
        sat_loc: list or np.array
            A (nTime, 3) time series of satellite locations. Columns must map to
            (latitude, longitude, altitude) (LLA) coordinates.
        """
        assert sat_loc.shape[1] == 3, 'sat_loc must have 3 columns.'
        assert hasattr(imager, 'skymap'), 'imager does not contain a skymap.'

        self.imager = imager
        self.sat = pd.DataFrame(
            index=sat_time, data={'lat': sat_loc[:, 0], 'lon': sat_loc[:, 1], 'alt': sat_loc[:, 2]}
        )
        return

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
        lon_map, lat_map, _ = self.imager._mask_low_horizon(
            self.imager.skymap['lon'],
            self.imager.skymap['lat'],
            self.imager.skymap['el'],
            min_elevation=min_el,
        )

        # Search LLA for times when it was inside the map box
        conjunction_idx = np.where(
            (self.sat['lat'] > np.nanmin(lat_map))
            & (self.sat['lat'] < np.nanmax(lat_map))
            & (self.sat['lon'] > np.nanmin(lon_map))
            & (self.sat['lon'] < np.nanmax(lon_map))
        )[0]
        if conjunction_idx.shape[0] == 0:
            return pd.DataFrame(columns=['start', 'end'])

        start, end = self._conjunction_intervals(self.sat.index[conjunction_idx], min_dt=time_gap_s)

        df = pd.DataFrame(
            data={
                'start': self.sat.index[conjunction_idx][start],
                'end': self.sat.index[conjunction_idx][end],
            }
        )
        # # Test script
        # fig, ax = plt.subplots()
        # self.imager._pcolormesh_nan(lon_map, lat_map, 1+0*lat_map, ax)
        # ax.scatter(self.sat['lon'], self.sat['lat'], c='k')
        # ax.plot(self.sat['lon'][conjunction_idx], self.sat['lat'][conjunction_idx], 'r')
        # ax.scatter(self.sat['lon'][df['start']], self.sat['lat'][df['start']], c='g', s=100)
        # ax.scatter(self.sat['lon'][df['end']], self.sat['lat'][df['end']], c='c', s=100)
        # plt.show()
        return df

    def resample(self):
        """
        Resample the sat_loc from the satellite to imager time stamps. Useful for
        one-to-one comparison between the images a satellite trajectory.
        """
        imager_times, _ = self.imager.data(which='time')
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

        resampled_lla = {}

        for key in ['lat', 'lon', 'alt']:
            if key == 'lon':
                period = 180  # TODO: Add a test when longitude wraps around.
            else:
                period = None

            resampled_lla[key] = np.interp(
                numeric_imager_times, numeric_sat_times, self.sat.loc[:, key], period=period
            )

        # TODO: Test script, remove when done.
        # key = 'lon'
        # plt.scatter(self.sat.index, self.sat[key], c='k', s=50)
        # plt.scatter(imager_times, resampled_lla[key], c='orange')
        # plt.show()
        self.sat = pd.DataFrame(index=imager_times, data=resampled_lla)
        return

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

    def map_lla_azel(self, min_el=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps, a satellite's latitude, longitude, and altitude (LLA) coordinates
        to the ASI's azimuth and elevation (azel) coordinates and image pixel index.

        This function is useful to plot a satellite's location in the ASI image using the
        pixel indices.

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
        | from datetime import datetime
        |
        | import numpy as np
        | from asilib import lla2azel
        |
        | # THEMIS/ATHA's LLA coordinates are (54.72, -113.301, 676 (meters)).
        | # The LLA is a North-South pass right above ATHA..
        | n = 50
        | lats = np.linspace(60, 50, n)
        | lons = -113.64*np.ones(n)
        | alts = 500**np.ones(n)
        | lla = np.array([lats, lons, alts]).T
        |
        | time = datetime(2015, 10, 1)  # To load the proper skymap file.
        |
        | azel, pixels = lla2azel('REGO', 'ATHA', time, lla)
        """
        self.sat_azel = np.nan * np.zeros((self.sat.shape[0], 2))

        # TODO: Fix a bug where the path is wrong. (See SAMPEX-GILL conjunction
        # at 10:59:20-11:01:08, SAMPEX-GILL 2009-03-06 07:33:48-07:35:36)

        # Loop over every set of LLA coordinates and use pymap3d.geodetic2aer
        # to map to the azimuth and elevation.
        for i, (time, (lat_i, lon_i, alt_km_i)) in enumerate(self.sat.iterrows()):
            # Check if lat, lon, or alt is nan or -1E31
            # (the error value in the IRBEM library).
            any_nan = bool(len(np.where(np.isnan([lat_i, lon_i, alt_km_i]))[0]))
            any_neg = bool(len(np.where(np.array([lat_i, lon_i, alt_km_i]) == -1e31)[0]))
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
            self.sat_azel[i, :] = [az, el]

        # Now find the corresponding x- and y-axis pixel indices.
        self.asi_pixels = self._map_azel_to_pixel(self.sat_azel)

        # Mask elevations below min_el as NaN. This is a good idea because
        # _map_azel_to_pixel will find the nearest pixel even if the delta
        # az and delta el is large.
        invalid_el = np.where(self.sat_azel[:, 1] < min_el)[0]
        self.sat_azel[invalid_el, :] = np.nan
        self.asi_pixels[invalid_el, :] = np.nan
        return self.sat_azel, self.asi_pixels

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

    def _map_azel_to_pixel(self, sat_azel, chunk_size=1000) -> np.ndarray:
        """
        Locate the nearest ASI skymap (x, y) pixel indices.

        Parameters
        ----------
        sat_azel: array
            A 1d or 2d array of satelite azimuth and elevation points.
            If 2d, the rows correspons to time.
        chunk_size: int
            The chunks to calculate the 2d distance array. This is necessary
            to maintain a small memory footprint.

        Returns
        -------
        pixel_index: np.ndarray
            An array with the same shape as sat_azel, but representing the
            x- and y-axis pixel indices in the ASI image.
        """
        az_coords = self.imager.skymap['az'].ravel().copy()
        az_coords[np.isnan(az_coords)] = -10000
        el_coords = self.imager.skymap['el'].ravel().copy()
        el_coords[np.isnan(el_coords)] = -10000
        asi_azel_cal = np.stack((az_coords, el_coords), axis=-1)

        pixel_index = np.nan * np.ones_like(sat_azel)
        # Find the distance between the satellite azel points and
        # the asi_azel points. dist_matrix[i,j] is the distance
        # between ith asi_azel_cal value and jth sat_azel.

        if sat_azel.shape[0] < chunk_size:
            x_pixel, y_pixel = self._nearest_pixel(asi_azel_cal, sat_azel)
            pixel_index[:, 0] = x_pixel
            pixel_index[:, 1] = y_pixel
        else:
            for i in range(sat_azel.shape[0] // chunk_size):
                sat_azel_chunk = sat_azel[i * chunk_size : (i + 1) * chunk_size, :]
                x_pixel, y_pixel = self._nearest_pixel(asi_azel_cal, sat_azel_chunk)
                pixel_index[i * chunk_size : (i + 1) * chunk_size, 0] = x_pixel
                pixel_index[i * chunk_size : (i + 1) * chunk_size, 1] = y_pixel
            # The unchunked remainder
            sat_azel_chunk = sat_azel[(i + 1) * chunk_size :, :]
            x_pixel, y_pixel = self._nearest_pixel(asi_azel_cal, sat_azel_chunk)
            pixel_index[(i + 1) * chunk_size :, 0] = x_pixel
            pixel_index[(i + 1) * chunk_size :, 1] = y_pixel
        return pixel_index

    def _nearest_pixel(self, asi_azel_cal, sat_azel):
        dist_matrix = scipy.spatial.distance.cdist(asi_azel_cal, sat_azel, metric='euclidean')
        # Now find the minimum distance for each sat_azel.
        idx_min_dist = np.argmin(dist_matrix, axis=0)
        # if idx_min_dist == 0, it is very likely to be a NaN value
        # because np.argmin returns 0 if a row has a NaN.
        idx_min_dist = np.array(idx_min_dist, dtype=object)
        idx_min_dist[idx_min_dist == 0] = np.nan
        # For use the 1D index for the flattened ASI skymap
        # to get out the azimuth and elevation pixels.
        pixel_index = np.nan * np.ones_like(sat_azel)
        x_pixel = np.remainder(idx_min_dist, self.imager.skymap['az'].shape[1])
        y_pixel = np.floor_divide(idx_min_dist, self.imager.skymap['az'].shape[1])
        return x_pixel, y_pixel

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


if __name__ == '__main__':
    import dateutil.parser
    from datetime import datetime, timedelta

    import asilib

    t0 = dateutil.parser.parse('2014-05-05T04:49:10')

    n_times = 100
    img = asilib.themis('gill', time='2014-05-05T04:49:10', load_images=False, alt=110)
    lats = np.linspace(90, 0, num=n_times)
    lons = img.meta['lon'] * np.ones_like(lats)
    alts = 110 * np.ones_like(lats)
    lla = np.array([lats, lons, alts]).T
    times = np.array([t0 + timedelta(seconds=i * 0.5) for i in range(n_times)])
    c = asilib.Conjunction(img, times, lla)
    c.find()
