from datetime import datetime
import pathlib

import numpy as np
import skyfield.api
import cdflib
import scipy.spatial

from asi.load import load_cal_file


def lla_to_skyfield(mission, station, sat_lla, 
                        force_download: bool=False):
    """
    This function projects, i.e. maps, a satellite's latitude, longitude, and altitude 
    (LLA) coordinates to the ASI's azimuth and elevation coordinates and index. 
    This function is useful to plot a satellite's location in the ASI image.

    Parameters
    ----------
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    sat_lla: np.ndarray or pd.DataFrame
        The satellite's latitude, longitude, and altitude coordinates in a 2d array
        with shape (nPosition, 3) where each row is the number of satellite positions
        to map, and the columns correspond to latitude, longitude, and altitude, 
        respectively. The altitude is in kilometer units.
    force_download: bool (optional)
        If True, download the calibration file even if it already exists.

    Returns
    -------
    sat_azel: np.ndarray
        An array with shape (nPosition, 2) of the satellite's azimuth and 
        elevation coordinates.
    sat_azel_index: np.ndarray
        An array with shape (nPosition, 2) of the x- and y-axis pixel 
        indices for the ASI image.  

    Example
    -------
    """

    # Check the sat_lla input parameter to make sure it is of the correct shape
    input_shape = sat_lla.shape
    # Reshape into a (1,3) array if the user provided only one set of satellite 
    # coordinates. 
    if len(input_shape) == 1:
        sat_lla = sat_lla.reshape(1, input_shape[0])
    assert sat_lla.shape[1] == 3, 'sat_lla must have 3 columns.'

    # Load the catalog
    cal_dict = load_cal_file(mission, station, force_download=force_download)

    # Set up the ground station object.
    earth = skyfield.api.load('de421.bsp')['earth']
    station = earth + skyfield.api.Topos(
                            latitude_degrees=cal_dict['SITE_MAP_LATITUDE'], 
                            longitude_degrees=cal_dict['SITE_MAP_LONGITUDE'], 
                            elevation_m=cal_dict['SITE_MAP_ALTITUDE'])
    ts = skyfield.api.load.timescale()
    t = ts.utc(datetime.now().year)  # This argument should not matter.

    sat_azel = np.nan*np.zeros((sat_lla.shape[0], 2))

    for i, (lat_i, lon_i, alt_km_i) in enumerate(sat_lla):
        # Check if lat, lon, or alt is nan or -1E31 
        # (the error value in the IRBEM library).
        any_nan = len(np.where(np.isnan([lat_i, lon_i, alt_km_i]))[0])
        any_neg = len(np.where([lat_i, lon_i, alt_km_i] == -1E31)[0])
        if any_nan or any_neg:
            continue
        sat_i = earth + skyfield.api.Topos(
                            latitude_degrees=lat_i, 
                            longitude_degrees=lon_i, 
                            elevation_m=1E3*alt_km_i
                            )
        astro = station.at(t).observe(sat_i)
        app = astro.apparent()
        el_i, az_i, _ = app.altaz()
        sat_azel[i, :] = az_i.degrees, el_i.degrees
    
    # If len(inuput_shape) == 1, a 1d array, flatten the (1x3) sat_azel and 
    # sat_azel_index arrays into a (3,) array. This way the input and output
    # lla arrays have the same number of dimentions.
    if len(input_shape) == 1:
        return sat_azel.flatten(), None
    else:
        return sat_azel, None

if __name__ == '__main__':
    # ATHA's LLA coordintaes are (54.72, -113.301, 676 (meters)).
    lla = np.array([54.72, -113.301, 500])
    azel, azel_index = lla_to_skyfield('THEMIS', 'ATHA', lla)

    lla_2 = np.array([[54.72, -113.301, 500]])
    azel, azel_index = lla_to_skyfield('THEMIS', 'ATHA', lla_2)


class Map_THEMIS_ASI:
    def __init__(self, site, time):
        """
        This class uses the THEMIS_ASI class to map the lat-lon-alt location
        of a satellite, ballon, etc. to AzEl coordinates using the THEMIS
        ASI calibration data and the Skyfield library. 

        Attributes
        ----------
        asi_dir : str
            The ASI directory
        site : str
            Shorthand ASI site name
        time : datetime.datetime
            The time used to lookup and load the ASI data.

        Methods
        -------
        map_satazel_to_asiazel()
            Maps from the satellite azimuth and elevation (azel) coordimates
            and finds the nearest all sky imager (ASI) azel calibration pixel. 
        map_lla_to_sat_azel()
            Maps the satellite lat, lon, and alt (km) coorinates to azel 
            coordinates for the ground station. The Skyfield library is used
            for this mapping.
        """
        super().__init__(site, time)
        return

    def map_lla_to_asiazel(self, lla):
        """
        Wrapper for map_satazel_to_asiazel() and map_lla_to_sat_azel().
        """
        self.sat_azel = self.map_lla_to_sat_azel(lla)
        self.asi_azel_index = self.map_satazel_to_asiazel(self.sat_azel)
        return self.asi_azel_index

    def map_satazel_to_asiazel(self, sat_azel, deg_thresh=0.1,
                            deg_thresh_scale_factor=2):
        """
        Given the azimuth and elevation of the satellite, sat_azel, locate 
        the THEMIS ASI calibration pixel index that is closest to the 
        satellite az and el. Note that the old scipy.spatial.KDTree() 
        implementation does not work because the calibration values are 
        in polar coorinates.

        deg_thresh is the starting degree threshold between az, el and the 
        calibration file. If no matches are found during the first pass,
        the deg_thresh is scaled by deg_thresh_scale_factor to expand the search
        to a wider range of calibration pixels. 

        Parameters
        ----------
        sat_azel : array
            A 1d or 2d array of satelite azimuth and elevation points.
            If 2d, the rows correspons to time.
        deg_thresh : float (optional)
            The degree threshold first used to find the ASI calibration pixel.
        deg_thresh_scale_factor : float (optional)
            If no ASI pixel is found using the deg_thresh, the deg_thresh
            is scaled by deg_thresh_scale_factor until a pixel value is found.

        Returns
        -------
        self.asi_azel_index : array
            An array with the same shape as sat_azel, but representing the
            indicies in the ASI calibration file (and image).
        """
        n_dims = len(sat_azel.shape)
        if n_dims == 2:
            self.asi_azel_index = np.nan*np.zeros(sat_azel.shape)
        elif n_dims == 1:
            self.asi_azel_index = np.nan*np.zeros((1, sat_azel.shape[0]))
            sat_azel = np.array([sat_azel])

        az_coords = self.cal['az'].copy().ravel()
        az_coords[np.isnan(az_coords)] = -10000
        el_coords = self.cal['el'].copy().ravel()
        el_coords[np.isnan(el_coords)] = -10000
        asi_azel_cal = np.stack((az_coords, el_coords), axis=-1)

        # Find the distance between the sattelite azel points and
        # the asi_azel points. dist_matrix[i,j] is the distance 
        # between ith asi_azel_cal value and jth sat_azel. 
        dist_matrix = scipy.spatial.distance.cdist(asi_azel_cal, sat_azel,
                                                metric='euclidean')
        # Now find the minimum distance for each sat_azel.
        idx_min_dist = np.argmin(dist_matrix, axis=0)
        # if idx_min_dist == 0, it is very likely to be a NaN value
        # beacause np.argmin returns 0 if a row has a NaN.
        # NaN's arise in the first place if there is an ASI image without
        # an AC6 data point nearby in time.
        idx_min_dist = np.array(idx_min_dist, dtype=object)
        idx_min_dist[idx_min_dist==0] = np.nan
        # For use the 1D index for the flattened ASI calibration
        # to get out the azimuth and elevation pixels.
        self.asi_azel_index[:, 0] = np.remainder(idx_min_dist, 
                                        self.cal['az'].shape[1])
        self.asi_azel_index[:, 1] = np.floor_divide(idx_min_dist, 
                                        self.cal['az'].shape[1])
        
        # Collapse the 2d asi_azel to 1d if the user specifed a
        # a 1d array argument.            
        if n_dims == 1:
            self.asi_azel_index = self.asi_azel_index[0, :]
        return self.asi_azel_index

    def map_lla_to_sat_azel(self, lla):
        """
        Get the satellite's azimuth and elevation given the satellite's
        lat, lon, and alt_km coordinates.

        Parameters
        ----------
        lla : 1d or 2d array of floats
            The lat, lon, and alt_km values of the satellite. If 2d, 
            the rows correspond to time.

        Returns
        -------
        sat_azel : array
            An array of shape lla.shape[0] x 2 with the satellite azimuth 
            and elevation columns.
        """
        planets = skyfield.api.load('de421.bsp')
        earth = planets['earth']
        station = earth + skyfield.api.Topos(latitude_degrees=self.cal['lat'], 
                                longitude_degrees=self.cal['lon'], 
                                elevation_m=self.cal['alt_m'])
        ts = skyfield.api.load.timescale()
        t = ts.utc(2020) #.now()

        # Check if the user passed in one set of LLA values or a 2d array. 
        # Save the number of dimensions and if is 1D, turn into a 2D array of
        # shape 1 x 3. 
        n_dims = len(lla.shape)
        if n_dims == 1:
            lla = np.array([lla])

        sat_azel = np.nan*np.zeros((lla.shape[0], 2))

        for i, (lat_i, lon_i, alt_km_i) in enumerate(lla):
            # Check if lat, lon, or alt is nan (happens when there is no 
            # corresponding AC6 data point close to an ASI image).
            if (
                any(np.isnan([lat_i, lon_i, alt_km_i])) or
                (np.min([lat_i, lon_i, alt_km_i]) == -1E31)
                ):
                continue
            sat_i = earth + skyfield.api.Topos(
                                latitude_degrees=lat_i, 
                                longitude_degrees=lon_i, 
                                elevation_m=1E3*alt_km_i
                                )
            astro = station.at(t).observe(sat_i)
            app = astro.apparent()
            el_i, az_i, _ = app.altaz()
            sat_azel[i, 1], sat_azel[i, 0] = el_i.degrees, az_i.degrees
        # Remove the extra dimension if user fed in one set of LLA values.
        if n_dims == 1:
            sat_azel = sat_azel[0, :]
        return sat_azel