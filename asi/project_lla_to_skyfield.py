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
        any_nan = bool(len(
            np.where(np.isnan([lat_i, lon_i, alt_km_i]))[0]
            ))
        any_neg = bool(len(
            np.where([lat_i, lon_i, alt_km_i] == -1E31)[0]
            ))
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

    # Now find the corresponding x- and y-axis pixel indices.
    asi_azel_index = _map_azel_to_azel_index(sat_azel, cal_dict)
    
    # If len(inuput_shape) == 1, a 1d array, flatten the (1x3) sat_azel and 
    # sat_azel_index arrays into a (3,) array. This way the input and output
    # lla arrays have the same number of dimentions.
    if len(input_shape) == 1:
        return sat_azel.flatten(), asi_azel_index.flatten()
    else:
        return sat_azel, asi_azel_index

def _map_azel_to_azel_index(sat_azel, cal_dict, deg_thresh=1):
    """
    Given the 2d array of the satellite's azimuth and elevation, locate 
    the nearest ASI calibration x- and y-axis pixel indices. Note that the 
    scipy.spatial.KDTree() algorithm is efficient at finding nearest 
    neighbors. However it does not work in a polar coordinate system. 

    Parameters
    ----------
    sat_azel : array
        A 1d or 2d array of satelite azimuth and elevation points.
        If 2d, the rows correspons to time.
    cal_dict: dict
        The calibration file dictionary containing 
    deg_thresh : float (optional)
        The degree threshold first used to find the ASI calibration pixel.

    Returns
    -------
    asi_azel_index: np.ndarray
        An array with the same shape as sat_azel, but representing the
        x- and y-axis pixel indices in the ASI image.
    """
    az_coords = cal_dict['FULL_AZIMUTH'].ravel()
    az_coords[np.isnan(az_coords)] = -10000
    el_coords = cal_dict['FULL_ELEVATION'].ravel()
    el_coords[np.isnan(el_coords)] = -10000
    asi_azel_cal = np.stack((az_coords, el_coords), axis=-1)

    # Find the distance between the satellite azel points and
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
    asi_azel_index = np.nan*np.ones_like(sat_azel)
    asi_azel_index[:, 0] = np.remainder(idx_min_dist, 
                                    cal_dict['FULL_AZIMUTH'].shape[1])
    asi_azel_index[:, 1] = np.floor_divide(idx_min_dist, 
                                    cal_dict['FULL_AZIMUTH'].shape[1])
    return asi_azel_index

if __name__ == '__main__':
    # ATHA's LLA coordintaes are (54.72, -113.301, 676 (meters)).
    lla = np.array([54.72, -113.301, 500])
    azel, azel_index = lla_to_skyfield('THEMIS', 'ATHA', lla)

    lla_2 = np.array([[54.72, -113.301, 500], [54.72, -113.301, 500]])
    azel, azel_index = lla_to_skyfield('THEMIS', 'ATHA', lla_2)
    print(azel, azel_index)