from typing import Sequence, Tuple

import numpy as np
import pymap3d
import scipy.spatial

from asilib.load import load_cal_file


def lla_to_skyfield(
    mission, station, sat_lla, force_download: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function projects, i.e. maps, a satellite's latitude, longitude, and altitude
    (LLA) coordinates to the ASI's azimuth and elevation coordinates and pixel index.
    This function is useful to plot a satellite's location in the ASI image using the
    pixel indices.

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
    asi_pixels: np.ndarray
        An array with shape (nPosition, 2) of the x- and y-axis pixel
        indices for the ASI image.

    Example
    -------
    import numpy as np

    from asilib import lla_to_skyfield

    # THEMIS/ATHA's LLA coordinates are (54.72, -113.301, 676 (meters)).
    # The LLA is a North-South pass right above ATHA..
    n = 50
    lats = np.linspace(60, 50, n)
    lons = -113.64*np.ones(n)
    alts = 500**np.ones(n)
    lla = np.array([lats, lons, alts]).T

    azel, pixels = lla_to_skyfield('REGO', 'ATHA', lla)
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

    sat_azel = np.nan * np.zeros((sat_lla.shape[0], 2))

    # Loop over every set of LLA coordinates and use pymap3d.geodetic2aer
    # to map to the azimuth and elevation.
    for i, (lat_i, lon_i, alt_km_i) in enumerate(sat_lla):
        # Check if lat, lon, or alt is nan or -1E31
        # (the error value in the IRBEM library).
        any_nan = bool(len(np.where(np.isnan([lat_i, lon_i, alt_km_i]))[0]))
        any_neg = bool(len(np.where(np.array([lat_i, lon_i, alt_km_i]) == -1e31)[0]))
        if any_nan or any_neg:
            continue

        az, el, _ = pymap3d.geodetic2aer(
            lat_i,
            lon_i,
            1e3 * alt_km_i,
            cal_dict['SITE_MAP_LATITUDE'],
            cal_dict['SITE_MAP_LONGITUDE'],
            cal_dict['SITE_MAP_ALTITUDE'],
        )
        sat_azel[i, :] = [az, el]

    # Now find the corresponding x- and y-axis pixel indices.
    asi_pixels = _map_azel_to_pixel(sat_azel, cal_dict)

    # If len(input_shape) == 1, a 1d array, flatten the (1x3) sat_azel and
    # asi_pizels arrays into a (3,) array. This way the input and output
    # lla arrays have the same number of dimentions.
    if len(input_shape) == 1:
        return sat_azel.flatten(), asi_pixels.flatten()
    else:
        return sat_azel, asi_pixels


def _map_azel_to_pixel(sat_azel: np.ndarray, cal_dict: dict) -> np.ndarray:
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
        The calibration file dictionary

    Returns
    -------
    pixel_index: np.ndarray
        An array with the same shape as sat_azel, but representing the
        x- and y-axis pixel indices in the ASI image.
    """
    az_coords = cal_dict['FULL_AZIMUTH'][::-1, ::-1].ravel()
    az_coords[np.isnan(az_coords)] = -10000
    el_coords = cal_dict['FULL_ELEVATION'][::-1, ::-1].ravel()
    el_coords[np.isnan(el_coords)] = -10000
    asi_azel_cal = np.stack((az_coords, el_coords), axis=-1)

    # Find the distance between the satellite azel points and
    # the asi_azel points. dist_matrix[i,j] is the distance
    # between ith asi_azel_cal value and jth sat_azel.
    dist_matrix = scipy.spatial.distance.cdist(asi_azel_cal, sat_azel, metric='euclidean')
    # Now find the minimum distance for each sat_azel.
    idx_min_dist = np.argmin(dist_matrix, axis=0)
    # if idx_min_dist == 0, it is very likely to be a NaN value
    # beacause np.argmin returns 0 if a row has a NaN.
    # NaN's arise in the first place if there is an ASI image without
    # an AC6 data point nearby in time.
    idx_min_dist = np.array(idx_min_dist, dtype=object)
    idx_min_dist[idx_min_dist == 0] = np.nan
    # For use the 1D index for the flattened ASI calibration
    # to get out the azimuth and elevation pixels.
    pixel_index = np.nan * np.ones_like(sat_azel)
    pixel_index[:, 0] = np.remainder(idx_min_dist, cal_dict['FULL_AZIMUTH'].shape[1])
    pixel_index[:, 1] = np.floor_divide(idx_min_dist, cal_dict['FULL_AZIMUTH'].shape[1])
    return pixel_index
