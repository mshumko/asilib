import numpy as np
import pandas as pd

import asilib

earth_radius_km = 6371 # Earth radius

def equal_area(mission, station, lla, box_km=(5, 5), alt_thresh_km=3):
    """
    Given a square are in kilometers and a series of (latitude, 
    longitude, altitude) coordinates, calculate the pixel box
    width and height.

    Parameters
    ----------
    mission: str
        The mission used to look up the calibration file.
    station: str
        The station used to look up the calibration file.
    lla: np.ndarray
        An array with (n_time, 3) dimensions with the columns
        representing the latitude, longitude, and altitude 
        coordinates.
    box_size_km: iterable
        A length 2 iterable with the box dimensions in
        longitude and latitude in units of kilometers.

    Returns
    -------
    pixel_mask: np.ndarray
        An array with (n_time, 2) dimensions with the columns
        represneting the latitude and longitude coordinates, in 
        that order.
    """
    assert len(box_km) == 2, 'The box_km parameter must have a length of 2.'

    cal_dict = asilib.load_cal_file(mission, station)

    # Get numpy array if pd.DataFrame passed
    if isinstance(lla, pd.DataFrame):
        lla = lla.to_numpy()

    initial_shape = lla.shape

    # Check that the altitudes are in the appropriate range
    if len(initial_shape) == 1:  # 1d array
        assert np.min(np.abs(cal_dict['FULL_MAP_ALTITUDE']/1000-lla[-1])) < alt_thresh_km , (
            f'Got {lla[-1]} km altitude, but it must be one of these: {cal_dict["FULL_MAP_ALTITUDE"]/1000}')
        alt_index = np.argmin(np.abs(cal_dict['FULL_MAP_ALTITUDE']/1000-lla[-1]))
        lla = np.array([lla])
    elif len(initial_shape) == 2:  # 2d array
        for alt in lla[:, -1]:
            assert np.min(np.abs(cal_dict['FULL_MAP_ALTITUDE']/1000-alt)) < alt_thresh_km , (
                f'Got {alt} km altitude, but it must be one of these: {cal_dict["FULL_MAP_ALTITUDE"]/1000}')
        alt_index = np.argmin(np.abs(cal_dict['FULL_MAP_ALTITUDE']/1000-alt))

    pixel_mask = np.zeros((lla.shape[0], *cal_dict['FULL_MAP_LATITUDE'].shape[1:]))

    dlat = _dlat(box_km[1], lla[:, -1])
    dlon = _dlon(box_km[0], lla[:, -1], lla[:, 0])

    lat_map = cal_dict['FULL_MAP_LATITUDE'][alt_index, :, :]
    lon_map = cal_dict['FULL_MAP_LONGITUDE'][alt_index, :, :]

    for i, (lat, lon, _)in enumerate(lla):
        idx_box = np.where(
            (lat_map >= lat-dlat[i]/2) &
            (lat_map <= lat+dlat[i]/2) &
            (lon_map >= lon-dlon[i]/2) &
            (lon_map <= lon+dlon[i]/2)
        )
        pixel_mask[i, idx_box] = 1

    if len(initial_shape) == 1:
        return pixel_mask.reshape(cal_dict['FULL_MAP_LATITUDE'].shape[1:])
    else:
        return pixel_mask


def _dlat(d, alt):
    """
    Calculate the change in latitude that correpsponds to arc length distance d at 
    alt altitude. Input units are in kilometers.

    Parameters
    ----------
    d: float or np.ndarray
        A float or an array of arc length.
    alt: float or np.ndarray
        A float or an array of satellite altitudes.
    
    Returns
    -------
    dlat: float or np.ndarray
        A float or an array of the corresponding latitude differences in degrees.
    """
    return np.rad2deg(np.divide(d, (earth_radius_km+alt)))


def _dlon(d, alt, lat):
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

    numerator = np.sin(d/(2*(earth_radius_km+alt)))
    denominator = np.cos(np.deg2rad(lat))
    dlon_rads = 2*np.arcsin(numerator/denominator)
    return np.rad2deg(dlon_rads)