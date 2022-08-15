from typing import Tuple
import importlib

import numpy as np
import pymap3d
import scipy.spatial

try:
    import IRBEM
except ImportError:
    pass  # make sure that asilb.__init__ fully loads and crashes if the user calls asilib.lla2footprint().

import asilib
from asilib.io import utils


def lla2azel(
    asi_array_code: str, location_code: str, time: utils._time_type, sat_lla: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps, a satellite's latitude, longitude, and altitude (LLA) coordinates
    to the ASI's azimuth and elevation (azel) coordinates and image pixel index.

    This function is useful to plot a satellite's location in the ASI image using the
    pixel indices.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to find the relevant skymap file. If str, ``time``
        must be in the ISO 8601 standard.
    sat_lla: np.ndarray or pd.DataFrame
        The satellite's latitude, longitude, and altitude coordinates in a 2d array
        with shape (nPosition, 3) where each row is the number of satellite positions
        to map, and the columns correspond to latitude, longitude, and altitude,
        respectively. The altitude is in kilometer units.

    Returns
    -------
    sat_azel: np.ndarray
        An array with shape (nPosition, 2) of the satellite's azimuth and
        elevation coordinates.
    asi_pixels: np.ndarray
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

    # Check the sat_lla input parameter to make sure it is of the correct shape
    input_shape = sat_lla.shape
    # Reshape into a (1,3) array if the user provided only one set of satellite
    # coordinates.
    if len(input_shape) == 1:
        sat_lla = sat_lla.reshape(1, input_shape[0])
    assert sat_lla.shape[1] == 3, 'sat_lla must have 3 columns.'

    # Load the catalog
    skymap_dict = asilib.io.load.load_skymap(asi_array_code, location_code, time)

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
            skymap_dict['SITE_MAP_LATITUDE'],
            skymap_dict['SITE_MAP_LONGITUDE'],
            skymap_dict['SITE_MAP_ALTITUDE'],
        )
        sat_azel[i, :] = [az, el]

    # Now find the corresponding x- and y-axis pixel indices.
    asi_pixels = _map_azel_to_pixel(sat_azel, skymap_dict)

    # If len(input_shape) == 1, a 1d array, flatten the (1x3) sat_azel and
    # asi_pizels arrays into a (3,) array. This way the input and output
    # lla arrays have the same number of dimentions.
    if len(input_shape) == 1:
        return sat_azel.flatten(), asi_pixels.flatten()
    else:
        return sat_azel, asi_pixels


def lla2footprint(
    space_time: np.ndarray,
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
    space_time: np.ndarray
        A 2d array with shape (n_times, 4) with the columns containing the
        time, latitude, longitude, and altitude coordinates in that order.
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

    magnetic_footprint = np.nan * np.zeros((space_time.shape[0], 3))

    m = IRBEM.MagFields(kext=b_model)  # Initialize the IRBEM model.

    # Loop over every set of satellite coordinates.
    for i, (time, lat, lon, alt) in enumerate(space_time):
        X = {'datetime': time, 'x1': alt, 'x2': lat, 'x3': lon}
        m_output = m.find_foot_point(X, maginput, map_alt, hemisphere)
        magnetic_footprint[i, :] = m_output['XFOOT']
    # Map from IRBEM's (alt, lat, lon) -> (lat, lon, alt)
    magnetic_footprint[:, [2, 0, 1]] = magnetic_footprint[:, [0, 1, 2]]
    return magnetic_footprint


def _map_azel_to_pixel(sat_azel: np.ndarray, skymap_dict: dict) -> np.ndarray:
    """
    Given the 2d array of the satellite's azimuth and elevation, locate
    the nearest ASI skymap x- and y-axis pixel indices. Note that the
    scipy.spatial.KDTree() algorithm is efficient at finding nearest
    neighbors. However it does not work in a polar coordinate system.

    Parameters
    ----------
    sat_azel : array
        A 1d or 2d array of satelite azimuth and elevation points.
        If 2d, the rows correspons to time.
    skymap_dict: dict
        The skymap file dictionary

    Returns
    -------
    pixel_index: np.ndarray
        An array with the same shape as sat_azel, but representing the
        x- and y-axis pixel indices in the ASI image.
    """
    az_coords = skymap_dict['FULL_AZIMUTH'].ravel()
    az_coords[np.isnan(az_coords)] = -10000
    el_coords = skymap_dict['FULL_ELEVATION'].ravel()
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
    # For use the 1D index for the flattened ASI skymap
    # to get out the azimuth and elevation pixels.
    pixel_index = np.nan * np.ones_like(sat_azel)
    pixel_index[:, 0] = np.remainder(idx_min_dist, skymap_dict['FULL_AZIMUTH'].shape[1])
    pixel_index[:, 1] = np.floor_divide(idx_min_dist, skymap_dict['FULL_AZIMUTH'].shape[1])
    return pixel_index
