from typing import Tuple

import numpy as np

import pymap3d


def geodetic_skymap(
        imager_lla:Tuple[float],
        az_skymap:np.ndarray, 
        el_skymap:np.ndarray, 
        alt:float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project each pixel's (az, el) coordinates to geodetic (lat, lon) coordinates at an emission
    altitude.

    Parameters
    ----------
    imager_lla:tuple
        The imager's latitude, longitude, and altitude in km.
    az_skymap:np.ndarray 
        A 2d array of pixel azimuths. 
    el_skymap:np.ndarray 
        A 2d array of pixel elevations.
    alt:float
        Auroral emission altitude in km to map to.

    Returns
    -------
    np.ndarray
        The latitude skymap.
    np.ndarray
        The longitude skymap.
    """
    lat_skymap = np.zeros_like(az_skymap)
    lon_skymap = np.zeros_like(az_skymap)

    for row in range(az_skymap.shape[0]):
        for col in range(az_skymap.shape[1]):
            # Based on Michael Hirsh's (scivision) dascasi package.
            lat_skymap[row, col], lon_skymap[row, col], _ = pymap3d.aer2geodetic(
                az_skymap[row, col],
                el_skymap[row, col],
                alt * 1e3 / np.sin(np.radians(el_skymap[row, col])),
                *imager_lla
            )
    return lat_skymap, lon_skymap