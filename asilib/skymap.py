from typing import Tuple

import numpy as np

import pymap3d


Re = 6378.14  # km

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
    _alts = np.zeros_like(az_skymap)

    el_skymap[el_skymap <= 0] = np.nan

    # Range from observer to target assuming Earth is spherical, i.e. not an ellipsoid.
    # https://github.com/space-physics/dascasi/blob/4d72aa91e471a495566044c3fc387344dd12461f/src/dascasi/io.py#L107C32-L107C32
    _el_rad = np.deg2rad(el_skymap)
    _range = np.sqrt(
        (Re+alt)**2 +
        (Re+imager_lla[-1])**2 - 
        2*(Re+alt)*(Re+imager_lla[-1])*np.sin(
            _el_rad + np.arcsin((Re+imager_lla[-1])/(Re+alt)*np.cos(_el_rad))
            )
        )

    for row in range(az_skymap.shape[0]):
        for col in range(az_skymap.shape[1]):
            # Based on Michael Hirsh's (scivision) dascasi package.
            lat_skymap[row, col], lon_skymap[row, col], _alts[row, col] = pymap3d.aer2geodetic(
                az_skymap[row, col],
                el_skymap[row, col],
                _range[row, col] * 1e3,
                *imager_lla[:2], 
                1E3*imager_lla[-1],
                deg=True,
            )
    return lat_skymap, lon_skymap