from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

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

    for row in range(az_skymap.shape[0]):
        for col in range(az_skymap.shape[1]):
            # _range = np.sqrt((Re+alt)**2+(Re+imager_lla[-1])**2)
            # Based on Michael Hirsh's (scivision) dascasi package.
            lat_skymap[row, col], lon_skymap[row, col], _alts[row, col] = pymap3d.aer2geodetic(
                az_skymap[row, col],
                el_skymap[row, col],
                alt * 1e3 / np.sin(np.radians(el_skymap[row, col])),
                *imager_lla[:2], 
                1E3*imager_lla[-1],
                deg=True
            )
            pass
    plt.pcolormesh(_alts/1E3, vmin=0, vmax=200, cmap='seismic'); 
    plt.colorbar()
    plt.show()
    return lat_skymap, lon_skymap