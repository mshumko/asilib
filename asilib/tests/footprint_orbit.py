from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

def footprint(center_lon:float, alt:float=110, obit_period:float=95, 
    precession_rate:float=10, cadence:float=0.1
    )->Tuple[np.array, np.array]:
    """
    Get the time stamps, and lat, lon, alt (LLA) coordinates.

    Parameters
    ----------
    center_lon:float
        The longitude at the midpoint of the time range.
    alt:float
        The footprint altitude in kilometers.
    obit_period:float
        The orbit period in minutes.
    precession_rate:float
        Controls how much the longitude change in one orbit.
    cadence:float
        The cadence of the footprint timestamps in minutes.

    Returns
    -------
    np.array
        Footprint timestamps starting on 2010-01-01.
    np.array
        An array of shape (n_times, 3) with the footprint's
        lat, lon, alt coordinates.
    """