from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

def footprint(center_lon:float, alt:float=110, obit_period:float=95, 
    precession_rate:float=10, cadence:float=0.1, 
    time_range:tuple=(datetime(2015,1,1), datetime(2015,1,2))
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
    time_range:tuple
        The start and end times that bound the footprint path.

    Returns
    -------
    np.array
        Footprint timestamps in datetime.datetime format.
    np.array
        An array of floats shaped (n_times, 3) with the footprint's
        lat, lon, alt coordinates.
    """
    n_minutes = (time_range[1]-time_range[0]).total_seconds()/60
    n_time_stamps = int(n_minutes/cadence)
    times = np.array([time_range[0] + timedelta(minutes=i*cadence) for i in range(n_time_stamps)])

    return times

if __name__ == '__main__':
    times, lla = footprint(-100)
    pass