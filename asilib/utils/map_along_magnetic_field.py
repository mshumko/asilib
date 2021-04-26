import pathlib
import warnings

import pandas as pd
import numpy as np

import asilib

try:
    import IRBEM
except ImportError:
    if asilib.config['IRBEM_WARNING']:
        warnings.warn(
            "The IRBEM magnetic field library is not installed and is "
            "a dependency of asilib.map_along_magnetic_field()."
        )


def map_along_magnetic_field(
    space_time: np.ndarray,
    map_alt: float,
    b_model: str = 'OPQ77',
    maginput: dict = None,
    hemisphere: int = 0,
) -> np.ndarray:
    """
    This function uses the IRBEM-Lib library to map the spacecraft's position:
    latitude, longitude, altitude with the spacecraft's time to the map_alt
    altitude in units of km, in the same hemisphere.

    Parameters
    ----------
    space_time: np.ndarray
        A 2d array with shape (nPositions, 4) with the columns containing the
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
    """

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
