"""
This script maps the AC6 data location, along Earth's magnetic field lines
(assumed IGRF + ext_model), to map_alt in km.

Parameters
__________
ext_model: string
    The external magnetic field model. The internal model is IGRF.
map_alt: float
    The AC6 mapping altitude in kilometers.
catalog_name: str
    The catalog (or data) name to load
catalog_dir: str
    The catalog directory to load the data from.
"""
import pathlib

import pandas as pd
import numpy as np
import IRBEM


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
    """

    mapped_footprint = np.nan * np.zeros((space_time.shape[0], 3))

    m = IRBEM.MagFields(kext=b_model)  # Initialize the IRBEM model.

    # Loop over every set of satellite coordinates.
    for i, (time, lat, lon, alt) in enumerate(space_time):
        X = {'datetime': time, 'x1': alt, 'x2': lat, 'x3': lon}
        m_output = m.find_foot_point(X, maginput, map_alt, hemisphere)
        mapped_footprint[i, :] = m_output['XFOOT']
    return mapped_footprint
