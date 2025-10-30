"""
Test suite for the GPS class that handles Combined X-ray Dosimeter (CXD) data from LANL.
"""
import pytest
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from clowncar import GPS

def test_gps_download():
    """
    Test the download functionality of the GPS class.
    """
    time_range = (datetime(2007, 2, 10), datetime(2007, 2, 17))
    gps = GPS(time_range)

    # Download data for the specified time range
    assert list(gps.data.keys()) == ['ns41', 'ns53', 'ns54', 'ns56', 'ns58', 'ns59', 'ns60', 'ns61']

    assert np.all(gps.energies == np.array(
        [0.12, 0.21, 0.3, 0.425, 0.6, 0.8, 1.0, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        ))
    
    assert gps.l_keys == ['L_shell', 'L_LGM_TS04IGRF', 'L_LGM_OP77IGRF', 'L_LGM_T89CDIP', 'L_LGM_T89IGRF']
    
    # # Check that the timestamps are within the specified range
    # assert data.index.min() >= start_time, "Data start time is before requested start time."
    # assert data.index.max() <= end_time, "Data end time is after requested end time."
    return