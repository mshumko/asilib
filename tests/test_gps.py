"""
Test suite for the GPS class that handles Combined X-ray Dosimeter (CXD) data from LANL.
"""
import pytest
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from clowncar import GPS

def test_download():
    """
    Test the download functionality of the GPS class.
    """
    time_range = (datetime(2007, 2, 10), datetime(2007, 2, 17))
    gps = GPS(time_range, redownload=True)

    # Download data for the specified time range
    assert list(gps.data.keys()) == ['ns41', 'ns53', 'ns54', 'ns56', 'ns58', 'ns59', 'ns60', 'ns61']

    assert np.all(gps.energies == np.array(
        [0.12, 0.21, 0.3, 0.425, 0.6, 0.8, 1.0, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        ))
    
    assert gps.l_keys == ['L_shell', 'L_LGM_TS04IGRF', 'L_LGM_OP77IGRF', 'L_LGM_T89CDIP', 'L_LGM_T89IGRF']
    return

def test_avg_flux():
    """
    Test the average flux calculation.
    """
    time_range = (datetime(2007, 2, 10), datetime(2007, 2, 17))
    gps = GPS(time_range, redownload=False)
    _flux = gps.avg_flux(dt_min=3*60, min_samples=5)

    assert str(_flux.index[0]) == '2007-02-10 00:00:00'
    assert str(_flux.index[-1]) == '2007-02-16 21:00:00'
    assert _flux.shape == (56, 15)
    print(_flux.loc[:, 0.12].values.round())
    assert np.all(np.isclose(
        _flux.loc[:, .12].values.round(),
        np.array([
            2479977.,  2051500.,  1842723.,  1590309.,  1471912.,  1242690.,   736164.,
            522411. ,  476537.,   410026.,   390929.,   359535.,   367981.,   333073.,
            303915. ,  297472.,   289132.,   282117.,   161626.,   399965.,   442245.,
            244945. ,  538795.,   557499.,   924367.,   815934.,  1318079.,  2118577.,
            2144812.,  1898019.,  1613639., 26352712., 34890809., 15542616., 10912266.,
            8160543.,  7437673.,  6525212., 11878927., 21733047., 18555957., 13369918.,
            9904678., 10776063.,  8122306.,  7218946.,  6448182.,  6645735.,  6191205.,
            5687341.,  4012664.,  5103959.,  5301199.,  4821802.,  4321616.,  4223188.,
            ])
        ))
    return