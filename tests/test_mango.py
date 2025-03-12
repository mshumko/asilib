"""
Test the mango loaders and a few basic loading functions.
"""

from datetime import datetime

import requests
import pytest
import matplotlib.testing.decorators

from asilib.asi import mango

def test_mango_redline_time():
    """
    Get one timestamp from the redline imager.
    """
    time=datetime(2024, 9, 24, 3, 56, 0)
    location_code='CVO'
    asi = mango(location_code, 'redline', time=time)
    assert asi.data.time == time
    assert asi.data.image.shape == (500, 500)
    return

def test_mango_greenline_time():
    """
    Get one timestamp from the greenline imager.
    """
    time=datetime(2024, 10, 25, 4, 20, 0)
    location_code='BDR'
    asi = mango(location_code, 'greenline', time=time)
    assert asi.data.time == time
    assert asi.data.image.shape == (500, 500)
    return

def test_mango_redline_time_range():
    """
    Get multiple timestamps from the redline imager.
    """
    time_range=(datetime(2021, 11, 4, 1, 0), datetime(2021, 11, 4, 12, 24))
    location_code='CFS'
    asi = mango(location_code, 'redline', time_range=time_range)
    assert asi.data.time.shape[0] == 158
    assert asi.data.time[0] == datetime(2021, 11, 4, 1, 56)
    assert asi.data.time[-1] == time_range[-1]
    return

def test_mango_greenline_time_range():
    """
    Get multiple timestamps from the greenline imager.
    """
    time_range=(datetime(2024, 10, 25, 1, 50, 0), datetime(2024, 10, 25, 11, 22, 0))
    location_code='BDR'
    asi = mango(location_code, 'greenline', time_range=time_range)
    assert asi.data.time[0] == time_range[0]
    assert asi.data.time[-1] == time_range[-1]
    assert asi.data.image.shape == (287, 500, 500)
    return