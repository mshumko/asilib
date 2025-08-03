"""
Tests the psa_emccd() data loading and the example plotting functions.
"""
from datetime import datetime

import numpy
import pandas
import matplotlib
import pytest
import matplotlib.testing.decorators

import asilib.asi

def test_valid_camera_code():
    location_code = 'c2'
    asi = asilib.asi.psa_emccd(
        location_code, 
        time=datetime(2019, 3, 1, 18, 30, 0)
    )
    assert asi.meta['location'] == 'C2'
    return

def test_invalid_camera_code():
    location_code = 'C100'
    with pytest.raises(ValueError) as excinfo:
        asi = asilib.asi.psa_emccd(
            location_code, 
            time=datetime(2019, 3, 1, 18, 30, 0)
        )
    assert f"{location_code=} is invalid." in str(excinfo.value)
    return

def test_valid_name():
    location_name = 'Tromsoe'
    asi = asilib.asi.psa_emccd(
        location_name, 
        time=datetime(2019, 3, 1, 18, 30, 0)
    )
    assert asi.meta['location'] == 'C1'
    return

def test_invalid_name():
    location_name = 'Test'
    with pytest.raises(ValueError) as excinfo:
        asi = asilib.asi.psa_emccd(
            location_name, 
            time=datetime(2019, 3, 1, 18, 30, 0)
        )
    assert f"location_code='TEST' is invalid." in str(excinfo.value)
    return