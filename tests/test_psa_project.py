"""
Tests the psa_project() data loading and the example plotting functions.
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
    asi = asilib.asi.psa_project(
        location_code, 
        time=datetime(2019, 3, 1, 18, 30, 0),
        load_images=False
    )
    assert asi.meta['location'] == 'C2'
    return

def test_invalid_camera_code():
    location_code = 'C100'
    with pytest.raises(ValueError) as excinfo:
        asi = asilib.asi.psa_project(
            location_code, 
            time=datetime(2019, 3, 1, 18, 30, 0)
        )
    assert f"{location_code=} is invalid." in str(excinfo.value)
    return

@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_plot_fisheye'], 
    tol=10, 
    remove_text=True, 
    extensions=['png']
)
def test_plot_fisheye():
    """
    https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi?year=2017&month=03&day=07&jump=Plot
    """
    asi = asilib.asi.psa_project(
       'C1', 
       time=datetime(2017, 3, 7, 19, 35, 0),
       redownload=False
       )
    asi.plot_fisheye(origin=(0.9, 0.1))
    return