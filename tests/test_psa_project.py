"""
Tests the psa_project() data loading and the example plotting functions.
"""
from datetime import datetime
import warnings

import matplotlib
import pytest
import matplotlib.testing.decorators
import requests
import urllib3

import asilib
import asilib.asi


def test_valid_camera_code():
    location_code = 'c2'
    try:
        asi = asilib.asi.psa_project(
            location_code, 
            time=datetime(2019, 3, 1, 18, 30, 0),
            load_images=False
        )
        assert asi.meta['location'] == 'C2'
    except (urllib3.exceptions.ConnectTimeoutError, requests.exceptions.ConnectTimeout) as err:
        warnings.warn(UserWarning(f"Connection timed out, skipping test. {err}"))
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

# @matplotlib.testing.decorators.image_comparison(
#     baseline_images=['test_plot_fisheye'], 
#     tol=asilib.config['_test_image_tol'], 
#     remove_text=True, 
#     extensions=['png'],
#     style='classic'
# )
# def test_plot_fisheye():
#     """
#     https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi?year=2017&month=03&day=07&jump=Plot
#     """
#     try:
#         asi = asilib.asi.psa_project(
#            'C1', 
#            time=datetime(2017, 3, 7, 19, 35, 0),
#            redownload=False
#            )
#         asi.plot_fisheye(origin=(0.9, 0.1))

#     except (urllib3.exceptions.ConnectTimeoutError, requests.exceptions.ConnectTimeout) as err:
#         warnings.warn(UserWarning(f"Connection timed out, skipping test. {err}"))
#     return