"""
Tests asilib/analysis.equal_area.py
"""

import unittest
import pathlib

import numpy as np

from asilib.io.load import load_cal
from asilib.analysis.equal_area import equal_area, _dlon, _dlat

# Number of km in a degree of latitude. Also a degree of longitude at the equator
deg_distance_km = 111.321

class Test_keogram(unittest.TestCase):
    def test_dlat(self):
        """ 
        Tests the _dlat() helper function. Given one altitude, these should be the same.
        https://www.thoughtco.com/degree-of-latitude-and-longitude-distance-4070616
        """
        # Check that one degree of latitude is ~111 km at sea-level 
        assert np.isclose(_dlat(deg_distance_km, 0), 1, rtol=0.005)  # rtol=0.005 -> 0.5 km accuracy.
        # Same, but for array inputs
        assert np.all(np.isclose(_dlat(3*[deg_distance_km], [0, 0, 0]), [1,1,1], rtol=0.005))
        assert np.all(np.isclose(_dlat(deg_distance_km*np.ones(3), np.array([0, 0, 0])), 
                                [1,1,1], rtol=0.005))
        return

    def test_dlon(self):
        """ 
        Tests the _dlon() helper function 
        https://www.thoughtco.com/degree-of-latitude-and-longitude-distance-4070616
        """
        # A degree of longitude at the equator is 111.321 km.
        assert np.isclose(_dlon(deg_distance_km, 0, 0), 1, rtol=0.005)
        assert np.all(np.isclose(_dlon(3*[deg_distance_km], 3*[0], 3*[0]), np.ones(1), rtol=0.005))
        assert np.all(np.isclose(_dlon(deg_distance_km*np.ones(3), np.zeros(3), np.zeros(3)), 
                                np.ones(1), rtol=0.005))
            
        # Test at lat = +/-40 degrees
        assert np.isclose(_dlon(85, 0, 40), 1, rtol=0.005)
        assert np.isclose(_dlon(85, 0, -40), 1, rtol=0.005)
        return

    def test_equal_area(self):
        mission='THEMIS'
        station='RANK'
        box_km = (10, 10)  # in (Lat, Lon) directions.
        cal_dict = load_cal(mission, station)

        # Set up a north-south satellite track oriented to the east of the THEMIS/RANK 
        # station.
        n = 10
        lats = np.linspace(cal_dict["SITE_MAP_LATITUDE"] + 10, cal_dict["SITE_MAP_LATITUDE"] - 10, n)
        lons = (cal_dict["SITE_MAP_LONGITUDE"] + 3) * np.ones(n)
        alts = 500 * np.ones(n)
        lla = np.array([lats, lons, alts]).T

        
        return