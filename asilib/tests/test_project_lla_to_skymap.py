import unittest
import pathlib
from datetime import datetime

import numpy as np

from asi.project_lla_to_skyfield import lla_to_skyfield

"""
Tests that the LLA to AzEl projecting (mapping) returns correct outputs
for the Athabasca (ATHA) THEMIS and REGO imagers. The LLA is assumed 
overhead and at 500 km altitude.
"""

class TestProjectLLAtoSkyfield(unittest.TestCase):
    def test_themis_lla_to_azel(self):
        """
        Tests that the input LLA corresponds to right overhead of the THEMIS ATHA
        station (elevation ~ 90 degrees) and (az_index, el_index) ~= (127, 127).
        """
        lla = np.array([54.72, -113.301, 500])
        azel, pixel_index = lla_to_skyfield('THEMIS', 'ATHA', lla)

        # Test the El values
        self.assertEqual(round(azel[1]), 90)

        # Test the AzEl indices
        self.assertEqual(pixel_index[0], 126)
        self.assertEqual(pixel_index[1], 127)
        return

    def test_rego_lla_to_azel(self):
        """
        Tests that the input LLA corresponds to right overhead of the ATHA REGO station 
        (elevation ~ 90 degrees) and (az_index, el_index) ~= (127, 127).
        """
        lla = np.array([54.60, -113.64, 500])
        azel, pixel_index = lla_to_skyfield('REGO', 'ATHA', lla)

        # Test the AzEl values
        self.assertEqual(round(azel[0]), 137)
        self.assertEqual(round(azel[1]), 90)

        # Test that the AzEl indices are within 20 pixels of zenith.
        self.assertTrue(abs(pixel_index[0] - 256) < 20)
        self.assertTrue(abs(pixel_index[0] - 256) < 20)
        return

    def test_rego_lla_to_azel_track(self):
        """
        Tests that the input LLA track corresponds to a north-south 
        path other REGO/ATHA.
        """
        n = 50
        lats = np.linspace(60, 50, n)
        lons = -113.64*np.ones(n)
        alts = 500**np.ones(n)
        lla = np.array([lats, lons, alts]).T

        azel, pixel_index = lla_to_skyfield('REGO', 'ATHA', lla)
        return


if __name__ == '__main__':
    unittest.main()