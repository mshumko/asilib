import unittest
import pathlib
from datetime import datetime

import numpy as np

from asilib.project_lla_to_skyfield import lla_to_skyfield

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

        ATHA is at lon=-113, and the satellite track is lon=-112
        (the pass is to the east). The azimuth values should go
        0 -> 90 -> 180 degrees and not 0 -> 270 -> 180 degrees
        """
        n = 10
        lats = np.linspace(60, 50, n)
        lons = -112.64*np.ones(n)
        alts = 500*np.ones(n)
        lla = np.array([lats, lons, alts]).T

        sat_azel, asi_pixels = lla_to_skyfield('REGO', 'ATHA', lla)

        reference_sat_azel = np.array(
            [[  5.32950314,  35.80304471],
            [  6.91647509,  42.82120867],
            [  9.58464224,  51.65418672],
            [ 14.9730986 ,  62.57537388],
            [ 30.74319057,  74.99556918],
            [104.74451595,  81.72800427],
            [154.58750264,  71.34691226],
            [165.44993506,  59.1799343 ],
            [169.72328434,  48.8782155 ],
            [171.98265159,  40.62183087]]
            )

        reference_asi_pixels = np.array(
            [[235., 116.],
            [233., 134.],
            [231., 158.],
            [229., 188.],
            [227., 223.],
            [226., 262.],
            [225., 300.],
            [224., 335.],
            [225., 363.],
            [225., 385.]]
            )
        np.testing.assert_array_almost_equal(sat_azel, reference_sat_azel)
        np.testing.assert_array_almost_equal(asi_pixels, reference_asi_pixels)
        return


if __name__ == '__main__':
    unittest.main()