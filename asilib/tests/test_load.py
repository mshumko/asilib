import re
import unittest
import pathlib
from datetime import datetime
import numpy as np
import cdflib

from asilib.io import load
from asilib import config

"""
Most of these tests just makes sure that the functions run correctly
and it doesn't validate the output.
"""


class TestPlotFrame(unittest.TestCase):
    def setUp(self):
        self.load_date = datetime(2016, 10, 29, 4)
        self.time_range = [datetime(2016, 10, 29, 4, 0), 
                           datetime(2016, 10, 29, 4, 10)]
        self.station = 'GILL'

    def test_rego_find_img(self):
        """Checks that the REGO ASI image file can be loaded."""
        cdf_path = load._find_img_path(self.load_date, 'REGO', self.station)
        assert cdf_path.name == 'clg_l1_rgf_gill_2016102904_v01.cdf'
        return

    def test_themis_find_img(self):
        """Checks that the REGO ASI image file can be loaded."""
        cdf_path = load._find_img_path(self.load_date, 'THEMIS', self.station)
        assert cdf_path.name == 'thg_l1_asf_gill_2016102904_v01.cdf'
        return

    def test_rego_load_skymap(self):
        """Load the REGO skymap file."""
        skymap = load.load_skymap('REGO', self.station, self.load_date)
        assert skymap['skymap_path'].name == 'rego_skymap_gill_20160129_vXX.sav'
        return

    def test_themis_load_skymap(self):
        """Load the THEMIS skymap file."""
        skymap = load.load_skymap('THEMIS', self.station, self.load_date)
        assert skymap['skymap_path'].name == 'themis_skymap_gill_20151121_vXX.sav'
        return

    def test_themis_get_frame(self, create_reference=False):
        """Get one THEMIS ASI image."""
        time, frame = load.get_frame(self.load_date, 'THEMIS', 'GILL')

        reference_path = pathlib.Path(config['ASILIB_DIR'], 'tests', 'data', 'test_themis_get_frame.npy')
        if create_reference:
            np.save(reference_path, frame)
        frame_reference = np.load(reference_path)

        time_diff = (self.load_date - time).total_seconds()
        self.assertTrue(abs(time_diff) < 3)

        np.testing.assert_equal(frame_reference, frame)
        return

    def test_rego_get_frame(self, create_reference=False):
        """Get one REGO ASI image."""
        time, frame = load.get_frame(self.load_date, 'REGO', 'GILL')

        reference_path = pathlib.Path(config['ASILIB_DIR'], 'tests', 'data', 'test_rego_get_frame.npy')
        if create_reference:
            np.save(reference_path, frame)
        frame_reference = np.load(reference_path)

        time_diff = (self.load_date - time).total_seconds()
        self.assertTrue(abs(time_diff) < 3)

        np.testing.assert_equal(frame_reference, frame)
        return

    def test_themis_get_frames(self, create_reference=False):
        """Get 10 minutes of THEMIS images."""
        times, frames = load.get_frames(self.time_range, 'THEMIS', 'GILL')

        # np.save can't save an array of datetime objects without allow_pickle=True. 
        # Since this can be a security concern, we'll save a string version of
        # datetimes.
        times = np.array([t.isoformat() for t in times])

        frames_reference_path = pathlib.Path(config['ASILIB_DIR'], 'tests', 'data', 
            'test_themis_get_frames_frames.npy')
        times_reference_path = pathlib.Path(config['ASILIB_DIR'], 'tests', 'data', 
            'test_themis_get_frames_times.npy')
        if create_reference:
            np.save(frames_reference_path, frames)
            np.save(times_reference_path, times)

        frame_reference = np.load(frames_reference_path)
        times_reference = np.load(times_reference_path)

        np.testing.assert_equal(frame_reference, frames)
        np.testing.assert_equal(times_reference, times)
        return

    def test_rego_get_frames(self, create_reference=True):
        """Get 10 minutes of REGO images."""
        times, frames = load.get_frames(self.time_range, 'REGO', 'GILL')

        # np.save can't save an array of datetime objects without allow_pickle=True. 
        # Since this can be a security concern, we'll save a string version of
        # datetimes.
        times = np.array([t.isoformat() for t in times])

        reference_path = pathlib.Path(config['ASILIB_DIR'], 'tests', 'data', 
            'test_rego_get_frames.npz')

        if create_reference:
            np.savez(reference_path, frames=frames, times=times)

        reference = np.load(reference_path)

        np.testing.assert_equal(reference['frames'], frames)
        np.testing.assert_equal(reference['times'], times)
        return


if __name__ == '__main__':
    unittest.main()
