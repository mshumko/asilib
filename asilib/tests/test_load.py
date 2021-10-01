import re
import unittest
import pathlib
from datetime import datetime, time
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
        self.time_range = [datetime(2016, 10, 29, 4, 0), datetime(2016, 10, 29, 4, 1)]
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

    def test_load_image_themis(self, create_reference=False):
        """Load one THEMIS ASI image."""
        time, frame = load.load_image('THEMIS', 'GILL', self.load_date)

        reference_path = pathlib.Path(
            config['ASILIB_DIR'], 'tests', 'data', 'test_load_image_themis.npy'
        )
        if create_reference:
            np.save(reference_path, frame)
        frame_reference = np.load(reference_path)

        time_diff = (self.load_date - time).total_seconds()
        self.assertTrue(abs(time_diff) < 3)

        np.testing.assert_equal(frame_reference, frame)
        return

    def test_load_image_rego(self, create_reference=False):
        """Load one REGO ASI image."""
        time, frame = load.load_image('REGO', 'GILL', time=self.load_date)

        reference_path = pathlib.Path(
            config['ASILIB_DIR'], 'tests', 'data', 'test_load_image_rego.npy'
        )
        if create_reference:
            np.save(reference_path, frame)
        frame_reference = np.load(reference_path)

        time_diff = (self.load_date - time).total_seconds()
        self.assertTrue(abs(time_diff) < 3)

        np.testing.assert_equal(frame_reference, frame)
        return

    def test_load_images_themis(self, create_reference=False):
        """load one minute of THEMIS images."""
        times, frames = load.load_image('THEMIS', 'GILL', time_range=self.time_range)

        # np.save can't save an array of datetime objects without allow_pickle=True.
        # Since this can be a security concern, we'll save a string version of
        # datetimes.
        times = np.array([t.isoformat() for t in times])

        reference_path = pathlib.Path(
            config['ASILIB_DIR'], 'tests', 'data', 'test_load_images_themis.npz'
        )
        if create_reference:
            np.savez_compressed(reference_path, frames=frames, times=times)

        reference = np.load(reference_path)

        np.testing.assert_equal(reference['frames'], frames)
        np.testing.assert_equal(reference['times'], times)
        return

    def test_load_images_rego(self, create_reference=False):
        """Load one minute of REGO images."""
        times, frames = load.load_image('REGO', 'GILL', time_range=self.time_range)

        # np.save can't save an array of datetime objects without allow_pickle=True.
        # Since this can be a security concern, we'll save a string version of
        # datetimes.
        times = np.array([t.isoformat() for t in times])

        reference_path = pathlib.Path(
            config['ASILIB_DIR'], 'tests', 'data', 'test_load_images_rego.npz'
        )
        if create_reference:
            np.savez_compressed(reference_path, frames=frames, times=times)

        reference = np.load(reference_path)

        np.testing.assert_equal(reference['frames'], frames)
        np.testing.assert_equal(reference['times'], times)
        return


if __name__ == '__main__':
    unittest.main()
