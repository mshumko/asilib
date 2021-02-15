import unittest
import requests
import pathlib
from datetime import datetime

from asilib.download import download_themis
from asilib import config

"""
Unit tests to check that the functions in download_themis.py are working correctly.
Run with "python3 test_download_themis.py -v" for the verbose output.
"""


class TestDownloadThemis(unittest.TestCase):
    def setUp(self):
        """ Set up a few variables. """
        self.day = datetime(2016, 10, 29, 4)
        self.station = 'Gill'
        self.url = (
            download_themis.IMG_BASE_URL
            + f'{self.station.lower()}/{self.day.year}/{str(self.day.month).zfill(2)}/'
        )
        return

    def test_server_response(self):
        """ Check that the server responds without an error. """
        r = requests.get(self.url)
        status_code = r.status_code
        # Check that the server status code is not
        # between 400-599 (error).
        self.assertNotEqual(status_code // 100, 4)
        self.assertNotEqual(status_code // 100, 5)
        return

    def test_download_img(self):
        """
        Test the full THEMIS data downloader and download an hour file
        clg_l1_rgf_luck_2020080104_v01.cdf to ./themis/.
        """
        temp_image_dir = pathlib.Path(config.ASI_DATA_DIR, 'themis')
        temp_image_path = temp_image_dir / 'thg_l1_asf_gill_2016102904_v01.cdf'

        download_themis.download_themis_img(self.day, self.station, force_download=True)

        self.assertTrue(temp_image_path.is_file())
        return


if __name__ == '__main__':
    unittest.main()
