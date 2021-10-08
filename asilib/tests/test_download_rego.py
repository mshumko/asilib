import unittest
import requests
import pathlib
from datetime import datetime

from asilib.io import download_rego
import asilib

"""
Unit tests to check that the functions in download_rego.py are working correctly.
Run with "python3 test_download_rego.py -v" for the verbose output.
"""


class TestDownloadRego(unittest.TestCase):
    def setUp(self):
        """Set up a few variables."""
        # http://themis.ssl.berkeley.edu/data/themis/thg/l1/reg/luck/2020/08/clg_l1_rgf_luck_2020080104_v01.cdf
        self.day = datetime(2020, 8, 1, 4)
        self.location_code = 'Luck'
        self.url = (
            download_rego.IMG_BASE_URL
            + f'{self.location_code.lower()}/{self.day.year}/{str(self.day.month).zfill(2)}/'
        )
        return

    def test_download_img(self):
        """
        Test the full REGO data downloader and download an hour file
        clg_l1_rgf_luck_2020080104_v01.cdf to ./rego/.
        """
        image_dir = pathlib.Path(asilib.config['ASI_DATA_DIR'], 'rego')
        image_path = image_dir / 'clg_l1_rgf_luck_2020080104_v01.cdf'

        download_rego.download_rego_img(self.location_code, time=self.day, force_download=True)

        assert image_path.is_file()
        return


if __name__ == '__main__':
    unittest.main()
