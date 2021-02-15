import unittest
import requests
import pathlib
from datetime import datetime

from asilib.download import download_rego
from asilib import config

"""
Unit tests to check that the functions in download_rego.py are working correctly.
Run with "python3 test_download_rego.py -v" for the verbose output.
"""


class TestDownloadRego(unittest.TestCase):
    def setUp(self):
        """ Set up a few variables. """
        # http://themis.ssl.berkeley.edu/data/themis/thg/l1/reg/luck/2020/08/clg_l1_rgf_luck_2020080104_v01.cdf
        self.day = datetime(2020, 8, 1, 4)
        self.station = 'Luck'
        self.url = (
            download_rego.IMG_BASE_URL
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

    def test_href_is_found(self):
        """ Test that the href function can find the first file on August 2020. """
        search_pattern = f'{self.station}_{self.day.strftime("%Y%m%d%H")}'
        matched_hrefs = download_rego.search_hrefs(self.url, search_pattern=search_pattern)
        self.assertTrue(matched_hrefs[0], 'clg_l1_rgf_luck_2020080104_v01.cdf')

    def test_download_img(self):
        """
        Test the full REGO data downloader and download an hour file
        clg_l1_rgf_luck_2020080104_v01.cdf to ./rego/.
        """
        temp_image_dir = pathlib.Path(config.ASI_DATA_DIR, 'rego')
        temp_image_path = temp_image_dir / 'clg_l1_rgf_luck_2020080104_v01.cdf'

        download_rego.download_rego_img(self.day, self.station, force_download=True)

        self.assertTrue(temp_image_path.is_file())
        return


if __name__ == '__main__':
    unittest.main()
