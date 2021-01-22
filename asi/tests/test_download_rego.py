import unittest
import requests
from datetime import datetime

from asi.download import download_rego
import asi.config as config

"""
Unit tests to check that the functions in download_rego.py are working correctly.
Run with "python3 test_download_rego.py -v" for the verbose output.
"""

class TestDownloadRego(unittest.TestCase):

    def setUp(self):
        """ Set up a few variables. """
        self.day = datetime(2017, 4, 13, 5, 10)
        self.station = 'Luck'
        self.url = download_rego.BASE_URL + \
            f'{self.day.year}/{str(self.day.month).zfill(2)}/{str(self.day.day).zfill(2)}/'
        return

    def test_server_response(self):
        """ Check that the server responds without an error. """
        r = requests.get(self.url)
        status_code = r.status_code
        # Check that the server status code is not
        # between 400-599 (error).
        self.assertNotEqual(status_code//100, 4)
        self.assertNotEqual(status_code//100, 5)
        return

    def test_href_is_found(self):
        """ Test that the href function can find the correct station_id """
        matched_hrefs = download_rego.search_hrefs(self.url, search_pattern=self.station)
        print(matched_hrefs)


if __name__ == '__main__':
    unittest.main()