import unittest
import requests
import pathlib
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
        config.ASI_DATA_DIR = pathlib.Path(__file__).parent.resolve() # Overwrite the data directory to here.
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
        """ Test that the href function can find the Lucky Lake (LUCK) station_id """
        matched_hrefs = download_rego.search_hrefs(self.url, search_pattern=self.station)
        self.assertEqual(matched_hrefs, ['luck_rego-649/'])

    def test_download(self):
        """ 
        Test the full REGO data downloader and download a minute file 
        '20170413_0510_luck_rego-649_6300.pgm.gz' to ./rego/.
        """
        temp_image_path = pathlib.Path(config.ASI_DATA_DIR / 'rego' / 
            '20170413_0510_luck_rego-649_6300.pgm.gz')
        download_rego.download(self.day, self.station)

        self.assertTrue(temp_image_path.is_file())
        
        # Remove the temp folder and image file.
        temp_image_path.unlink(missing_ok=True)
        temp_image_path.parent.rmdir()
        return


if __name__ == '__main__':
    unittest.main()