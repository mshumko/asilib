import unittest
import requests
from datetime import datetime

import asilib.io.utils as utils
import asilib

"""
Unit tests to check that the functions in download_rego.py are working correctly.
Run with "python3 test_download_rego.py -v" for the verbose output.
"""


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up a few variables."""
        # http://themis.ssl.berkeley.edu/data/themis/thg/l1/reg/luck/2020/08/clg_l1_rgf_luck_2020080104_v01.cdf
        self.day = datetime(2020, 8, 1, 4)
        self.station = 'LuCk'
        self.url = (
            asilib.io.download_rego.IMG_BASE_URL
            + f'{self.station.lower()}/{self.day.year}/{str(self.day.month).zfill(2)}/'
        )
        return

    def test_server_response(self):
        """Check that the server responds without an error, 400-599 status_codes."""
        r = requests.get(self.url)
        assert r.status_code // 100 != 4
        assert r.status_code // 100 != 5
        return

    def test_href_is_found(self):
        """Test that the href function can find the first file on August 2020."""
        search_pattern = f'{self.station}_{self.day.strftime("%Y%m%d%H")}'
        matched_hrefs = utils._search_hrefs(self.url, search_pattern=search_pattern)
        assert 'clg_l1_rgf_luck_2020080104_v01.cdf' in matched_hrefs
        return

    def test_validate_time(self):
        """
        Tests utils._validate_time.
        """
        valid_time_inputs = [
                    '2016-01-01T10:05',
                    '2016-01-01 10:05',
                    datetime(2016, 1, 1, 10, 5)
        ]

        invalid_time_inputs = [
                    'Two thousand and sixteen',
                    5,
                    10000.0,
        ]
        
        for t in valid_time_inputs:
            assert datetime(2016, 1, 1, 10, 5) == utils._validate_time(t)

        for t in invalid_time_inputs:        
            with self.assertRaises(ValueError):
                utils._validate_time(t)

        return

    def test_validate_time_range(self):
        raise NotImplementedError

    def test_get_hours(self):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
