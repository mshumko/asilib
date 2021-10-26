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
        self.location_code = 'LuCk'
        self.url = (
            'http://themis.ssl.berkeley.edu/data/themis/thg/l1/reg/'
            + f'{self.location_code.lower()}/{self.day.year}/{str(self.day.month).zfill(2)}/'
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
        search_pattern = f'{self.location_code}_{self.day.strftime("%Y%m%d%H")}'
        matched_hrefs = utils._search_hrefs(self.url, search_pattern=search_pattern)
        assert 'clg_l1_rgf_luck_2020080104_v01.cdf' in matched_hrefs
        return

    def test_validate_time(self):
        """
        Tests utils._validate_time.
        """
        valid_time_inputs = ['2016-01-01T10:05', '2016-01-01 10:05', datetime(2016, 1, 1, 10, 5)]

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
        """
        Tests utils._validate_time_range.
        """
        valid_time_inputs = [
            ['2016-01-01T10:05', '2016-01-01T10:10'],
            ['2016-01-01 10:05', '2016-01-01 10:10'],
            [datetime(2016, 1, 1, 10, 5), datetime(2016, 1, 1, 10, 10)],
        ]

        invalid_time_input_length = [
            ['2016-01-01T10:05', '2016-01-01T10:10', '2016-01-01T10:10'],
            [datetime(2016, 1, 1, 10, 5)],
            [],
            5,
            5.5,
        ]

        invalid_time_format = [['test', '2016-01-01T10:05'], [5, 5]]

        for time_range in valid_time_inputs:
            assert [
                datetime(2016, 1, 1, 10, 5),
                datetime(2016, 1, 1, 10, 10),
            ] == utils._validate_time_range(time_range)

        for time_range in invalid_time_input_length:
            with self.assertRaises(AssertionError):
                utils._validate_time_range(time_range)

        for time_range in invalid_time_format:
            with self.assertRaises(ValueError):
                utils._validate_time_range(time_range)
        return

    def test_get_hours(self):
        """
        Tests the hour time series from _get_hours. The four test cases comprise of two
        tests with either the start or end time are top of the hour, and two tests
        where either the start or end time are not at the top of the hour.
        """
        time_range = [datetime(2016, 1, 1, 10), datetime(2016, 1, 1, 12)]
        hours = utils._get_hours(time_range)
        assert hours == [datetime(2016, 1, 1, 10), datetime(2016, 1, 1, 11)]

        time_range = [datetime(2016, 1, 1, 10, 5, 1, 50000), datetime(2016, 1, 1, 12)]
        hours = utils._get_hours(time_range)
        assert hours == [datetime(2016, 1, 1, 10), datetime(2016, 1, 1, 11)]

        time_range = [datetime(2016, 1, 1, 10), datetime(2016, 1, 1, 12, 1)]
        hours = utils._get_hours(time_range)
        assert hours == [
            datetime(2016, 1, 1, 10),
            datetime(2016, 1, 1, 11),
            datetime(2016, 1, 1, 12),
        ]

        time_range = [datetime(2016, 1, 1, 10, 5), datetime(2016, 1, 1, 12, 1)]
        hours = utils._get_hours(time_range)
        assert hours == [
            datetime(2016, 1, 1, 10),
            datetime(2016, 1, 1, 11),
            datetime(2016, 1, 1, 12),
        ]
        return


if __name__ == '__main__':
    unittest.main()
