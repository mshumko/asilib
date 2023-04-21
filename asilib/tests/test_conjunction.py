"""
Tests the asilib's Conjunction class.
"""
from datetime import datetime, timedelta
import dateutil.parser

import matplotlib.pyplot as plt
import pytest
import numpy as np

try:
    import IRBEM
    irbem_imported = True
except ImportError as err:
    # You can also get a ModuleNotFoundError if IRBEM is not installed
    # (as compared to failed to import), but it is a subclass of ImportError.
    irbem_imported = False

import asilib
from asilib.tests.mock_footprint import footprint


t0 = dateutil.parser.parse('2014-05-05T04:49:10')

def test_find_none():
    """
    Verifies that no start or end conjunction intervals are identified.
    """
    img = asilib.themis('gill', time=t0, load_images=False, alt=110)
    times, lla = footprint(img.meta['lon']+100)
    c = asilib.Conjunction(img, times, lla)
    df = c.find()
    assert df.shape == (0,2)
    return

def test_find_multiple():
    """
    Verifies that multiple start and end conjunction intervals are identified.
    """
    img = asilib.themis('gill', time=t0, load_images=False, alt=110)
    times, lla = footprint(img.meta['lon'], alt=110)

    c = asilib.Conjunction(img, times, lla)
    df = c.find()
    assert df.shape == (18,2)
    assert np.all(
        df.to_numpy() == np.array([
            ['2015-01-01T01:44:48.000000000', '2015-01-01T01:45:42.000000000'],
            ['2015-01-01T02:11:48.000000000', '2015-01-01T02:12:42.000000000'],
            ['2015-01-01T03:19:48.000000000', '2015-01-01T03:20:42.000000000'],
            ['2015-01-01T03:46:48.000000000', '2015-01-01T03:47:42.000000000'],
            ['2015-01-01T04:54:48.000000000', '2015-01-01T04:55:42.000000000'],
            ['2015-01-01T05:21:48.000000000', '2015-01-01T05:22:42.000000000'],
            ['2015-01-01T06:29:48.000000000', '2015-01-01T06:30:42.000000000'],
            ['2015-01-01T06:56:48.000000000', '2015-01-01T06:57:42.000000000'],
            ['2015-01-01T08:04:48.000000000', '2015-01-01T08:05:42.000000000'],
            ['2015-01-01T08:31:48.000000000', '2015-01-01T08:32:42.000000000'],
            ['2015-01-01T09:39:48.000000000', '2015-01-01T09:40:42.000000000'],
            ['2015-01-01T10:06:48.000000000', '2015-01-01T10:07:42.000000000'],
            ['2015-01-01T11:14:48.000000000', '2015-01-01T11:15:42.000000000'],
            ['2015-01-01T11:41:48.000000000', '2015-01-01T11:42:42.000000000'],
            ['2015-01-01T12:49:48.000000000', '2015-01-01T12:50:42.000000000'],
            ['2015-01-01T13:16:48.000000000', '2015-01-01T13:17:42.000000000'],
            ['2015-01-01T14:24:48.000000000', '2015-01-01T14:25:42.000000000'],
            ['2015-01-01T14:51:48.000000000', '2015-01-01T14:52:42.000000000'],
            ], dtype='datetime64[ns]')
        )
    return

@pytest.mark.skipif(not irbem_imported, reason='IRBEM is not installed.')
def test_magnetic_tracing():
    raise AssertionError
    return