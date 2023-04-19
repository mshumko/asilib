"""
Tests the asilib's Conjunction class.
"""
from datetime import datetime, timedelta
import dateutil.parser

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


t0 = dateutil.parser.parse('2014-05-05T04:49:10')

def test_find_none():
    """
    Verifies that no start or end conjunction intervals are identified.
    """
    img = asilib.themis('gill', time=t0, load_images=False, alt=110)
    times, lla = get_one_time_lla(img, alt=110)
    lla[:, 1] += 50  # Move the footprint path outside of the ASI FOV. 
    c = asilib.Conjunction(img, times, lla)
    df = c.find()
    assert df.shape == (0,2)
    return

def test_find_one():
    """
    Verifies that the start and end conjunction intervals are identified.
    """
    img = asilib.themis('gill', time=t0, load_images=False, alt=110)
    times, lla = get_one_time_lla(img, alt=110)
    c = asilib.Conjunction(img, times, lla)
    df = c.find()
    assert df.shape == (1,2)
    assert np.all(
        df.to_numpy() == np.array(
        [['2014-05-05T04:49:44.500000000', '2014-05-05T04:49:49.500000000']], 
        dtype='datetime64[ns]')
        )
    return

def test_find_multiple():
    """
    Verifies that the start and end conjunction intervals are identified.
    """
    n_passes = 5
    img = asilib.themis('gill', time=t0, load_images=False, alt=110)
    _times, _lla = get_one_time_lla(img, alt=110)

    times = np.repeat(_times, n_passes)
    lla = np.repeat(_lla, n_passes, axis=0)
    lon_offsets = np.linspace(-10, 10, num=n_passes)
    for i, lon_offset in enumerate(lon_offsets):
        times[i*_times.shape[0]:(i+1)*_times.shape[0]] += np.timedelta64(i, 'h')
        lla[i*_times.shape[0]:(i+1)*_times.shape[0], 1] += lon_offset

    c = asilib.Conjunction(img, times, lla)
    df = c.find()
    assert df.shape == (1,2)
    assert np.all(
        df.to_numpy() == np.array(
        [['2014-05-05T04:49:44.500000000', '2014-05-05T04:49:49.500000000']], 
        dtype='datetime64[ns]')
        )
    return

@pytest.mark.skipif(not irbem_imported, reason='IRBEM is not installed.')
def test_magnetic_tracing():
    raise AssertionError
    return

def get_one_time_lla(img:asilib.Imager, alt:int=110):
    """
    Get the satellite's footprint location at alt without mapping.

    Parameters
    ----------
    img: asilib.Imager
        An Imager instance used to create the path through zenith.
    alt: int
        The mapping altitude in km units.
    """
    n_times = 200
    lats = np.linspace(90, 0, num=n_times)
    lons = img.meta['lon'] * np.ones_like(lats)
    alts = alt * np.ones_like(lats)
    lla = np.array([lats, lons, alts]).T
    times = np.array([t0 + timedelta(seconds=i * 0.5) for i in range(n_times)])
    return times, lla