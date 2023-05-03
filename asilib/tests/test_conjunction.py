"""
Tests the asilib's Conjunction class.
"""
from datetime import datetime, timedelta
import dateutil.parser

import matplotlib.pyplot as plt
import matplotlib.testing.decorators
import pandas as pd
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
import asilib.asi.fake_asi
from asilib.tests.mock_footprint import footprint


t0 = dateutil.parser.parse('2014-05-05T04:49:10')
location_code = 'gill'


def test_conjunction_find_none():
    """
    Verifies that no start or end conjunction intervals are identified.
    """
    asi = asilib.themis(location_code, time=t0, load_images=False, alt=110)
    times, lla = footprint(asi.meta['lon'] + 100)
    c = asilib.Conjunction(asi, (times, lla))
    df = c.find()
    assert df.shape == (0, 4)
    return


def test_conjunction_find_multiple():
    """
    Verifies that multiple start and end conjunction intervals are identified.
    """
    asi = asilib.themis(location_code, time=t0, load_images=False, alt=110)
    times, lla = footprint(asi.meta['lon'], alt=110)

    c = asilib.Conjunction(asi, (times, lla))
    df = c.find(min_el=20)
    assert df.shape == (18, 4)
    assert np.all(
        df.to_numpy()
        == np.array(
            [
                [
                    pd.Timestamp('2015-01-01 01:44:48'),
                    pd.Timestamp('2015-01-01 01:45:42'),
                    1048,
                    1057,
                ],
                [
                    pd.Timestamp('2015-01-01 02:11:48'),
                    pd.Timestamp('2015-01-01 02:12:42'),
                    1318,
                    1327,
                ],
                [
                    pd.Timestamp('2015-01-01 03:19:48'),
                    pd.Timestamp('2015-01-01 03:20:42'),
                    1998,
                    2007,
                ],
                [
                    pd.Timestamp('2015-01-01 03:46:48'),
                    pd.Timestamp('2015-01-01 03:47:42'),
                    2268,
                    2277,
                ],
                [
                    pd.Timestamp('2015-01-01 04:54:48'),
                    pd.Timestamp('2015-01-01 04:55:42'),
                    2948,
                    2957,
                ],
                [
                    pd.Timestamp('2015-01-01 05:21:48'),
                    pd.Timestamp('2015-01-01 05:22:42'),
                    3218,
                    3227,
                ],
                [
                    pd.Timestamp('2015-01-01 06:29:48'),
                    pd.Timestamp('2015-01-01 06:30:42'),
                    3898,
                    3907,
                ],
                [
                    pd.Timestamp('2015-01-01 06:56:48'),
                    pd.Timestamp('2015-01-01 06:57:42'),
                    4168,
                    4177,
                ],
                [
                    pd.Timestamp('2015-01-01 08:04:48'),
                    pd.Timestamp('2015-01-01 08:05:42'),
                    4848,
                    4857,
                ],
                [
                    pd.Timestamp('2015-01-01 08:31:48'),
                    pd.Timestamp('2015-01-01 08:32:42'),
                    5118,
                    5127,
                ],
                [
                    pd.Timestamp('2015-01-01 09:39:48'),
                    pd.Timestamp('2015-01-01 09:40:42'),
                    5798,
                    5807,
                ],
                [
                    pd.Timestamp('2015-01-01 10:06:48'),
                    pd.Timestamp('2015-01-01 10:07:42'),
                    6068,
                    6077,
                ],
                [
                    pd.Timestamp('2015-01-01 11:14:48'),
                    pd.Timestamp('2015-01-01 11:15:42'),
                    6748,
                    6757,
                ],
                [
                    pd.Timestamp('2015-01-01 11:41:48'),
                    pd.Timestamp('2015-01-01 11:42:42'),
                    7018,
                    7027,
                ],
                [
                    pd.Timestamp('2015-01-01 12:49:48'),
                    pd.Timestamp('2015-01-01 12:50:42'),
                    7698,
                    7707,
                ],
                [
                    pd.Timestamp('2015-01-01 13:16:48'),
                    pd.Timestamp('2015-01-01 13:17:42'),
                    7968,
                    7977,
                ],
                [
                    pd.Timestamp('2015-01-01 14:24:48'),
                    pd.Timestamp('2015-01-01 14:25:42'),
                    8648,
                    8657,
                ],
                [
                    pd.Timestamp('2015-01-01 14:51:48'),
                    pd.Timestamp('2015-01-01 14:52:42'),
                    8918,
                    8927,
                ],
            ],
            dtype=object,
        )
    )
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_plot_conjunction_find_multiple'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_plot_conjunction_find_multiple():
    """
    Plots the ASI map and superposes the footprint start and end indices.
    """
    asi = asilib.themis(location_code, time=t0, load_images=False, alt=110)
    times, lla = footprint(asi.meta['lon'], alt=110)

    c = asilib.Conjunction(asi, (times, lla))
    df = c.find(min_el=20)

    _, ax = plt.subplots()
    asi._pcolormesh_nan(c._lon_map, c._lat_map, np.ones_like(c._lat_map), ax)
    ax.plot(lla[:, 1], lla[:, 0], 'k')
    ax.scatter(lla[df['start_index'], 1], lla[df['start_index'], 0], c='g', s=100)
    ax.scatter(lla[df['end_index'], 1], lla[df['end_index'], 0], c='c', s=100)

    ax.set(
        xlim=(np.nanmin(c._lon_map) - 5, np.nanmax(c._lon_map) + 5),
        ylim=(np.nanmin(c._lat_map) - 2, np.nanmax(c._lat_map) + 2),
    )
    return


def test_interp_sat():
    """
    Test if the satellite LLA is correctly interpolated and aligned to the ASI timestamps.

    The satellite LLA timestamps every 6 seconds, while THEMIS ASI timestamps are every
    3 seconds.
    """
    asi_time_range = (t0, t0 + timedelta(minutes=1))
    # Add 6 seconds so that the footprint interval completely encompasses the
    # asi time_range.
    footprint_time_range = (t0, t0 + timedelta(minutes=1, seconds=6))
    asi = asilib.themis(location_code, time_range=asi_time_range, alt=110)
    times, lla = footprint(asi.meta['lon'], time_range=footprint_time_range)
    c = asilib.Conjunction(asi, (times, lla))

    c.interp_sat()
    assert c.sat.shape == (20, 3)
    assert np.all(c.sat.index == c.imager.data.times)
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_plot_interp_sat'], tol=10, remove_text=True, extensions=['png']
)
def test_plot_interp_sat():
    """
    Test if the satellite LLA is correctly interpolated and aligned to the ASI timestamps.

    The satellite LLA timestamps every 6 seconds, while THEMIS ASI timestamps are every
    3 seconds.
    """
    asi_time_range = (t0, t0 + timedelta(minutes=1))
    # Add 6 seconds so that the footprint interval completely encompasses the
    # asi time_range.
    footprint_time_range = (t0, t0 + timedelta(minutes=1, seconds=6))
    asi = asilib.themis(location_code, time_range=asi_time_range, alt=110)
    times, lla = footprint(asi.meta['lon'], time_range=footprint_time_range)
    c = asilib.Conjunction(asi, (times, lla))

    c.interp_sat()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].scatter(times, lla[:, 0], c='k', s=50)
    ax[0].scatter(c.sat.index, c.sat['lat'], c='orange')

    ax[1].scatter(times, lla[:, 1], c='k', s=50)
    ax[1].scatter(c.sat.index, c.sat['lon'], c='orange')

    ax[0].axvline(c.imager.data.times[0])
    ax[0].axvline(c.imager.data.times[-1])
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_plot_interp_sat_wrap'], tol=10, remove_text=True, extensions=['png']
)
def test_plot_interp_sat_wrap():
    """
    Test if the satellite LLA is correctly interpolated and aligned to the ASI timestamps
    when the satellite's longitudes pass through the 180 meridian.

    The satellite LLA timestamps every 6 seconds, while THEMIS ASI timestamps are every
    3 seconds.
    """
    asi_time_range = (t0, t0 + timedelta(minutes=1))
    # Add 6 seconds so that the footprint interval completely encompasses the
    # asi time_range.
    footprint_time_range = (t0, t0 + timedelta(minutes=1, seconds=6))
    asi = asilib.themis(location_code, time_range=asi_time_range, alt=110)
    times, lla = footprint(-180, time_range=footprint_time_range, precession_rate=20)
    lla[lla[:, 1] < -180, 1] += 360
    c = asilib.Conjunction(asi, (times, lla))

    with pytest.warns(UserWarning):
        c.interp_sat()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].scatter(times, lla[:, 0], c='k', s=50)
    ax[0].scatter(c.sat.index, c.sat['lat'], c='orange')

    ax[1].scatter(times, lla[:, 1], c='k', s=50)
    ax[1].scatter(c.sat.index, c.sat['lon'], c='orange')

    ax[0].axvline(c.imager.data.times[0])
    ax[0].axvline(c.imager.data.times[-1])
    return


@pytest.mark.skipif(not irbem_imported, reason='IRBEM is not installed.')
def test_magnetic_tracing():
    raise AssertionError
    return


def test_azel_single_lla():
    """
    Tests that one LLA input, right above the imager, is mapped 90 degree elevation
    and near the pixel midpoint (it won't be exactly in the middle).
    """
    location = 'ATHA'
    time = datetime(2020, 1, 1)
    asi = asilib.asi.fake_asi.fake_asi(location, time=time)

    sat_lla = np.array([[asi.meta['lat'], asi.meta['lon'], 500]])
    sat_time = np.array([time])
    c = asilib.Conjunction(asi, (sat_time, sat_lla))
    azel, pixels = c.map_azel()

    # Test the El values
    assert round(azel[0, 1]) == 90

    # Test that the AzEl indices are within a pixel of zenith.
    assert np.max(abs(pixels[0] - asi.meta['resolution'][0] / 2)) <= 1
    return


def test_azel_multiple_lla():
    """
    Tests that multiple LLA inputs, nearby and above the imager, are mapped correctly.
    """
    location = 'ATHA'
    time = datetime(2020, 1, 1)
    asi = asilib.asi.fake_asi.fake_asi(location, time=time)

    num = 10
    lats = np.linspace(asi.meta['lat'] - 2, asi.meta['lat'] + 2, num=num)
    lons = np.linspace(asi.meta['lon'] - 2, asi.meta['lon'] + 2, num=num)
    sat_lla = np.stack((lats, lons, 500 * np.ones_like(lons))).T
    times = np.repeat(np.array([time]), num)
    c = asilib.Conjunction(asi, (times, sat_lla))
    azel, pixels = c.map_azel()

    assert np.all(
        np.isclose(
            azel,
            np.array(
                [
                    [211.47839446, 60.51190213],
                    [211.16566566, 66.37949079],
                    [210.85158923, 72.72795178],
                    [210.53615933, 79.4686885],
                    [210.2193701, 86.46258136],
                    [29.90121566, 86.4673325],
                    [29.58169008, 79.51031894],
                    [29.26078743, 72.8378054],
                    [28.93850173, 66.57984562],
                    [28.61482701, 60.81521379],
                ]
            ),
        )
    )

    assert np.all(
        np.isclose(
            pixels,
            np.array(
                [
                    [146.0, 364.0],
                    [161.0, 329.0],
                    [196.0, 379.0],
                    [197.0, 295.0],
                    [284.0, 322.0],
                    [272.0, 239.0],
                    [316.0, 193.0],
                    [250.0, 219.0],
                    [354.0, 164.0],
                    [279.0, 20.0],
                ]
            ),
        )
    )
    return
