"""
Tests the asilib's Conjunction class.
"""
from datetime import datetime, timedelta
import dateutil.parser
import string

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
import asilib.asi
from asilib.imager import Skymap_Cleaner
from .mock_footprint import footprint


t0 = dateutil.parser.parse('2014-05-05T04:49:10')
location_code = 'gill'


def test_conjunction_find_none():
    """
    Verifies that no start or end conjunction intervals are identified.
    """
    asi = asilib.asi.themis(location_code, time=t0, load_images=False, alt=110)
    times, lla = footprint(asi.meta['lon'] + 100)
    c = asilib.Conjunction(asi, (times, lla))
    df = c.find()
    assert df.shape == (0, 4)
    return


def test_conjunction_find_multiple():
    """
    Verifies that multiple start and end conjunction intervals are identified.
    """
    asi = asilib.asi.themis(location_code, time=t0, load_images=False, alt=110)
    times, lla = footprint(asi.meta['lon'], alt=110)

    c = asilib.Conjunction(asi, (times, lla))
    df = c.find(min_elevation=20)
    assert df.shape == (18, 4)
    assert np.all(
        df.to_numpy()
        == np.array(
            [
            [
                pd.Timestamp('2015-01-01 01:44:42'),
                pd.Timestamp('2015-01-01 01:45:42'), 1047, 1057
            ],
            [   pd.Timestamp('2015-01-01 02:11:48'),
                pd.Timestamp('2015-01-01 02:12:48'), 1318, 1328
            ],
            [   pd.Timestamp('2015-01-01 03:19:42'),
                pd.Timestamp('2015-01-01 03:20:42'), 1997, 2007
            ],
            [   pd.Timestamp('2015-01-01 03:46:48'),
                pd.Timestamp('2015-01-01 03:47:48'), 2268, 2278
            ],
            [   pd.Timestamp('2015-01-01 04:54:42'),
                pd.Timestamp('2015-01-01 04:55:42'), 2947, 2957
            ],
            [   pd.Timestamp('2015-01-01 05:21:48'),
                pd.Timestamp('2015-01-01 05:22:48'), 3218, 3228
            ],
            [   pd.Timestamp('2015-01-01 06:29:42'),
                pd.Timestamp('2015-01-01 06:30:42'), 3897, 3907
            ],
            [   pd.Timestamp('2015-01-01 06:56:48'),
                pd.Timestamp('2015-01-01 06:57:48'), 4168, 4178
            ],
            [   pd.Timestamp('2015-01-01 08:04:42'),
                pd.Timestamp('2015-01-01 08:05:42'), 4847, 4857
            ],
            [   pd.Timestamp('2015-01-01 08:31:48'),
                pd.Timestamp('2015-01-01 08:32:48'), 5118, 5128
            ],
            [   pd.Timestamp('2015-01-01 09:39:42'),
                pd.Timestamp('2015-01-01 09:40:42'), 5797, 5807
            ],
            [   pd.Timestamp('2015-01-01 10:06:48'),
                pd.Timestamp('2015-01-01 10:07:48'), 6068, 6078
            ],
            [   pd.Timestamp('2015-01-01 11:14:42'),
                pd.Timestamp('2015-01-01 11:15:42'), 6747, 6757
            ],
            [   pd.Timestamp('2015-01-01 11:41:48'),
                pd.Timestamp('2015-01-01 11:42:48'), 7018, 7028
            ],
            [   pd.Timestamp('2015-01-01 12:49:42'),
                pd.Timestamp('2015-01-01 12:50:42'), 7697, 7707
            ],
            [   pd.Timestamp('2015-01-01 13:16:48'),
                pd.Timestamp('2015-01-01 13:17:48'), 7968, 7978
            ],
            [   pd.Timestamp('2015-01-01 14:24:42'),
                pd.Timestamp('2015-01-01 14:25:42'), 8647, 8657
            ],
            [
                pd.Timestamp('2015-01-01 14:51:48'),
                pd.Timestamp('2015-01-01 14:52:48'), 8918, 8928
            ]
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
    asi = asilib.asi.themis(location_code, time=t0, load_images=False, alt=110)
    times, lla = footprint(asi.meta['lon'], alt=110)

    c = asilib.Conjunction(asi, (times, lla))
    df = c.find(min_elevation=20)

    _cleaner = Skymap_Cleaner(asi.skymap['lon'], asi.skymap['lat'], asi.skymap['el'])
    _cleaner.mask_elevation(20)
    _lon_map, _lat_map = _cleaner.remove_nans()

    _, ax = plt.subplots()    
    ax.pcolormesh(_lon_map, _lat_map, np.ones_like(_lat_map))
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
    asi = asilib.asi.themis(location_code, time_range=asi_time_range, alt=110)
    times, lla = footprint(asi.meta['lon'], time_range=footprint_time_range)
    c = asilib.Conjunction(asi, (times, lla))

    c.interp_sat()
    assert c.sat.shape == (20, 3)
    assert np.all(c.sat.index == c.imager.data.time)
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
    asi = asilib.asi.themis(location_code, time_range=asi_time_range, alt=110)
    times, lla = footprint(asi.meta['lon'], time_range=footprint_time_range)
    c = asilib.Conjunction(asi, (times, lla))

    c.interp_sat()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].scatter(times, lla[:, 0], c='k', s=50)
    ax[0].scatter(c.sat.index, c.sat['lat'], c='orange')

    ax[1].scatter(times, lla[:, 1], c='k', s=50)
    ax[1].scatter(c.sat.index, c.sat['lon'], c='orange')

    ax[0].axvline(c.imager.data.time[0])
    ax[0].axvline(c.imager.data.time[-1])
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
    asi = asilib.asi.themis(location_code, time_range=asi_time_range, alt=110)
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

    ax[0].axvline(c.imager.data.time[0])
    ax[0].axvline(c.imager.data.time[-1])
    return


@pytest.mark.skipif(not irbem_imported, reason='IRBEM is not installed.')
def test_magnetic_tracing():
    location = 'ATHA'
    dt = 10  # seconds
    time_range = (datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 0, 1, 0))
    asi = asilib.asi.fake_asi.fake_asi(location, time_range=time_range)

    n = int((time_range[1]-time_range[0]).total_seconds()//dt)
    sat_time = np.array([time_range[0] + timedelta(seconds=i*dt) 
                         for i in range(n)])
    lats = np.linspace(asi.meta['lat'] - 2, asi.meta['lat'] + 2, num=sat_time.shape[0])
    lons = np.linspace(asi.meta['lon'] - 2, asi.meta['lon'] + 2, num=sat_time.shape[0])
    sat_lla = np.stack((lats, lons, 500 * np.ones_like(lons))).T
    c = asilib.Conjunction(asi, (sat_time, sat_lla))

    c.lla_footprint(110)
    
    assert np.all(c.sat.index == sat_time)
    np.testing.assert_almost_equal(c.sat['alt'], 110*np.ones_like(c.sat['alt']), decimal=1)
    np.testing.assert_almost_equal(
        c.sat['lat'],
        np.array([53.63169896, 54.39262177, 55.1545005 , 55.91733138, 56.68123125,
            57.44607773])
    )
    np.testing.assert_almost_equal(
        c.sat['lon'],
        np.array([-114.9137138, -114.12793  , -113.342572 , -112.5576476,
            -111.7731121, -110.9890267])
    )
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
    assert azel[0, 1] == 90

    # Test that the AzEl indices are within a pixel of zenith.
    assert np.max(abs(pixels[0, :] - asi.meta['resolution'][0] / 2)) <= 1
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
                    [310.0, 167.0],
                    [303.0, 178.0],
                    [295.0, 190.0],
                    [285.0, 205.0],
                    [273.0, 227.0],
                    [239.0, 285.0],
                    [226.0, 306.0],
                    [218.0, 322.0],
                    [212.0, 335.0],
                    [206.0, 346.0],
                ]
            ),
        )
    )
    return


def test_intensity_closest_pixel():
    """
    Test the auroral intensity from the nearest pixel.
    """
    location_code = 'RANK'
    alt = 110  # km
    time_range = (datetime(2017, 9, 15, 2, 32, 0), datetime(2017, 9, 15, 2, 35, 0))

    asi = asilib.asi.themis(location_code, time_range=time_range, alt=alt)

    # Create the fake satellite track coordinates: latitude, longitude, altitude (LLA).
    # This is a north-south satellite track oriented to the east of the THEMIS/RANK
    # imager.
    n = int((time_range[1] - time_range[0]).total_seconds() / 3)  # 3 second cadence.
    lats = np.linspace(asi.meta["lat"] + 5, asi.meta["lat"] - 5, n)
    lons = (asi.meta["lon"] - 0.5) * np.ones(n)
    alts = alt * np.ones(n)  # Altitude needs to be the same as the skymap.
    sat_lla = np.array([lats, lons, alts]).T
    # Normally the satellite time stamps are not the same as the ASI.
    # You may need to call Conjunction.interp_sat() to find the LLA coordinates
    # at the ASI timestamps.
    sat_time = asi.data.time

    conjunction_obj = asilib.Conjunction(asi, (sat_time, sat_lla))
    nearest_pixel_intensity = conjunction_obj.intensity(box=None)
    assert np.all(
        np.isclose(
            nearest_pixel_intensity,
            np.array(
                [
                    4328.0,
                    4298.0,
                    4258.0,
                    4252.0,
                    4288.0,
                    4211.0,
                    4118.0,
                    4120.0,
                    4089.0,
                    4004.0,
                    4060.0,
                    4005.0,
                    3991.0,
                    3908.0,
                    3890.0,
                    3887.0,
                    3824.0,
                    3757.0,
                    3678.0,
                    3643.0,
                    3644.0,
                    3658.0,
                    3643.0,
                    3548.0,
                    3574.0,
                    3565.0,
                    3526.0,
                    3534.0,
                    3480.0,
                    3494.0,
                    3933.0,
                    12374.0,
                    16140.0,
                    5523.0,
                    7557.0,
                    12812.0,
                    5121.0,
                    4202.0,
                    4051.0,
                    4101.0,
                    4015.0,
                    4198.0,
                    4136.0,
                    4331.0,
                    4306.0,
                    4368.0,
                    4164.0,
                    4462.0,
                    4541.0,
                    4515.0,
                    4612.0,
                    4705.0,
                    4861.0,
                    4960.0,
                    4959.0,
                    5036.0,
                    5098.0,
                    5092.0,
                    5158.0,
                    5295.0,
                ]
            ),
        )
    )
    return

@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_rgb_intensity_closest_pixel'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_rgb_intensity_closest_pixel():
    """
    Test the RGB auroral intensity from the nearest pixel.
    """
    time_range = (datetime(2021, 11, 4, 7, 2, 0), datetime(2021, 11, 4, 7, 4, 0))
    location_code = 'LUCK'
    alt = 110

    asi = asilib.asi.trex_rgb(location_code, time_range=time_range, alt=alt)

    # Create a orbit with a footprint every 5 seconds.
    dt_sat = 5
    n_sat_times = int((time_range[1]-time_range[0]).total_seconds()//dt_sat)
    sat_times = np.array([time_range[0]+timedelta(seconds=dt_sat*i) for i in range(n_sat_times)])
    lats = np.linspace(asi.meta['lat']-3, asi.meta['lat']+3, num=n_sat_times)
    lons = (asi.meta['lon']-2) * np.linspace(1.02, 0.98, num=n_sat_times)
    alts = alt * np.ones_like(lats)
    lla = np.stack((lats, lons, alts)).T

    c = asilib.Conjunction(asi, (sat_times, lla))
    intensity = c.intensity()

    plot_n_times = 4
    fig = plt.figure(figsize=(3*plot_n_times, 6))
    gs = gridspec.GridSpec(2, plot_n_times)
    ax = [fig.add_subplot(gs[0, i]) for i in range(plot_n_times)]
    bx = fig.add_subplot(gs[1, :])
    dt_plot = int((time_range[1]-time_range[0]).total_seconds()//plot_n_times)

    for i, ax_i in enumerate(ax):
        t_i = time_range[0] + timedelta(seconds=(i+0.5)*dt_plot)
        asi_image = asi[t_i]
        asi_image.plot_map(ax=ax_i)
        ax_i.text(
            0, 0.99, f'({string.ascii_uppercase[i]}) {asi_image.file_info["time"]}', 
            transform=ax_i.transAxes, va='top', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='red', pad=0.01),
            )
        ax_i.plot(lla[:, 1], lla[:, 0], c='r', ls=':')

        idt_min = np.argmin(np.abs([sat_time_i - t_i for sat_time_i in sat_times]))
        ax_i.scatter(lla[idt_min, 1], lla[idt_min, 0], c='r', marker='x')
        bx.axvline(sat_times[idt_min], c='k')

    for color, _intensity in zip(['r', 'g', 'b'], intensity.T):
        bx.plot(c.sat.index, _intensity, c=color)
    fig.tight_layout()
    return

def test_intensity_area():
    """
    Test the mean auroral intensity in a 10x10 km box.
    """
    # ASI parameters
    location_code = 'RANK'
    alt = 110  # km
    time_range = (datetime(2017, 9, 15, 2, 32, 0), datetime(2017, 9, 15, 2, 35, 0))

    asi = asilib.asi.themis(location_code, time_range=time_range, alt=alt)

    # Create the fake satellite track coordinates: latitude, longitude, altitude (LLA).
    # This is a north-south satellite track oriented to the east of the THEMIS/RANK
    # imager.
    n = int((time_range[1] - time_range[0]).total_seconds() / 3)  # 3 second cadence.
    lats = np.linspace(asi.meta["lat"] + 5, asi.meta["lat"] - 5, n)
    lons = (asi.meta["lon"] - 0.5) * np.ones(n)
    alts = alt * np.ones(n)  # Altitude needs to be the same as the skymap.
    sat_lla = np.array([lats, lons, alts]).T
    # Normally the satellite time stamps are not the same as the ASI.
    # You may need to call Conjunction.interp_sat() to find the LLA coordinates
    # at the ASI timestamps.
    sat_time = asi.data.time

    conjunction_obj = asilib.Conjunction(asi, (sat_time, sat_lla))
    area_intensity = conjunction_obj.intensity(box=(10, 10))

    assert np.all(
        np.isclose(
            area_intensity,
            np.array(
                [
                    4429.5,
                    4307.0,
                    4359.0,
                    4258.0,
                    4277.0,
                    4307.0,
                    4142.66666667,
                    4129.0,
                    4092.0,
                    4002.0,
                    3991.66666667,
                    3990.0,
                    3987.0,
                    3906.66666667,
                    3895.33333333,
                    3909.0,
                    3829.75,
                    3774.0,
                    3741.75,
                    3699.8,
                    3685.88888889,
                    3616.9,
                    3615.3125,
                    3585.3125,
                    3548.33333333,
                    3532.23076923,
                    3501.77142857,
                    3496.58536585,
                    3485.09803922,
                    3532.40740741,
                    3904.0,
                    11476.43137255,
                    15238.31818182,
                    5658.125,
                    7793.53571429,
                    12330.76190476,
                    5629.0,
                    4244.69230769,
                    4105.81818182,
                    4064.2,
                    4055.14285714,
                    4131.33333333,
                    4180.16666667,
                    4236.4,
                    4297.33333333,
                    4354.66666667,
                    4291.0,
                    4386.0,
                    4530.5,
                    4537.0,
                    4605.5,
                    4627.0,
                    4711.0,
                    4799.5,
                    4959.0,
                    5078.0,
                    5014.0,
                    5007.5,
                    5070.0,
                    5156.0,
                ]
            ),
        )
    )
    return

@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_rgb_intensity_area'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_rgb_intensity_area():
    """
    Test the RGB auroral intensity in a (10, 10) km area.
    """
    time_range = (datetime(2021, 11, 4, 7, 2, 0), datetime(2021, 11, 4, 7, 4, 0))
    location_code = 'LUCK'
    alt = 110

    asi = asilib.asi.trex_rgb(location_code, time_range=time_range, alt=alt)

    # Create a orbit with a footprint every 5 seconds.
    dt_sat = 5
    n_sat_times = int((time_range[1]-time_range[0]).total_seconds()//dt_sat)
    sat_times = np.array([time_range[0]+timedelta(seconds=dt_sat*i) for i in range(n_sat_times)])
    lats = np.linspace(asi.meta['lat']-3, asi.meta['lat']+3, num=n_sat_times)
    lons = (asi.meta['lon']-2) * np.linspace(1.02, 0.98, num=n_sat_times)
    alts = alt * np.ones_like(lats)
    lla = np.stack((lats, lons, alts)).T

    c = asilib.Conjunction(asi, (sat_times, lla))
    intensity = c.intensity(box=(10,10))

    plot_n_times = 4
    fig = plt.figure(figsize=(3*plot_n_times, 6))
    gs = gridspec.GridSpec(2, plot_n_times)
    ax = [fig.add_subplot(gs[0, i]) for i in range(plot_n_times)]
    bx = fig.add_subplot(gs[1, :])
    dt_plot = int((time_range[1]-time_range[0]).total_seconds()//plot_n_times)

    for i, ax_i in enumerate(ax):
        t_i = time_range[0] + timedelta(seconds=(i+0.5)*dt_plot)
        asi_image = asi[t_i]
        asi_image.plot_map(ax=ax_i)
        ax_i.text(
            0, 0.99, f'({string.ascii_uppercase[i]}) {asi_image.file_info["time"]}', 
            transform=ax_i.transAxes, va='top', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='red', pad=0.01),
            )
        ax_i.plot(lla[:, 1], lla[:, 0], c='r', ls=':')

        idt_min = np.argmin(np.abs([sat_time_i - t_i for sat_time_i in sat_times]))
        ax_i.scatter(lla[idt_min, 1], lla[idt_min, 0], c='r', marker='x')
        bx.axvline(sat_times[idt_min], c='k')

    for color, _intensity in zip(['r', 'g', 'b'], intensity.T):
        bx.plot(c.sat.index, _intensity, c=color)
    fig.tight_layout()
    return