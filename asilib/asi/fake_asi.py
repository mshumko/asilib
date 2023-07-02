"""
An ASI for testing asilib.Imager.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import asilib
import asilib.utils as utils


def fake_asi(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    pixel_center: bool = True,
) -> asilib.Imager:
    """
    Create an Imager instance with the fake_asi images and skymaps.

    Parameters
    ----------
    location_code: str
        The ASI's location code (four letters). Can be either GILL, ATHA, or TPAS.
        Case insensitive.
    time: str or datetime.datetime
        A time to look for the ASI data at. Either the time or the time_range
        must be specified.
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    alt: int
        The reference skymap altitude, in kilometers.
    pixel_center: bool
        If True, then the skymap specifies the pixel centers, otherwise the skymap specifies
        the pixel vertices. Specifying the vertices more accurately describes how the pixels
        field of view transforms from a square to a polynomial.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        The an Imager instance with the fake_asi data.
    """
    location_code = location_code.upper()
    locations = asi_info()
    _location = locations.loc[locations.index == location_code, :]

    meta = get_meta(_location)
    skymap = get_skymap(meta, alt=alt, pixel_center=pixel_center)
    file_info = get_file_info(meta, time=time, time_range=time_range)
    return asilib.Imager(file_info, meta, skymap)


def asi_info() -> pd.DataFrame:
    """
    The test ASI has three locations, a, b, and c.
    """
    locations = pd.DataFrame(
        data=np.array(
            [
                [56.3494, -94.7056, 500],
                [54.7213, -113.285, 1000],
                [53.8255, -101.2427, 0],
            ]
        ),
        columns=['lat', 'lon', 'alt'],
        index=['GILL', 'ATHA', 'TPAS'],
    )
    return locations


def get_meta(location_dict):
    """
    Get the ASI metadata.
    """
    meta = {
        'array': 'TEST',
        'location': location_dict.index[0],
        'lat': location_dict['lat'][0],
        'lon': location_dict['lon'][0],
        'alt': location_dict['alt'][0] / 1e3,  # km
        'cadence': 10,
        'resolution': (512, 512),
    }
    return meta


def get_skymap(meta: dict, alt: int, pixel_center: bool = True):
    """
    Create a skymap based on the ASI location in the metadata.

    Parameters
    ----------
    meta: dict
        The ASI metadata with the imager resolution and cadence.
    alt: int
        The reference skymap altitude, in kilometers.
    pixel_center: bool
        If True, then the skymap specifies the pixel centers, otherwise the skymap specifies
        the pixel vertices. Specifying the vertices more accurately describes how the pixels
        field of view transforms from a square to a polynomial.
    """
    assert alt in [90, 110, 150], (
        f'The {alt} km altitude does not have a corresponding map: '
        'valid_altitudes=[90, 110, 150].'
    )
    skymap = {}
    lon_bounds = [meta['lon'] - 10 * (alt / 110), meta['lon'] + 10 * (alt / 110)]
    lat_bounds = [meta['lat'] - 10 * (alt / 110), meta['lat'] + 10 * (alt / 110)]

    if pixel_center:
        pad = 0
    else:
        pad = 1

    # TODO: Add tests for skymaps specifying pixel edges too.
    # lon_bounds[::-1] so that east is to the right.
    _lons, _lats = np.meshgrid(
        np.linspace(*lon_bounds[::-1], num=meta['resolution'][0] + pad),
        np.linspace(*lat_bounds, num=meta['resolution'][1] + pad),
    )
    std = 5 * (alt / 110)
    dst = np.sqrt((_lons - meta['lon']) ** 2 + (_lats - meta['lat']) ** 2)
    # the 105 multiplier could be 90, but I chose 105 (and -15 offset) to make the
    # skymap realistic: the edges are NaNs.
    elevations = 105 * np.exp(-(dst**2) / (2.0 * std**2)) - 15
    elevations[elevations < 0] = np.nan
    # These are the required skymap keys for asilib.Imager to work.
    skymap['el'] = elevations
    skymap['alt'] = alt
    skymap['lon'] = _lons
    skymap['lon'][~np.isfinite(skymap['el'])] = np.nan
    skymap['lat'] = _lats
    skymap['lat'][~np.isfinite(skymap['el'])] = np.nan
    skymap['path'] = __file__

    # Calculate the azimuthal angle using cross product between a northward-pointing unit vector
    # and the (_lons, _lats) grid. See https://stackoverflow.com/a/16544330 for an explanation.
    dot_product = 0 * (_lons - meta['lon']) + 1 * (_lats - meta['lat'])
    determinant = 0 * (_lats - meta['lat']) - 1 * (_lons - meta['lon'])
    skymap['az'] = (180 / np.pi) * np.arctan2(determinant, dot_product)
    # transform so it goes 0->360 in a clockwise direction.
    skymap['az'] = -1 * skymap['az']
    skymap['az'][skymap['az'] < 0] = 360 + skymap['az'][skymap['az'] < 0]
    return skymap


def get_file_info(
    meta: dict, time: utils._time_type = None, time_range: utils._time_range_type = None
) -> dict:
    """
    Get some images and time stamps. One image and time stamp if time is specified,
    or multiple images if time_range is specified.

    Parameters
    ----------
    meta: dict
        The ASI metadata with the imager resolution and cadence.
    time: str or datetime.datetime
        A time to look for the ASI data at. Either the time or the time_range
        must be specified.
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    """
    if (time is not None) and (time_range is not None):
        raise ValueError("time and time_range can't be simultaneously specified.")
    if (time is None) and (time_range is None):
        raise ValueError("Either time or time_range must be specified.")

    if time is not None:
        time = utils.validate_time(time)
        file_path = _get_file_path(meta, time)
        image_times, images = _data_loader(file_path)
        start_file_time = time.replace(minute=0, second=0, microsecond=0)

        _data = {
            'time': time,
            'path': [file_path],
            'start_time': [start_file_time],
            'end_time': [start_file_time + timedelta(hours=1)],
            'loader': _data_loader,
        }

    else:
        time_range = utils.validate_time_range(time_range)
        start_file_time = time_range[0].replace(minute=0, second=0, microsecond=0)
        hours = (
            int((time_range[1] - time_range[0]).total_seconds() // 3600) + 1
        )  # +1 to load the final hour.

        # These are all of the keys required by asilib.Imager.
        _data = {
            'time_range': time_range,
            'path': [
                _get_file_path(meta, start_file_time + timedelta(hours=i)) for i in range(hours)
            ],
            'start_time': [start_file_time + timedelta(hours=i) for i in range(hours)],
            'end_time': [start_file_time + timedelta(hours=1 + i) for i in range(hours)],
            'loader': _data_loader,
        }
    return _data


def _data_loader(file_path):
    """
    Given a file_path, open the file and return the time stamps and images.

    Time stamps are every 10 seconds. The images are all 0s except a horizontal and a vertical
    line whose position is determined by seconds_since_start modulo resolution (512).
    """
    # Assume that the image time stamps are at exact seconds
    date_time_str = '_'.join(file_path.split('_')[:2])
    file_time = datetime.strptime(date_time_str, '%Y%m%d_%H%M%S')

    times = np.array(
        [file_time + timedelta(seconds=i * 10) for i in range(3600 // 10)]
    )  # 10 s cadence

    images = np.ones((times.shape[0], 512, 512))
    for i, time in enumerate(times):
        sec = (time - file_time).total_seconds()
        images[i, int(sec % 512) - 10 : int(sec % 512) + 10, :] = 255
        images[i, :, int(sec % 512) - 10 : int(sec % 512) + 10] = 100
    return times, images


def _get_file_path(meta, time):
    return f'{time:%Y%m%d_%H}0000_{meta["location"]}_fake_asi.png'  # does not exist.


def plot_skymap(location_code, alt=110, pixel_center=True):
    """
    Visualize the skymap to get a better idea on what a realistic one looks like
    (if perfectly aligned).
    """
    location_code = location_code.upper()
    locations = asi_info()
    _location = locations.loc[locations.index == location_code, :]
    meta = get_meta(_location)

    skymap = get_skymap(meta, alt=alt, pixel_center=pixel_center)
    keys = ['el', 'az', 'lat', 'lon']
    fig, ax = plt.subplots(1, len(keys), sharex=True, sharey=True, figsize=(3.7 * len(keys), 4))

    for ax_i, key in zip(ax, keys):
        p = ax_i.pcolormesh(skymap[key])
        plt.colorbar(p, ax=ax_i)
        ax_i.set_title(key)
    plt.suptitle('asilib | test ASI skymap')
    plt.tight_layout()
    return


if __name__ == '__main__':
    # asi = fake_asi('GILL', time_range=('2015-01-01T15:00:15.17', '2015-01-01T20:00'))
    asi = fake_asi('GILL', time='2015-01-01T15:14:00.17')
    asi.plot_fisheye(color_bounds=(1, 255), origin=(0.85, 0.15), cardinal_directions='NEWS')
    plt.show()
