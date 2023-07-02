"""
Transition Region Explorer (TREx) is a CFI-funded project and aims to be the world's foremost auroral imaging facility for remote sensing the near-Earth space environment. The project consists of the development and deployment of an extensive ground-based network of sophisticated optical and radio instrumentation across Alberta, Saskatchewan, Manitoba, and the Northwest Territories.

The TREx project will design and deploy the following instruments:
- 6 blue-line all-sky imagers (3s-30Hz cadence)
- 6 RGB colour all-sky imagers (3s cadence)
- 6 Near-Infrared all-sky imagers (6s cadence)
- 10 Imaging Riometers (1s-100Hz cadence - operations project is GO-IRIS)
- 2 Proton Aurora Meridian Imaging Spectographs (15s cadence)
- 13 Global Navigation Satellite System receiver systems (GNSS)

In partnership with IBM, TREx will include sensor web technology to autonomously control and coordinate sensor behaviour across the observational region. This architecture will allow TREx to produce the first high resolution data (estimated at 120 TB of data per year) over a region large enough to study multi-scale coupling of key physical processes in the space environment. 
"""

from datetime import datetime, timedelta
import re
import warnings
import pathlib
import copy
import os
import dateutil.parser

import numpy as np
import pandas as pd
import scipy.io

import asilib
from asilib.asi.themis import _get_pgm_files
import asilib.utils as utils
import asilib.io.download as download


pgm_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/NIR/stream0/'
skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/NIR/skymaps/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'trex'


def trex_nir(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    redownload: bool = False,
    missing_ok: bool = True,
    load_images: bool = True,
    imager=asilib.Imager,
) -> asilib.Imager:
    """
    Create an Imager instance using the TREX-NIR ASI images and skymaps.

    Parameters
    ----------
    location_code: str
        The ASI's location code (four letters).
    time: str or datetime.datetime
        A time to look for the ASI data at. Either time or time_range
        must be specified (not both or neither).
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    alt: int
        The reference skymap altitude, in kilometers.
    redownload: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).
    load_images: bool
        Create an Imager object without images. This is useful if you need to
        calculate conjunctions and don't need to download or load unnecessary data.
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        The THEMIS Imager instance.

    Examples
    --------
    >>> import asilib
    >>> import asilib.map
    >>> import asilib.asi
    >>> import matplotlib.pyplot as plt
    >>>
    >>> fig = plt.figure(figsize=(10, 6))
    >>> ax = fig.add_subplot(121)
    >>> bx = asilib.map.create_map(fig_ax=(fig, 122), lon_bounds=(-102, -86), lat_bounds=(51, 61))
    >>> asi = asilib.asi.trex_nir('gill', time='2020-03-21T06:00')
    >>> asi.plot_fisheye(ax=ax)
    >>> asi.plot_map(ax=bx)
    >>> plt.tight_layout()
    >>> plt.show()

    >>> import asilib
    >>> import asilib.map
    >>> import asilib.asi
    >>> import matplotlib.pyplot as plt
    >>>
    >>> time_range = ('2020-03-21T05:00', '2020-03-21T07:00')
    >>> fig, ax = plt.subplots(2, sharex=True)
    >>> asi = asilib.asi.trex_nir('gill', time_range=time_range)
    >>> asi.plot_keogram(ax=ax[0])
    >>> asi.plot_keogram(ax=ax[1], aacgm=True)
    >>> ax[0].set_title(f'TREX_NIR GILL keogram | {time_range}')
    >>> ax[0].set_ylabel('Geo Lat')
    >>> ax[1].set_ylabel('Mag Lat')
    >>> ax[1].set_xlabel('Time')
    >>> plt.show()
    """
    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_pgm_dir = local_base_dir / 'nir' / 'images' / location_code.lower()

    if load_images:
        # Download and find image data
        file_paths = _get_pgm_files(
            'nir',
            location_code,
            time,
            time_range,
            pgm_base_url,
            local_pgm_dir,
            redownload,
            missing_ok,
        )

        start_times = len(file_paths) * [None]
        end_times = len(file_paths) * [None]
        for i, file_path in enumerate(file_paths):
            date_match = re.search(r'\d{8}_\d{4}', file_path.name)
            start_times[i] = datetime.strptime(date_match.group(), '%Y%m%d_%H%M')
            end_times[i] = start_times[i] + timedelta(minutes=1)
        file_info = {
            'path': file_paths,
            'start_time': start_times,
            'end_time': end_times,
            'loader': _load_nir_pgm,
        }
    else:
        file_info = {
            'path': [],
            'start_time': [],
            'end_time': [],
            'loader': None,
        }

    if time_range is not None:
        file_info['time_range'] = time_range
    else:
        file_info['time'] = time

    # Download and find the appropriate skymap
    if time is not None:
        _time = time
    else:
        _time = time_range[0]
    _skymap = trex_nir_skymap(location_code, _time, redownload=redownload)
    alt_index = np.where(_skymap['FULL_MAP_ALTITUDE'] / 1000 == alt)[0]
    assert (
        len(alt_index) == 1
    ), f'{alt} km is not in the valid skymap altitudes: {_skymap["FULL_MAP_ALTITUDE"]/1000} km.'
    alt_index = alt_index[0]
    skymap = {
        'lat': _skymap['FULL_MAP_LATITUDE'][alt_index, :, :],
        'lon': _skymap['FULL_MAP_LONGITUDE'][alt_index, :, :],
        'alt': _skymap['FULL_MAP_ALTITUDE'][alt_index] / 1e3,
        'el': _skymap['FULL_ELEVATION'],
        'az': _skymap['FULL_AZIMUTH'],
        'path': _skymap['PATH'],
    }

    meta = {
        'array': 'TREX_NIR',
        'location': location_code.upper(),
        'lat': float(_skymap['SITE_MAP_LATITUDE']),
        'lon': float(_skymap['SITE_MAP_LONGITUDE']),
        'alt': float(_skymap['SITE_MAP_ALTITUDE']) / 1e3,
        'cadence': 6,
        'resolution': (256, 256),
    }
    return imager(file_info, meta, skymap)


def trex_nir_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the TREx NIR ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of REGO imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'TREx_NIR']
    return df.reset_index(drop=True)


def trex_nir_skymap(location_code: str, time: utils._time_type, redownload: bool = False) -> dict:
    """
    Load a TREx NIR ASI skymap file.

    Parameters
    ----------
    location_code: str
        The four character location name.
    time: str or datetime.datetime
        A ISO-fomatted time string or datetime object. Must be in UT time.
    redownload: bool
        Redownload the file.

    Returns
    -------
    dict:
        The skymap.
    """
    time = utils.validate_time(time)
    local_dir = local_base_dir / 'skymaps' / location_code.lower()
    local_dir.mkdir(parents=True, exist_ok=True)
    skymap_top_url = skymap_base_url + location_code.lower() + '/'

    if redownload:
        # Delete any existing skymap files.
        local_skymap_paths = pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
        for local_skymap_path in local_skymap_paths:
            os.unlink(local_skymap_path)
        local_skymap_paths = _download_all_skymaps(
            location_code, skymap_top_url, local_dir, redownload=redownload
        )

    else:
        local_skymap_paths = sorted(
            pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
        )
        # TODO: Add a check to periodically redownload the skymap data, maybe once a month?
        if len(local_skymap_paths) == 0:
            local_skymap_paths = _download_all_skymaps(
                location_code, skymap_top_url, local_dir, redownload=redownload
            )

    skymap_filenames = [local_skymap_path.name for local_skymap_path in local_skymap_paths]
    skymap_file_dates = []
    for skymap_filename in skymap_filenames:
        date_match = re.search(r'\d{8}', skymap_filename)
        skymap_file_dates.append(datetime.strptime(date_match.group(), '%Y%m%d'))

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date.
    dt = np.array([(time - skymap_date).total_seconds() for skymap_date in skymap_file_dates])
    dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    if np.all(~np.isfinite(dt)):
        # Edge case when time is before the first skymap_date.
        closest_index = 0
        warnings.warn(
            f'The requested skymap time={time} for TREX NIR-{location_code.upper()} is before first '
            f'skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = local_skymap_paths[closest_index]
    skymap = _load_skymap(skymap_path)
    return skymap


def _download_all_skymaps(location_code, url, save_dir, redownload):
    d = download.Downloader(url)
    # Find the dated subdirectories
    ds = d.ls(f'{location_code.lower()}')

    save_paths = []
    for d_i in ds:
        ds = d_i.ls(f'*skymap_{location_code.lower()}*.sav')
        for ds_j in ds:
            save_paths.append(ds_j.download(save_dir, redownload=redownload))
    return save_paths


def _load_skymap(skymap_path):
    """
    A helper function to load a TREX NIR skymap and transform it.
    """
    # Load the skymap file and convert it to a dictionary.
    skymap_file = scipy.io.readsav(str(skymap_path), python_dict=True)['skymap']
    skymap_dict = {key: copy.copy(skymap_file[key][0]) for key in skymap_file.dtype.names}

    skymap_dict = _tranform_longitude_to_180(skymap_dict)
    skymap_dict = _flip_skymap(skymap_dict)
    skymap_dict['PATH'] = skymap_path
    return skymap_dict


def _flip_skymap(skymap):
    """
    IDL is a column-major language while Python is row-major. This function
    tranposes the 2- and 3-D arrays to make them compatable with the images
    that are saved in row-major.
    """
    for key in skymap:
        if hasattr(skymap[key], 'shape'):
            shape = skymap[key].shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                skymap[key] = skymap[key][::-1, ::-1]  # For Az/El maps.
            elif (len(shape) == 3) and (shape[1] == shape[2]):
                skymap[key] = skymap[key][:, ::-1, ::-1]  # For lat/lon maps
    return skymap


def _tranform_longitude_to_180(skymap):
    """
    Transform the SITE_MAP_LONGITUDE and FULL_MAP_LONGITUDE arrays from
    (0 -> 360) to (-180 -> 180).
    """
    skymap['SITE_MAP_LONGITUDE'] = np.mod(skymap['SITE_MAP_LONGITUDE'] + 180, 360) - 180

    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(skymap['FULL_MAP_LONGITUDE']))
    skymap['FULL_MAP_LONGITUDE'][valid_val_idx] = (
        np.mod(skymap['FULL_MAP_LONGITUDE'][valid_val_idx] + 180, 360) - 180
    )
    return skymap


def _load_nir_pgm(path):
    # TODO: Remove the import once opencv is updated and trex_imager_readfile can be
    # updated https://github.com/opencv/opencv/issues/23059
    import trex_imager_readfile

    images, meta, problematic_file_list = trex_imager_readfile.read_nir(str(path))
    if len(problematic_file_list):
        raise ValueError(f'A problematic PGM file: {problematic_file_list[0]}')
    images = np.moveaxis(images, 2, 0)
    images = images[:, ::-1, ::-1]  # Flip north-south.
    times = np.array(
        [
            dateutil.parser.parse(dict_i['Image request start']).replace(tzinfo=None)
            for dict_i in meta
        ]
    )
    return times, images
