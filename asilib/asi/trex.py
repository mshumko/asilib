"""
Transition Region Explorer (TREx) data is courtesy of Space Environment Canada (space-environment.ca). Use of the data must adhere to the rules of the road for that dataset.  Please see below for the required data acknowledgement. Any questions about the TREx instrumentation or data should be directed to the University of Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan (edonovan@ucalgary.ca).

“The Transition Region Explorer RGB (TREx RGB) is a joint Canada Foundation for Innovation and Canadian Space Agency project developed by the University of Calgary. TREx-RGB is operated and maintained by Space Environment Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”

For more information see: https://www.ucalgary.ca/aurora/projects/trex
"""
from typing import Iterable, List, Union
from datetime import datetime, timedelta
from multiprocessing import Pool
import functools
import re
import warnings
import pathlib
import copy
import os
import dateutil.parser
import gzip
import shutil
import signal
import tarfile
import random
import string
import cv2
import h5py

import numpy as np
import pandas as pd
import scipy.io
import requests

import asilib
from asilib.asi.themis import _get_pgm_files
import asilib.utils as utils
import asilib.download as download
import asilib.skymap


nir_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/NIR/stream0/'
nir_skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/NIR/skymaps/'
rgb_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/RGB/stream0/'
rgb_skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/RGB/skymaps/'
blueline_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/blueline/stream0/'
blueline_skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/TREx/blueline/skymaps/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'trex'


def trex_rgb(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    custom_alt: bool = False,
    redownload: bool = False,
    missing_ok: bool = True,
    load_images: bool = True,
    colors: str = 'rgb',
    burst: bool = False,
    acknowledge: bool = True,
    imager=asilib.Imager,
) -> asilib.imager.Imager:
    """
    Create an Imager instance using the TREX-RGB ASI images and skymaps.

    Transition Region Explorer (TREx) RGB data is courtesy of Space Environment Canada (space-environment.ca). Use of the data must adhere to the rules of the road for that dataset.  Please see below for the required data acknowledgement. Any questions about the TREx instrumentation or data should be directed to the University of Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan (edonovan@ucalgary.ca).

    “The Transition Region Explorer RGB (TREx RGB) is a joint Canada Foundation for Innovation and Canadian Space Agency project developed by the University of Calgary. TREx-RGB is operated and maintained by Space Environment Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”

    For more information see: https://www.ucalgary.ca/aurora/projects/trex.

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
    custom_alt: bool
        If True, asilib will calculate (lat, lon) skymaps assuming a spherical Earth. Otherwise, it will use the official skymaps (Courtesy of University of Calgary).

        .. note::
        
            The spherical model of Earth's surface is less accurate than the oblate spheroid geometrical representation. Therefore, there will be a small difference between these and the official skymaps.
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
    colors: str
        Load all three color channels if "rgb", or individual color channels specified
        by "r", "g", "b" (or any combination of them).
    burst: bool
        Sometimes Trex-rgb uses a burst mode with higher resolution.
    acknowledge: bool
        If True, prints the acknowledgment statement for TREx-RGB.
    imager: :py:meth:`~asilib.imager.Imager`
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        The trex Imager instance.

    Examples
    --------
    >>> from datetime import datetime
    >>> 
    >>> import matplotlib.pyplot as plt
    >>> import asilib.map
    >>> import asilib
    >>> from asilib.asi import trex_rgb
    >>> 
    >>> time = datetime(2021, 11, 4, 7, 3, 51)
    >>> location_codes = ['FSMI', 'LUCK', 'RABB', 'PINA', 'GILL']
    >>> asi_list = []
    >>> ax = asilib.map.create_simple_map()
    >>> for location_code in location_codes:
    >>>     asi_list.append(trex_rgb(location_code, time=time, colors='rgb'))
    >>> 
    >>> asis = asilib.Imagers(asi_list)
    >>> asis.plot_map(ax=ax)
    >>> ax.set(title=time)
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if burst == True:
        raise NotImplementedError(
            'Burst mode still needs implementation as it is a different file format')
    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_rgb_dir = local_base_dir / 'rgb' / 'images' / location_code.lower()

    if load_images:
        # Download and find image data
        file_paths = _get_h5_files(
            'rgb',
            location_code,
            time,
            time_range,
            rgb_base_url,
            local_rgb_dir,
            redownload,
            missing_ok,
        )

        start_times = len(file_paths) * [None]
        end_times = len(file_paths) * [None]
        for i, file_path in enumerate(file_paths):
            date_match = re.search(r'\d{8}_\d{4}', file_path.name)
            start_times[i] = datetime.strptime(
                date_match.group(), '%Y%m%d_%H%M')
            end_times[i] = start_times[i] + timedelta(minutes=1)
        file_info = {
            'path': file_paths,
            'start_time': start_times,
            'end_time': end_times,
            'loader': lambda path: _load_rgb_h5(path),
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
    _skymap = trex_rgb_skymap(location_code, _time, redownload=redownload)
    if custom_alt==False:
        alt_index = np.where(_skymap['FULL_MAP_ALTITUDE'] / 1000 == alt)[0]
        assert (
            len(alt_index) == 1
        ), f'{alt} km is not in the valid skymap altitudes: {_skymap["FULL_MAP_ALTITUDE"]/1000} km. If you want a custom altitude with less percision, please use the custom_alt keyword'
        alt_index = alt_index[0]
        lat=_skymap['FULL_MAP_LATITUDE'][alt_index, :, :]
        lon=_skymap['FULL_MAP_LONGITUDE'][alt_index, :, :]
    else:
        lat,lon = asilib.skymap.geodetic_skymap(
            (float(_skymap['SITE_MAP_LATITUDE']), float(_skymap['SITE_MAP_LONGITUDE']), float(_skymap['SITE_MAP_ALTITUDE']) / 1e3),
            _skymap['FULL_AZIMUTH'],
            _skymap['FULL_ELEVATION'],
            alt
            )

    skymap = {
        'lat': lat,
        'lon': lon,
        'alt': alt,
        'el': _skymap['FULL_ELEVATION'],
        'az': _skymap['FULL_AZIMUTH'],
        'path': _skymap['PATH'],
    }

    meta = {
        'array': 'TREX_RGB',
        'location': location_code.upper(),
        'lat': float(_skymap['SITE_MAP_LATITUDE']),
        'lon': float(_skymap['SITE_MAP_LONGITUDE']),
        'alt': float(_skymap['SITE_MAP_ALTITUDE']) / 1e3,
        'cadence': 3,
        'resolution': (480, 553, 3),
        'colors': colors,
        'acknowledgment':(
            'Transition Region Explorer (TREx) RGB data is courtesy of Space Environment Canada '
            '(space-environment.ca). Use of the data must adhere to the rules of the road for '
            'that dataset.  Please see below for the required data acknowledgement. Any questions '
            'about the TREx instrumentation or data should be directed to the University of '
            'Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan '
            '(edonovan@ucalgary.ca).\n\n“The Transition Region Explorer RGB (TREx RGB) is a joint '
            'Canada Foundation for Innovation and Canadian Space Agency project developed by the '
            'University of Calgary. TREx-RGB is operated and maintained by Space Environment '
            'Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”'
        )
    }
    plot_settings = {
        'color_norm':'lin',  # Needed for to be compatible with plt.pcolormesh.
        'color_bounds':(15, 80)
        }

    if acknowledge and ('trex_rgb' not in asilib.config['ACKNOWLEDGED_ASIS']):
        print(meta['acknowledgment'])
        asilib.config['ACKNOWLEDGED_ASIS'].append('trex_rgb')
    return imager(file_info, meta, skymap, plot_settings=plot_settings)


def _get_h5_files(
    array: str,
    location_code: str,
    time: datetime,
    time_range: Iterable[datetime],
    base_url: str,
    save_dir: Union[str, pathlib.Path],
    redownload: bool,
    missing_ok: bool,
) -> List[pathlib.Path]:
    """
    Look for and download 1-minute h5 files.

    Parameters
    ----------
    array: str
        The ASI array name.
    location_code:str
        The THEMIS location code.
    time: datetime.datetime
        Time to download one file. Either time or time_range must be specified,
        but not both.
    time_range: Iterable[datetime]
        An iterable with a start and end time. Either time or time_range must be
        specified, but not both.
    base_url: str
        The starting URL to search for file.
    save_dir: str or pathlib.Path
        The parent directory where to save the data to.
    redownload: bool
        Download data even if the file is found locally. This is useful if data
        is corrupt.
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).

    Returns
    -------
    list(pathlib.Path)
        Local paths to each PGM file that was successfully downloaded.
    """

    if (time is None) and (time_range is None):
        raise ValueError('time or time_range must be specified.')
    elif (time is not None) and (time_range is not None):
        raise ValueError(
            'both time and time_range can not be simultaneously specified.')

    if redownload:
        # Option 1/4: Download one minute of data regardless if it is already saved
        if time is not None:
            return [
                _download_one_h5_file(
                    array, location_code, time, base_url, save_dir, redownload)
            ]

            # Option 2/4: Download the data in time range regardless if it is already saved.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='minutes')
            file_paths = []
            for file_time in file_times:
                try:
                    file_paths.append(
                        _download_one_h5_file(
                            array, location_code, file_time, base_url, save_dir, redownload
                        )
                    )
                except (FileNotFoundError, AssertionError, requests.exceptions.HTTPError) as err:
                    if missing_ok and (
                        ('does not contain any hyper references containing' in str(err)) or
                        ('Only one href is allowed' in str(err)) or
                        ('404 Client Error: Not Found for url:' in str(err))
                    ):
                        continue
                    raise
            return file_paths
    else:
        # Option 3/4: Download one minute of data if it is not already saved.
        if time is not None:
            file_search_str = f'{time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}*.h5'
            local_file_paths = list(pathlib.Path(
                save_dir).rglob(file_search_str))
            if len(local_file_paths) == 1:
                return local_file_paths
            else:
                return [
                    _download_one_h5_file(
                        array, location_code, time, base_url, save_dir, redownload
                    )
                ]

        # Option 4/4: Download the data in time range if they don't exist locally.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='minutes')
            file_paths = []
            for file_time in file_times:
                file_search_str = (
                    f'{file_time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}*.h5'
                )
                local_file_paths = list(pathlib.Path(
                    save_dir).rglob(file_search_str))
                if len(local_file_paths) == 1:
                    file_paths.append(local_file_paths[0])
                else:
                    try:
                        file_paths.append(
                            _download_one_h5_file(
                                array, location_code, file_time, base_url, save_dir, redownload
                            )
                        )
                    except (FileNotFoundError, AssertionError, requests.exceptions.HTTPError) as err:
                        if missing_ok and (
                            ('does not contain any hyper references containing' in str(err)) or
                            ('Only one href is allowed' in str(err)) or
                            ('404 Client Error: Not Found for url:' in str(err))
                        ):
                            continue
                        raise
            return file_paths


def _download_one_h5_file(
    array: str,
    location_code: str,
    time: datetime,
    base_url: str,
    save_dir: Union[str, pathlib.Path],
    redownload: bool,
) -> pathlib.Path:
    """
    Download one PGM file.

    Parameters
    ----------
    array: str
        The ASI array name.
    location_code: str
        The ASI four-letter location code.
    time: str or datetime.datetime
        A time to look for the ASI data at.
    base_url: str
        The starting URL to search for file.
    save_dir: str or pathlib.Path
        The parent directory where to save the data to.
    redownload: bool
        Will redownload an existing file.

    Returns
    -------
    pathlib.Path
        Local path to file.
    """
    start_url = base_url + f'{time.year}/{time.month:02}/{time.day:02}/'
    d = download.Downloader(start_url, headers={'User-Agent':'asilib'})
    # Find the unique directory
    matched_downloaders = d.ls(f'{location_code.lower()}_{array}*')
    assert len(matched_downloaders) == 1
    # Search that directory for the file and donload it.
    d2 = download.Downloader(
        matched_downloaders[0].url + f'ut{time.hour:02}/', 
        headers={'User-Agent':'asilib'}
        )
    file_search_str = f'{time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}*{array}*.h5'
    matched_downloaders2 = d2.ls(file_search_str)
    assert len(matched_downloaders2) == 1
    return matched_downloaders2[0].download(save_dir, redownload=redownload)


def _load_rgb_h5(path):
    images, meta, problematic_file_list = read_rgb(
        str(path))
    if len(problematic_file_list):
        raise ValueError(f'A problematic PGM file: {problematic_file_list[0]}')
    images = np.moveaxis(images, 3, 0)  # Move time to first axis.
    images = images[:, ::-1, ::-1, :]  # Flip north-south.
    
    times = np.array(
        [
            dateutil.parser.parse(
                dict_i['image_request_start_timestamp']).replace(tzinfo=None)
            for dict_i in meta
        ]
    )
    return times, images


def trex_rgb_skymap(location_code: str, time: utils._time_type, redownload: bool = False) -> dict:
    """
    Load a TREx RGB skymap file.

    Parameters
    ----------
    location_code: str
        The four character location name.
    time: str or datetime.datetime
        A ISO-formatted time string or datetime object. Must be in UT time.
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
    skymap_top_url = rgb_skymap_base_url + location_code.lower() + '/'
    if redownload:
        # Delete any existing skymap files.
        local_skymap_paths = pathlib.Path(local_dir).rglob(
            f'*rgb_skymap_{location_code.lower()}*.sav')
        for local_skymap_path in local_skymap_paths:
            os.unlink(local_skymap_path)
        local_skymap_paths = _download_all_skymaps(
            location_code, skymap_top_url, local_dir, redownload=redownload
        )

    else:
        local_skymap_paths = sorted(
            pathlib.Path(local_dir).rglob(
                f'rgb_skymap_{location_code.lower()}*.sav')
        )
        # TODO: Add a check to periodically redownload the skymap data, maybe once a month?
        if len(local_skymap_paths) == 0:
            local_skymap_paths = _download_all_skymaps(
                location_code, skymap_top_url, local_dir, redownload=redownload
            )
    skymap_filenames = [
        local_skymap_path.name for local_skymap_path in local_skymap_paths]
    skymap_file_dates = []
    for skymap_filename in skymap_filenames:
        date_match = re.search(r'\d{8}', skymap_filename)
        skymap_file_dates.append(
            datetime.strptime(date_match.group(), '%Y%m%d'))

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date.
    dt = np.array([(time - skymap_date).total_seconds()
                   for skymap_date in skymap_file_dates])
    dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    if np.all(~np.isfinite(dt)):
        # Edge case when time is before the first skymap_date.
        closest_index = 0
        warnings.warn(
            f'The requested skymap time={time} for TREX rgb-{location_code.upper()} is before first '
            f'skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = local_skymap_paths[closest_index]
    skymap = _load_rgb_skymap(skymap_path)
    return skymap


def trex_rgb_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the TREx RGB ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of TREx-RGB imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'TREx_RGB']
    return df.reset_index(drop=True)


def trex_nir(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    custom_alt: bool = False,
    redownload: bool = False,
    missing_ok: bool = True,
    load_images: bool = True,
    acknowledge: bool = True,
    imager=asilib.Imager,
) -> asilib.Imager:
    """
    Create an Imager instance using the TREX-NIR ASI images and skymaps.

    Transition Region Explorer (TREx) NIR data is courtesy of Space Environment Canada (space-environment.ca). Use of the data must adhere to the rules of the road for that dataset.  Please see below for the required data acknowledgement. Any questions about the TREx instrumentation or data should be directed to the University of Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan (edonovan@ucalgary.ca).

    “The Transition Region Explorer NIR (TREx NIR) is a joint Canada Foundation for Innovation and Canadian Space Agency project developed by the University of Calgary. TREx-NIR is operated and maintained by Space Environment Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”

    For more information see: https://www.ucalgary.ca/aurora/projects/trex.

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
    custom_alt: bool
        If True, asilib will calculate (lat, lon) skymaps assuming a spherical Earth. Otherwise, it will use the official skymaps (Courtesy of University of Calgary).

        .. note::
        
            The spherical model of Earth's surface is less accurate than the oblate spheroid geometrical representation. Therefore, there will be a small difference between these and the official skymaps.
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
    acknowledge: bool
        If True, prints the acknowledgment statement for TREx-NIR.
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
            nir_base_url,
            local_pgm_dir,
            redownload,
            missing_ok,
        )

        start_times = len(file_paths) * [None]
        end_times = len(file_paths) * [None]
        for i, file_path in enumerate(file_paths):
            date_match = re.search(r'\d{8}_\d{4}', file_path.name)
            start_times[i] = datetime.strptime(
                date_match.group(), '%Y%m%d_%H%M')
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
    
    if custom_alt==False:
        alt_index = np.where(_skymap['FULL_MAP_ALTITUDE'] / 1000 == alt)[0] #Compares the altitudes versus the ones provided by default and chooses the correct index that correlates to the chosen alitudes
        assert (
            len(alt_index) == 1
        ), f'{alt} km is not in the valid skymap altitudes: {_skymap["FULL_MAP_ALTITUDE"]/1000} km. If you want a custom altitude with less percision, please use the custom_alt keyword'
        alt_index = alt_index[0]
        lat=_skymap['FULL_MAP_LATITUDE'][alt_index, :, :] #selects lat lon coordinates from data provided in skymap
        lon=_skymap['FULL_MAP_LONGITUDE'][alt_index, :, :]
    else:
        lat,lon = asilib.skymap.geodetic_skymap( #Spherical projection for lat lon coordinates
            (float(_skymap['SITE_MAP_LATITUDE']), float(_skymap['SITE_MAP_LONGITUDE']), float(_skymap['SITE_MAP_ALTITUDE']) / 1e3),
            _skymap['FULL_AZIMUTH'],
            _skymap['FULL_ELEVATION'],
            alt
            )

    skymap = {
        'lat': lat,
        'lon': lon,
        'alt': alt,
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
        'acknowledgment':(
            'Transition Region Explorer (TREx) NIR data is courtesy of Space Environment Canada '
            '(space-environment.ca). Use of the data must adhere to the rules of the road for '
            'that dataset.  Please see below for the required data acknowledgement. Any questions '
            'about the TREx instrumentation or data should be directed to the University of '
            'Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan '
            '(edonovan@ucalgary.ca).\n\n“The Transition Region Explorer NIR (TREx NIR) is a joint '
            'Canada Foundation for Innovation and Canadian Space Agency project developed by the '
            'University of Calgary. TREx-NIR is operated and maintained by Space Environment '
            'Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”'
            )
    }

    plot_settings = {
        'color_bounds':(350, 1500)
        }

    if acknowledge and ('trex_nir' not in asilib.config['ACKNOWLEDGED_ASIS']):
        print(meta['acknowledgment'])
        asilib.config['ACKNOWLEDGED_ASIS'].append('trex_nir')
    return imager(file_info, meta, skymap, plot_settings=plot_settings)


def trex_nir_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the TREx NIR ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of TREx-RGB imager names and locations.
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
        A ISO-formatted time string or datetime object. Must be in UT time.
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
    skymap_top_url = nir_skymap_base_url + location_code.lower() + '/'

    if redownload:
        # Delete any existing skymap files.
        local_skymap_paths = pathlib.Path(local_dir).rglob(
            f'nir_skymap_{location_code.lower()}*.sav')
        for local_skymap_path in local_skymap_paths:
            os.unlink(local_skymap_path)
        local_skymap_paths = _download_all_skymaps(
            location_code, skymap_top_url, local_dir, redownload=redownload
        )

    else:
        local_skymap_paths = sorted(
            pathlib.Path(local_dir).rglob(
                f'nir_skymap_{location_code.lower()}*.sav')
        )
        # TODO: Add a check to periodically redownload the skymap data, maybe once a month?
        if len(local_skymap_paths) == 0:
            local_skymap_paths = _download_all_skymaps(
                location_code, skymap_top_url, local_dir, redownload=redownload
            )

    skymap_filenames = [
        local_skymap_path.name for local_skymap_path in local_skymap_paths]
    skymap_file_dates = []
    for skymap_filename in skymap_filenames:
        date_match = re.search(r'\d{8}', skymap_filename)
        skymap_file_dates.append(
            datetime.strptime(date_match.group(), '%Y%m%d'))

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date.
    dt = np.array([(time - skymap_date).total_seconds()
                   for skymap_date in skymap_file_dates])
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
    skymap = _load_nir_skymap(skymap_path)
    return skymap


def _download_all_skymaps(location_code, url, save_dir, redownload):
    d = download.Downloader(url, headers={'User-Agent':'asilib'})
    # Find the dated subdirectories
    ds = d.ls(f'{location_code.lower()}')

    save_paths = []
    for d_i in ds:
        ds = d_i.ls(f'*skymap_{location_code.lower()}*.sav')
        for ds_j in ds:
            save_paths.append(ds_j.download(save_dir, redownload=redownload))
    return save_paths


def _load_nir_skymap(skymap_path):
    """
    A helper function to load a TREx NIR skymap and transform it.
    """
    # Load the skymap file and convert it to a dictionary.
    skymap_file = scipy.io.readsav(
        str(skymap_path), python_dict=True)['skymap']
    skymap_dict = {key: copy.copy(
        skymap_file[key][0]) for key in skymap_file.dtype.names}

    skymap_dict = _tranform_longitude_to_180(skymap_dict)
    skymap_dict = _flip_skymap(skymap_dict)
    skymap_dict['PATH'] = skymap_path
    return skymap_dict

def _load_rgb_skymap(skymap_path):
    """
    A helper function to load a TREx RGB skymap and transform it.
    """
    # Load the skymap file and convert it to a dictionary.
    skymap_file = scipy.io.readsav(
        str(skymap_path), python_dict=True)['skymap']
    skymap_dict = {key: copy.copy(
        skymap_file[key][0]) for key in skymap_file.dtype.names}

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
            if len(shape) == 2:
                skymap[key] = skymap[key][::-1, ::-1]  # For Az/El maps.
            elif len(shape) == 3:
                skymap[key] = skymap[key][:, ::-1, ::-1]  # For lat/lon maps
    return skymap


def _tranform_longitude_to_180(skymap):
    """
    Transform the SITE_MAP_LONGITUDE and FULL_MAP_LONGITUDE arrays from
    (0 -> 360) to (-180 -> 180).
    """
    skymap['SITE_MAP_LONGITUDE'] = np.mod(
        skymap['SITE_MAP_LONGITUDE'] + 180, 360) - 180

    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(skymap['FULL_MAP_LONGITUDE']))
    skymap['FULL_MAP_LONGITUDE'][valid_val_idx] = (
        np.mod(skymap['FULL_MAP_LONGITUDE'][valid_val_idx] + 180, 360) - 180
    )
    return skymap


def _load_nir_pgm(path):
    images, meta, problematic_file_list = read_nir(
        str(path))
    if len(problematic_file_list):
        raise ValueError(f'A problematic PGM file: {problematic_file_list[0]}')
    images = np.moveaxis(images, 2, 0)
    images = images[:, ::-1, ::-1]  # Flip north-south.
    times = np.array(
        [
            dateutil.parser.parse(
                dict_i['Image request start']).replace(tzinfo=None)
            for dict_i in meta
        ]
    )
    return times, images


"""
The data readers below was developed by the University of Calgary under the 
MIT License. You can find the original source files at: 
https://github.com/ucalgary-srs/trex-imager-readfile.

MIT License

Copyright (c) 2020-present University of Calgary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# static globals
__RGB_PGM_EXPECTED_HEIGHT = 480
__RGB_PGM_EXPECTED_WIDTH = 553
__RGB_PGM_DT = np.dtype("uint16")
__RGB_PGM_DT = __RGB_PGM_DT.newbyteorder('>')  # force big endian byte ordering
__RGB_PNG_DT = np.dtype("uint8")
__RGB_H5_DT = np.dtype("uint8")
__PNG_METADATA_PROJECT_UID = "trex"


def __trex_readfile_worker(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = np.dtype("uint8")

    # check file extension to know how to process
    try:
        if (file_obj["filename"].endswith("pgm") or file_obj["filename"].endswith("pgm.gz")):
            return __rgb_readfile_worker_pgm(file_obj)
        elif (file_obj["filename"].endswith("png") or file_obj["filename"].endswith("png.tar")):
            return __rgb_readfile_worker_png(file_obj)
        elif (file_obj["filename"].endswith("h5")):
            return __rgb_readfile_worker_h5(file_obj)
        else:
            if (file_obj["quiet"] is False):
                print("Unrecognized file type: %s" % (file_obj["filename"]))
            problematic = True
            error_message = "Unrecognized file type"
    except Exception as e:
        if (file_obj["quiet"] is False):
            print("Failed to process file '%s' " % (file_obj["filename"]))
        problematic = True
        error_message = "failed to process file: %s" % (str(e))
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype


def __rgb_readfile_worker_h5(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = __RGB_H5_DT

    # open H5 file
    f = h5py.File(file_obj["filename"], 'r')

    # get images and timestamps
    if (file_obj["first_frame"] is True):
        # get only first frame
        images = f["data"]["images"][:, :, :, 0]
        timestamps = [f["data"]["timestamp"][0]]
    else:
        # get all frames
        images = f["data"]["images"][:]
        timestamps = f["data"]["timestamp"][:]

    # read metadata
    file_metadata = {}
    if (file_obj["no_metadata"] is True):
        metadata_dict_list = [{}] * len(timestamps)
    else:
        # get file metadata
        for key, value in f["metadata"]["file"].attrs.items():
            file_metadata[key] = value

        # read frame metadata
        for i in range(0, len(timestamps)):
            this_frame_metadata = file_metadata.copy()
            for key, value in f["metadata"]["frame"]["frame%d" % (i)].attrs.items():
                this_frame_metadata[key] = value
            metadata_dict_list.append(this_frame_metadata)

    # close H5 file
    f.close()

    # set image vars and reshape if multiple images
    image_height = images.shape[0]
    image_width = images.shape[1]
    image_channels = images.shape[2]
    if (len(images.shape) == 3):
        # force reshape to 4 dimensions
        images = images.reshape((image_height, image_width, image_channels, 1))

    # return
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype


def __rgb_readfile_worker_png(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    is_first = True
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = __RGB_PNG_DT
    working_dir_created = False

    # set up working dir
    this_working_dir = "%s/%s" % (file_obj["tar_tempdir"], ''.join(random.choices(string.ascii_lowercase, k=8)))

    # check if it's a tar file
    file_list = []
    if (file_obj["filename"].endswith(".png.tar")):
        # tar file, extract all frames and add to list
        try:
            tf = tarfile.open(file_obj["filename"])
            file_list = sorted(tf.getnames())
            if (file_obj["first_frame"] is True):
                file_list = [file_list[0]]
                tf.extract(file_list[0], path=this_working_dir)
            else:
                tf.extractall(path=this_working_dir)
            for i in range(0, len(file_list)):
                file_list[i] = "%s/%s" % (this_working_dir, file_list[i])
            tf.close()
            working_dir_created = True
        except Exception as e:
            # cleanup
            try:
                shutil.rmtree(this_working_dir)
            except Exception:
                pass

            # set error message
            if (file_obj["quiet"] is False):
                print("Failed to open file '%s' " % (file_obj["filename"]))
            problematic = True
            error_message = "failed to open file: %s" % (str(e))
            try:
                tf.close()
            except Exception:
                pass
            return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
                image_width, image_height, image_channels, image_dtype
    else:
        # regular png
        file_list = [file_obj["filename"]]

    # read each png file
    for f in file_list:
        if (file_obj["no_metadata"] is True):
            metadata_dict_list.append({})
        else:
            # process metadata
            try:
                # set metadata values
                file_split = os.path.basename(f).split('_')
                site_uid = file_split[3]
                device_uid = file_split[4]
                exposure = "%.03f ms" % (float(file_split[5][:-2]))
                mode_uid = file_split[6][:-4]

                # set timestamp
                if ("burst" in f or "mode-b"):
                    timestamp = datetime.strptime("%sT%s.%s" % (file_split[0], file_split[1], file_split[2]), "%Y%m%dT%H%M%S.%f")
                else:
                    timestamp = datetime.strptime("%sT%s" % (file_split[0], file_split[1]), "%Y%m%dT%H%M%S")

                # set the metadata dict
                metadata_dict = {
                    "Project unique ID": __PNG_METADATA_PROJECT_UID,
                    "Site unique ID": site_uid,
                    "Imager unique ID": device_uid,
                    "Mode unique ID": mode_uid,
                    "Image request start": timestamp,
                    "Subframe requested exposure": exposure,
                }
                metadata_dict_list.append(metadata_dict)
            except Exception as e:
                if (file_obj["quiet"] is False):
                    print("Failed to read metadata from file '%s' " % (f))
                problematic = True
                error_message = "failed to read metadata: %s" % (str(e))
                break

        # read png file
        try:
            # read file
            image_np = cv2.imread(f, cv2.IMREAD_COLOR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            image_height = image_np.shape[0]
            image_width = image_np.shape[1]
            image_channels = image_np.shape[2] if len(image_np.shape) > 2 else 1
            if (image_channels > 1):
                image_matrix = np.reshape(image_np, (image_height, image_width, image_channels, 1))
            else:
                image_matrix = np.reshape(image_np, (image_height, image_width, 1))

            # initialize image stack
            if (is_first is True):
                images = image_matrix
                is_first = False
            else:
                if (image_channels > 1):
                    images = np.concatenate([images, image_matrix], axis=3)  # concatenate (on last axis)
                else:
                    images = np.dstack([images, image_matrix])  # depth stack images (on last axis)
        except Exception as e:
            if (file_obj["quiet"] is False):
                print("Failed reading image data frame: %s" % (str(e)))
            metadata_dict_list.pop()  # remove corresponding metadata entry
            problematic = True
            error_message = "image data read failure: %s" % (str(e))
            continue  # skip to next frame

    # cleanup
    #
    # NOTE: we only clean up the working dir if we created it
    if (working_dir_created is True):
        shutil.rmtree(this_working_dir)

    # check to see if the image is empty
    if (images.size == 0):
        if (file_obj["quiet"] is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"

    # return
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype


def __rgb_readfile_worker_pgm(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    is_first = True
    metadata_dict = {}
    site_uid = ""
    device_uid = ""
    problematic = False
    error_message = ""
    image_width = __RGB_PGM_EXPECTED_WIDTH
    image_height = __RGB_PGM_EXPECTED_HEIGHT
    image_channels = 1
    image_dtype = np.dtype("uint16")

    # Set metadata values
    file_split = os.path.basename(file_obj["filename"]).split('_')
    site_uid = file_split[3]
    device_uid = file_split[4]

    # check file extension to see if it's gzipped or not
    try:
        if file_obj["filename"].endswith("pgm.gz"):
            unzipped = gzip.open(file_obj["filename"], mode='rb')
        elif file_obj["filename"].endswith("pgm"):
            unzipped = open(file_obj["filename"], mode='rb')
        else:
            if (file_obj["quiet"] is False):
                print("Unrecognized file type: %s" % (file_obj["filename"]))
            problematic = True
            error_message = "Unrecognized file type"
            try:
                unzipped.close()
            except Exception:
                pass
            return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
                image_width, image_height, image_channels, image_dtype
    except Exception as e:
        if (file_obj["quiet"] is False):
            print("Failed to open file '%s' " % (file_obj["filename"]))
        problematic = True
        error_message = "failed to open file: %s" % (str(e))
        try:
            unzipped.close()
        except Exception:
            pass
        return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
            image_width, image_height, image_channels, image_dtype

    # read the file
    prev_line = None
    line = None
    while True:
        # break out depending on first_frame param
        if (file_obj["first_frame"] is True and is_first is False):
            break

        # read a line
        try:
            prev_line = line
            line = unzipped.readline()
        except Exception as e:
            if (file_obj["quiet"] is False):
                print("Error reading before image data in file '%s'" % (file_obj["filename"]))
            problematic = True
            metadata_dict_list = []
            images = np.array([])
            error_message = "error reading before image data: %s" % (str(e))
            try:
                unzipped.close()
            except Exception:
                pass
            return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
                image_width, image_height, image_channels, image_dtype

        # break loop at end of file
        if (line == b''):
            break

        # magic number; this is not a metadata or image line, exclude
        if (line.startswith(b'P5\n')):
            continue

        # process line
        if (line.startswith(b'#"')):
            if (file_obj["no_metadata"] is True):
                metadata_dict = {}
                metadata_dict_list.append(metadata_dict)
            else:
                # metadata lines start with #"<key>"
                try:
                    line_decoded = line.decode("ascii")
                except Exception as e:
                    # skip metadata line if it can't be decoded, likely corrupt file
                    if (file_obj["quiet"] is False):
                        print("Error decoding metadata line: %s (line='%s', file='%s')" % (str(e), line, file_obj["filename"]))
                    problematic = True
                    error_message = "error decoding metadata line: %s" % (str(e))
                    continue

                # split the key and value out of the metadata line
                line_decoded_split = line_decoded.split('"')
                key = line_decoded_split[1]
                value = line_decoded_split[2].strip()

                # add entry to dictionary
                if (key in metadata_dict):
                    # key already exists, turn existing value into list and append new value
                    if (isinstance(metadata_dict[key], list)):
                        # is a list already
                        metadata_dict[key].append(value)
                    else:
                        metadata_dict[key] = [metadata_dict[key], value]
                else:
                    # normal metadata value
                    metadata_dict[key] = value

                # split dictionaries up per frame, exposure plus initial readout is
                # always the end of metadata for frame
                if (key.startswith("Effective image exposure")):
                    metadata_dict_list.append(metadata_dict)
                    metadata_dict = {}
        elif line == b'65535\n':
            # there are 2 lines between "exposure plus read out" and the image
            # data, the first is the image dimensions and the second is the max
            # value
            #
            # check the previous line to get the dimensions of the image
            prev_line_split = prev_line.decode("ascii").strip().split()
            image_width = int(prev_line_split[0])
            image_height = int(prev_line_split[1])
            bytes_to_read = image_width * image_height * 2  # 16-bit image depth

            # read image
            try:
                # read the image size in bytes from the file
                image_bytes = unzipped.read(bytes_to_read)

                # format bytes into numpy array of unsigned shorts (2byte numbers, 0-65536),
                # effectively an array of pixel values
                #
                # NOTE: this is set to a different dtype that what we return on purpose.
                image_np = np.frombuffer(image_bytes, dtype=__RGB_PGM_DT)

                # change 1d numpy array into matrix with correctly located pixels
                image_matrix = np.reshape(image_np, (image_height, image_width, 1))
            except Exception as e:
                if (file_obj["quiet"] is False):
                    print("Failed reading image data frame: %s" % (str(e)))
                metadata_dict_list.pop()  # remove corresponding metadata entry
                problematic = True
                error_message = "image data read failure: %s" % (str(e))
                continue  # skip to next frame

            # initialize image stack
            if (is_first is True):
                images = image_matrix
                is_first = False
            else:
                images = np.dstack([images, image_matrix])  # depth stack images (on 3rd axis)

    # close gzip file
    unzipped.close()

    # set the site/device uids, or inject the site and device UIDs if they are missing
    if ("Site unique ID" not in metadata_dict):
        metadata_dict["Site unique ID"] = site_uid

    if ("Imager unique ID" not in metadata_dict):
        metadata_dict["Imager unique ID"] = device_uid

    # check to see if the image is empty
    if (images.size == 0):
        if (file_obj["quiet"] is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"

    # return
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype


def read_rgb(file_list, workers=1, first_frame=False, no_metadata=False, tar_tempdir=None, quiet=False):
    """
    Read in a single H5 or PNG.tar file, or an array of them. All files
    must be the same type. This also works for reading in PGM or untarred PNG
    files.

    :param file_list: filename or list of filenames
    :type file_list: str
    :param workers: number of worker processes to spawn, defaults to 1
    :type workers: int, optional
    :param first_frame: only read the first frame for each file, defaults to False
    :type first_frame: bool, optional
    :param no_metadata: exclude reading of metadata (performance optimization if
                        the metadata is not needed), defaults to False
    :type no_metadata: bool, optional
    :param tar_tempdir: path to untar to, defaults to '~/.trex_imager_readfile'
    :type tar_tempdir: str, optional
    :param quiet: reduce output while reading data
    :type quiet: bool, optional

    :return: images, metadata dictionaries, and problematic files
    :rtype: numpy.ndarray, list[dict], list[dict]
    """
    # set tar path
    if (tar_tempdir is None):
        tar_tempdir = pathlib.Path("%s/.trex_imager_readfile" % (str(pathlib.Path.home())))
    os.makedirs(tar_tempdir, exist_ok=True)

    # if input is just a single file name in a string, convert to a list to be fed to the workers
    if isinstance(file_list, str):
        file_list = [file_list]

    # convert to object, injecting other data we need for processing
    processing_list = []
    for f in file_list:
        processing_list.append({
            "filename": f,
            "tar_tempdir": tar_tempdir,
            "first_frame": first_frame,
            "no_metadata": no_metadata,
            "quiet": quiet,
        })

    # check workers
    if (workers > 1):
        try:
            # set up process pool (ignore SIGINT before spawning pool so child processes inherit SIGINT handler)
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = Pool(processes=workers)
            signal.signal(signal.SIGINT, original_sigint_handler)  # restore SIGINT handler
        except ValueError:
            # likely the read call is being used within a context that doesn't support the usage
            # of signals in this way, proceed without it
            pool = Pool(processes=workers)

        # call readfile function, run each iteration with a single input file from file_list
        # NOTE: structure of data - data[file][metadata dictionary lists = 1, images = 0][frame]
        pool_data = []
        try:
            pool_data = pool.map(__trex_readfile_worker, processing_list)
        except KeyboardInterrupt:
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0, 0)), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        pool_data = []
        for p in processing_list:
            pool_data.append(__trex_readfile_worker(p))

    # set sizes
    image_width = pool_data[0][5]
    image_height = pool_data[0][6]
    image_channels = pool_data[0][7]
    image_dtype = pool_data[0][8]

    # derive number of frames to prepare for
    total_num_frames = 0
    for i in range(0, len(pool_data)):
        if (pool_data[i][2] is True):
            continue
        if (image_channels > 1):
            total_num_frames += pool_data[i][0].shape[3]
        else:
            total_num_frames += pool_data[i][0].shape[2]

    # pre-allocate array sizes
    if (image_channels > 1):
        images = np.empty([image_height, image_width, image_channels, total_num_frames], dtype=image_dtype)
    else:
        images = np.empty([image_height, image_width, total_num_frames], dtype=image_dtype)
    metadata_dict_list = [{}] * total_num_frames
    problematic_file_list = []

    # populate data
    list_position = 0
    for i in range(0, len(pool_data)):
        # check if file was problematic
        if (pool_data[i][2] is True):
            problematic_file_list.append({
                "filename": pool_data[i][3],
                "error_message": pool_data[i][4],
            })
            continue

        # check if any data was read in
        if (len(pool_data[i][1]) == 0):
            continue

        # find actual number of frames, this may differ from predicted due to dropped frames, end
        # or start of imaging
        if (image_channels > 1):
            this_num_frames = pool_data[i][0].shape[3]
        else:
            this_num_frames = pool_data[i][0].shape[2]

        # metadata dictionary list at data[][1]
        metadata_dict_list[list_position:list_position + this_num_frames] = pool_data[i][1]
        if (image_channels > 1):
            images[:, :, :, list_position:list_position + this_num_frames] = pool_data[i][0]
        else:
            images[:, :, list_position:list_position + this_num_frames] = pool_data[i][0]
        list_position = list_position + this_num_frames  # advance list position

    # trim unused elements from predicted array sizes
    metadata_dict_list = metadata_dict_list[0:list_position]
    if (image_channels > 1):
        images = np.delete(images, range(list_position, total_num_frames), axis=3)
    else:
        images = np.delete(images, range(list_position, total_num_frames), axis=2)

    # ensure entire array views as the desired dtype
    images = images.astype(image_dtype)

    # return
    pool_data = None
    return images, metadata_dict_list, problematic_file_list


# globals
__NIR_EXPECTED_HEIGHT = 256
__NIR_EXPECTED_WIDTH = 256
__NIR_DT = np.dtype("uint16")
__NIR_DT = __NIR_DT.newbyteorder('>')  # force big endian byte ordering


def __nir_readfile_worker(file, first_frame=False, no_metadata=False, quiet=False):
    # init
    images = np.array([])
    metadata_dict_list = []
    is_first = True
    metadata_dict = {}
    site_uid = ""
    device_uid = ""
    problematic = False
    error_message = ""

    # set site UID and device UID in case we need it (ie. dark frames, or unstacked files)
    file_split = os.path.basename(file).split('_')
    if (len(file_split) == 5):
        # is a regular file
        site_uid = file_split[2]
        device_uid = file_split[3]
    elif (len(file_split) > 5):
        # is likely a dark frame or a unstacked frame
        site_uid = file_split[3]
        device_uid = file_split[4]

    # check file extension to see if it's gzipped or not
    try:
        if file.endswith("pgm.gz"):
            unzipped = gzip.open(file, mode='rb')
        elif file.endswith("pgm"):
            unzipped = open(file, mode='rb')
        else:
            if (quiet is False):
                print("Unrecognized file type: %s" % (file))
            problematic = True
            error_message = "Unrecognized file type"
            try:
                unzipped.close()
            except Exception:
                pass
            return images, metadata_dict_list, problematic, file, error_message
    except Exception as e:
        if (quiet is False):
            print("Failed to open file '%s' " % (file))
        problematic = True
        error_message = "failed to open file: %s" % (str(e))
        try:
            unzipped.close()
        except Exception:
            pass
        return images, metadata_dict_list, problematic, file, error_message

    # read the file
    prev_line = None
    line = None
    while True:
        # break out depending on first_frame param
        if (first_frame is True and is_first is False):
            break

        # read a line
        try:
            prev_line = line
            line = unzipped.readline()
        except Exception as e:
            if (quiet is False):
                print("Error reading before image data in file '%s'" % (file))
            problematic = True
            metadata_dict_list = []
            images = np.array([])
            error_message = "error reading before image data: %s" % (str(e))
            try:
                unzipped.close()
            except Exception:
                pass
            return images, metadata_dict_list, problematic, file, error_message

        # break loop at end of file
        if (line == b''):
            break

        # magic number; this is not a metadata or image line, exclude
        if (line.startswith(b'P5\n')):
            continue

        # process line
        if (line.startswith(b'#"')):
            if (no_metadata is True):
                metadata_dict = {}
                metadata_dict_list.append(metadata_dict)
            else:
                # metadata lines start with #"<key>"
                try:
                    line_decoded = line.decode("ascii")
                except Exception as e:
                    # skip metadata line if it can't be decoded, likely corrupt file
                    if (quiet is False):
                        print("Error decoding metadata line: %s (line='%s', file='%s')" % (str(e), line, file))
                    problematic = True
                    error_message = "error decoding metadata line: %s" % (str(e))
                    continue

                # split the key and value out of the metadata line
                line_decoded_split = line_decoded.split('"')
                key = line_decoded_split[1]
                value = line_decoded_split[2].strip()

                # add entry to dictionary
                metadata_dict[key] = value

                # set the site/device uids, or inject the site and device UIDs if they are missing
                if ("Site unique ID" not in metadata_dict):
                    metadata_dict["Site unique ID"] = site_uid
                else:
                    site_uid = metadata_dict["Site unique ID"]
                if ("Imager unique ID" not in metadata_dict):
                    metadata_dict["Imager unique ID"] = device_uid
                else:
                    device_uid = metadata_dict["Imager unique ID"]

                # split dictionaries up per frame, exposure plus initial readout is
                # always the end of metadata for frame
                if (key.startswith("Exposure plus readout")):
                    metadata_dict_list.append(metadata_dict)
                    metadata_dict = {}
        elif line == b'65535\n':
            # there are 2 lines between "exposure plus read out" and the image
            # data, the first is the image dimensions and the second is the max
            # value
            #
            # check the previous line to get the dimensions of the image
            prev_line_split = prev_line.decode("ascii").strip().split()
            image_width = int(prev_line_split[0])
            image_height = int(prev_line_split[1])
            bytes_to_read = image_width * image_height * 2  # 16-bit image depth

            # read image
            try:
                # read the image size in bytes from the file
                image_bytes = unzipped.read(bytes_to_read)

                # format bytes into numpy array of unsigned shorts (2byte numbers, 0-65536),
                # effectively an array of pixel values
                image_np = np.frombuffer(image_bytes, dtype=__NIR_DT)

                # change 1d numpy array into matrix with correctly located pixels
                image_matrix = np.reshape(image_np, (image_height, image_width, 1))
            except Exception as e:
                if (quiet is False):
                    print("Failed reading image data frame: %s" % (str(e)))
                metadata_dict_list.pop()  # remove corresponding metadata entry
                problematic = True
                error_message = "image data read failure: %s" % (str(e))
                continue  # skip to next frame

            # initialize image stack
            if (is_first is True):
                images = image_matrix
                is_first = False
            else:
                images = np.dstack([images, image_matrix])  # depth stack images (on 3rd axis)

    # close gzip file
    unzipped.close()

    # check to see if the image is empty
    if (images.size == 0):
        if (quiet is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"

    # return
    return images, metadata_dict_list, problematic, file, error_message


def read_nir(file_list, workers=1, first_frame=False, no_metadata=False, quiet=False):
    """
    Read in a single PGM file or set of PGM files

    :param file_list: filename or list of filenames
    :type file_list: str
    :param workers: number of worker processes to spawn, defaults to 1
    :type workers: int, optional
    :param first_frame: only read the first frame for each file, defaults to False
    :type first_frame: bool, optional
    :param no_metadata: exclude reading of metadata (performance optimization if
                        the metadata is not needed), defaults to False
    :type no_metadata: bool, optional
    :param quiet: reduce output while reading data
    :type quiet: bool, optional

    :return: images, metadata dictionaries, and problematic files
    :rtype: numpy.ndarray, list[dict], list[dict]
    """
    # if input is just a single file name in a string, convert to a list to be fed to the workers
    if isinstance(file_list, str):
        file_list = [file_list]

    # check workers
    if (workers > 1):
        try:
            # set up process pool (ignore SIGINT before spawning pool so child processes inherit SIGINT handler)
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = Pool(processes=workers)
            signal.signal(signal.SIGINT, original_sigint_handler)  # restore SIGINT handler
        except ValueError:
            # likely the read call is being used within a context that doesn't support the usage
            # of signals in this way, proceed without it
            pool = Pool(processes=workers)

        # call readfile function, run each iteration with a single input file from file_list
        # NOTE: structure of data - data[file][metadata dictionary lists = 1, images = 0][frame]
        data = []
        try:
            data = pool.map(functools.partial(
                __nir_readfile_worker,
                first_frame=first_frame,
                no_metadata=no_metadata,
                quiet=quiet,
            ), file_list)
        except KeyboardInterrupt:
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0), dtype=__NIR_DT), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        data = []
        for f in file_list:
            data.append(__nir_readfile_worker(
                f,
                first_frame=first_frame,
                no_metadata=no_metadata,
                quiet=quiet,
            ))

    # derive number of frames to prepare for
    total_num_frames = 0
    image_height = __NIR_EXPECTED_HEIGHT
    image_width = __NIR_EXPECTED_WIDTH
    for i in range(0, len(data)):
        if (data[i][2] is True):
            continue
        total_num_frames += data[i][0].shape[2]
        image_height = data[i][0].shape[0]
        image_width = data[i][0].shape[1]

    # pre-allocate array sizes
    images = np.empty([image_height, image_width, total_num_frames], dtype=__NIR_DT)
    metadata_dict_list = [{}] * total_num_frames
    problematic_file_list = []

    # populate data
    list_position = 0
    for i in range(0, len(data)):
        # check if file was problematic
        if (data[i][2] is True):
            problematic_file_list.append({
                "filename": data[i][3],
                "error_message": data[i][4],
            })
            continue

        # check if any data was read in
        if (len(data[i][1]) == 0):
            continue

        # find actual number of frames, this may differ from predicted due to dropped frames, end
        # or start of imaging
        real_num_frames = data[i][0].shape[2]

        # metadata dictionary list at data[][1]
        metadata_dict_list[list_position:list_position + real_num_frames] = data[i][1]
        images[:, :, list_position:list_position + real_num_frames] = data[i][0]  # image arrays at data[][0]
        list_position = list_position + real_num_frames  # advance list position

    # trim unused elements from predicted array sizes
    metadata_dict_list = metadata_dict_list[0:list_position]
    images = np.delete(images, range(list_position, total_num_frames), axis=2)

    # ensure entire array views as uint16
    images = images.astype(np.uint16)

    # return
    data = None
    return images, metadata_dict_list, problematic_file_list