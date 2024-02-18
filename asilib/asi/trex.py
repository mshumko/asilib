"""
Transition Region Explorer (TREx) data is courtesy of Space Environment Canada (space-environment.ca). Use of the data must adhere to the rules of the road for that dataset.  Please see below for the required data acknowledgement. Any questions about the TREx instrumentation or data should be directed to the University of Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan (edonovan@ucalgary.ca).

“The Transition Region Explorer RGB (TREx RGB) is a joint Canada Foundation for Innovation and Canadian Space Agency project developed by the University of Calgary. TREx-RGB is operated and maintained by Space Environment Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”

For more information see: https://www.ucalgary.ca/aurora/projects/trex
"""

from datetime import datetime, timedelta
import re
import warnings
import pathlib
import copy
import os
import dateutil.parser
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import scipy.io
import trex_imager_readfile

import asilib
from asilib.asi.themis import _get_pgm_files
import asilib.utils as utils
import asilib.io.download as download
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
    imager=asilib.Imager,
) -> asilib.imager.Imager:
    # TODO: Remove the warning in 2024.
    """
    Create an Imager instance using the TREX-RGB ASI images and skymaps.

    Transition Region Explorer (TREx) RGB data is courtesy of Space Environment Canada (space-environment.ca). Use of the data must adhere to the rules of the road for that dataset.  Please see below for the required data acknowledgement. Any questions about the TREx instrumentation or data should be directed to the University of Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan (edonovan@ucalgary.ca).

    “The Transition Region Explorer RGB (TREx RGB) is a joint Canada Foundation for Innovation and Canadian Space Agency project developed by the University of Calgary. TREx-RGB is operated and maintained by Space Environment Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”

    For more information see: https://www.ucalgary.ca/aurora/projects/trex.

    .. warning::

        In early October 2023 the TREx-RGB data format changed, which resulted in a "ValueError: 
        A problematic PGM file..." exception for asilib versions <= 0.20.1. If you're having this 
        issue, you'll need to upgrade asilib to version >= 0.20.2 and delete the outdated TREx RGB
        image files. The code below is the simplest solution:

        .. code-block:: python

            import os
            import shutil

            os.system("pip install aurora-asi-lib -U")

            import asilib

            shutil.rmtree(asilib.config['ASI_DATA_DIR'] / 'trex' / 'rgb')

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
    }
    plot_settings = {'color_norm':'lin'}
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
                except (FileNotFoundError, AssertionError) as err:
                    if missing_ok and (
                        ('does not contain any hyper references containing' in str(err))
                        or ('Only one href is allowed' in str(err))
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
                    except (FileNotFoundError, AssertionError) as err:
                        if missing_ok and (
                            ('does not contain any hyper references containing' in str(
                                err))
                            or ('Only one href is allowed' in str(err))
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
    d = download.Downloader(start_url)
    # Find the unique directory
    matched_downloaders = d.ls(f'{location_code.lower()}_{array}*')
    assert len(matched_downloaders) == 1
    # Search that directory for the file and donload it.
    d2 = download.Downloader(matched_downloaders[0].url + f'ut{time.hour:02}/')
    file_search_str = f'{time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}*{array}*.h5'
    matched_downloaders2 = d2.ls(file_search_str)
    assert len(matched_downloaders2) == 1
    return matched_downloaders2[0].download(save_dir, redownload=redownload)


def _load_rgb_h5(path):
    images, meta, problematic_file_list = trex_imager_readfile.read_rgb(
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
    }
    return imager(file_info, meta, skymap)


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
    d = download.Downloader(url)
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
    # TODO: Remove the import once opencv is updated and trex_imager_readfile can be
    # updated https://github.com/opencv/opencv/issues/23059
    import trex_imager_readfile

    images, meta, problematic_file_list = trex_imager_readfile.read_nir(
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import asilib.map
    import asilib.asi

    time_range = ('2023-02-24T05:30', '2023-02-24T06:30')

    asi = asilib.asi.trex_rgb('RABB', time_range=time_range)
    asi.plot_keogram()
    # asi['2023-02-24T06:10'].plot_fisheye()

    plt.show()
