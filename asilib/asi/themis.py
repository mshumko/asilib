"""
The THEMIS ground-based All-Sky Imager (ASI) array observes the white light aurora over the North American continent from Canada to Alaska. The ASI array consists of 20 cameras covering a large section of the auroral oval with one-kilometer resolution. The all-sky imagers are time synchronized and have an image repetition rate of three seconds. During northern winter, continuous coverage is available from about 00:00-15:00 UT covering approximately 17-07 MLT for each individual site. Geographic locations and more details are available using asilib.asi.themis.themis_info(). The full-resolution 256x256 pixel images are transferred via hard-disk swap and become available approximately 3-5 months after data collection.
"""

from datetime import datetime, timedelta
from multiprocessing import Pool
import re
import warnings
import pathlib
import copy
import os
import dateutil.parser
from typing import Tuple, Iterable, List, Union

import numpy as np
import pandas as pd
import scipy.io
import themis_imager_readfile

import asilib
import asilib.utils as utils
import asilib.io.download as download


pgm_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/'
skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/skymaps/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'themis'


def themis(
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
    Create an Imager instance with the THEMIS ASI images and skymaps.

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
        A THEMIS ASI instance with the time stamps, images, skymaps, and metadata.
    """

    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_pgm_dir = local_base_dir / 'images' / location_code.lower()

    if load_images:
        # Download and find image data
        file_paths = _get_pgm_files(
            'themis',
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
            'loader': _load_pgm,
        }
    else:
        file_info = {
            'path': [],
            'start_time': [],
            'end_time': [],
            'loader': [],
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
    _skymap = themis_skymap(location_code, _time, redownload)
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
        'array': 'THEMIS',
        'location': location_code.upper(),
        'lat': float(_skymap['SITE_MAP_LATITUDE']),
        'lon': float(_skymap['SITE_MAP_LONGITUDE']),
        'alt': float(_skymap['SITE_MAP_ALTITUDE']) / 1e3,
        'cadence': 3,
        'resolution': (256, 256),
    }
    return imager(file_info, meta, skymap)


def themis_skymap(location_code, time, redownload):
    """
    Load a THEMIS ASI skymap file.

    Parameters
    ----------
    location_code: str
        The four character location name.
    time: str or datetime.datetime
        A ISO-fomatted time string or datetime object. Must be in UT time.
    redownload: bool
        Redownload all skymaps.
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
            f'The requested skymap time={time} for THEMIS-{location_code.upper()} is before first '
            f'skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = local_skymap_paths[closest_index]
    skymap = _load_skymap(skymap_path)
    return skymap


def themis_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the THEMIS ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of THEMIS imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'THEMIS']
    return df.reset_index(drop=True)


def _get_pgm_files(
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
    Look for and download 1-minute PGM files.

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
        raise ValueError('both time and time_range can not be simultaneously specified.')

    if redownload:
        # Option 1/4: Download one minute of data regardless if it is already saved
        if time is not None:
            return [
                _download_one_pgm_file(array, location_code, time, base_url, save_dir, redownload)
            ]

        # Option 2/4: Download the data in time range regardless if it is already saved.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='minutes')
            file_paths = []
            for file_time in file_times:
                try:
                    file_paths.append(
                        _download_one_pgm_file(
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
            file_search_str = f'{time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}*.pgm.gz'
            local_file_paths = list(pathlib.Path(save_dir).rglob(file_search_str))
            if len(local_file_paths) == 1:
                return local_file_paths
            else:
                return [
                    _download_one_pgm_file(
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
                    f'{file_time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}*.pgm.gz'
                )
                local_file_paths = list(pathlib.Path(save_dir).rglob(file_search_str))
                if len(local_file_paths) == 1:
                    file_paths.append(local_file_paths[0])
                else:
                    try:
                        file_paths.append(
                            _download_one_pgm_file(
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


def _download_one_pgm_file(
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
    file_search_str = f'{time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}*{array}*.pgm.gz'
    matched_downloaders2 = d2.ls(file_search_str)
    assert len(matched_downloaders2) == 1
    return matched_downloaders2[0].download(save_dir, redownload=redownload)


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
    A helper function to load a THEMIS skymap and transform it.
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
                skymap[key] = skymap[key][::-1, :]  # For Az/El maps.
            elif (len(shape) == 3) and (shape[1] == shape[2]):
                skymap[key] = skymap[key][:, ::-1, :]  # For lat/lon maps
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


def _load_pgm(path: Union[pathlib.Path, str]) -> Tuple[np.array, np.array]:
    """
    Read in a single THEMIS PGM file.

    Parameters
    ----------
    path: pathlib.Path or str
        The local file path.

    Returns
    -------
    times
        A 1d numpy array of time stamps.
    images
        A 3d numpy array of images with the first dimension corresponding to
        time. Images are oriented such that pixel located at (0, 0) is in the
        southeast corner.
    """
    images, meta, problematic_file_list = themis_imager_readfile.read(str(path), workers=1)
    if len(problematic_file_list):
        raise ValueError(f'A problematic PGM file: {problematic_file_list[0]}')
    images = np.moveaxis(images, 2, 0)
    images = images[:, ::-1, :]  # Flip north-south.
    times = np.array(
        [
            dateutil.parser.parse(dict_i['Image request start']).replace(tzinfo=None)
            for dict_i in meta
        ]
    )
    return times, images
