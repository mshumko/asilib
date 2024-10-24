"""
The Mid-latitude All-sky-imaging Network for Geophysical Observations (MANGO) employs a combination of two powerful optical techniques used to observe the dynamics of Earth's upper atmosphere: wide-field imaging and high-resolution spectral interferometry. Both techniques observe the naturally occurring airglow emissions produced in the upper atmosphere at 630.0- and 557.7-nm wavelengths. Instruments are deployed to sites across the continental United States, providing the capability to make measurements spanning mid to sub-auroral latitudes. The current instrument suite in MANGO has six all-sky imagers (ASIs) observing the 630.0-nm emission (integrated between ~200 and 400 km altitude), six ASIs observing the 557.7-nm emission (integrated between ~90 and 100 km altitude), and four Fabry-Perot interferometers measuring neutral winds and temperature at these wavelengths. The deployment of additional imagers is planned. The network makes unprecedented observations of the nighttime thermosphere-ionosphere dynamics with the expanded field-of-view provided by the distributed network of instruments. 

Instrument paper: https://doi.org/10.1029/2023JA031589
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

import matplotlib.colors
import pandas as pd
import h5py

import asilib
import asilib.utils as utils
import asilib.download as download

base_url = 'https://data.mangonetwork.org/data/transport/mango/archive/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'mango'


def mango(
    location_code: str,
    channel: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    acknowledge: bool = True,
    redownload: bool = False,
    missing_ok: bool = True,
    imager=asilib.Imager,
) -> asilib.Imager:
    """
    Create an Imager instance with the MANGO ASI images and skymaps.

    Parameters
    ----------
    location_code: str
        The ASI's location code (four letters).
    channel: str
        The color channel. Could be "redline", "greenline", "r", or "g". Case insensitive. 
    time: str or datetime.datetime
        A time to look for the ASI data at. Either time or time_range
        must be specified (not both or neither).
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    alt: int
        The reference skymap altitude, in kilometers.
    acknowledge: bool
        If True, prints the acknowledgment statement for REGO. 
    redownload: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.
        
    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A MANGO ASI instance with the time stamps, images, skymaps, and metadata.
    """
    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_image_dir = local_base_dir / 'images' / location_code.lower()

    channel = channel.lower()
    assert channel[0] in ['r', 'g'], (f"{channel} is an invalid MANGO color channel. "
                                      f"Try either 'redline' or 'greenline'.")

    file_paths = _get_image_files(
        location_code,
        channel,
        time,
        time_range,
        base_url,
        local_image_dir,
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
        'loader': _load_h5,
    }
    # meta = _load_h5_meta(file_paths[0])
    meta = {
        'array': 'MANGO',
        'location': location_code.upper(),
        'lat': None,
        'lon': None,
        'alt': None,
        'cadence': None,
        'resolution': (None, None),
        'acknowledgment': ''
    }
    plot_settings = {
        'color_map': matplotlib.colors.LinearSegmentedColormap.from_list('black_to_red', ['k', channel[0]])
    }
    skymap = {}
    # if acknowledge and ('mango' not in asilib.config['ACKNOWLEDGED_ASIS']):
    #     print(meta['acknowledgment'])
    #     asilib.config['ACKNOWLEDGED_ASIS'].append('mango')
    return imager(file_info, meta, skymap, plot_settings=plot_settings)


def mango_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the MANGO ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of THEMIS imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'MANGO']
    return df.reset_index(drop=True)

def _get_image_files(
    location_code: str,
    channel:str,
    time: datetime,
    time_range: Iterable[datetime],
    base_url: str,
    save_dir: Union[str, pathlib.Path],
    redownload: bool,
    missing_ok: bool,
    ) -> List[pathlib.Path]:
    """
    Find MANGO image files either locally or download them from the internet.

    Parameters
    ----------
    location_code:str
        The MANGO location code.
    channel: str
        The color channel. Could be "redline", "greenline", "r", or "g".
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
        Local paths to each h5 file that was successfully found.
    """
    if (time is None) and (time_range is None):
        raise ValueError('time or time_range must be specified.')
    elif (time is not None) and (time_range is not None):
        raise ValueError('both time and time_range can not be simultaneously specified.')
    
    if redownload:
        # Option 1/4: Download one minute of data regardless if it is already saved
        if time is not None:
            return [
                _download_one_file(location_code, channel, time, base_url, save_dir, redownload)
            ]

        # Option 2/4: Download the data in time range regardless if it is already saved.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='days')
            file_paths = []
            for file_time in file_times:
                try:
                    file_paths.append(
                        _download_one_file(
                            location_code, channel, file_time, base_url, save_dir, redownload
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
                    _download_one_file(
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
                            _download_one_file(
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
            if missing_ok and len(file_paths) == 0:
                if time_range is not None:
                    warnings.warn(
                        f'No local or online image files found for {array}-{location_code} '
                        f'for {time_range=}.'
                        )
                else:
                    warnings.warn(
                        f'No local or online image files found for {array}-{location_code} '
                        f'for {time=}.'
                        )
            return file_paths
    return

def _download_one_file(
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


def _load_h5(file_path):
    with h5py.File(file_path, 'r') as file:
        return file['Time'], file['ImageData']
            
    return

def _load_h5_meta(file_path):
    with h5py.File(file_path, 'r') as file:
        return file['Longitude'], file['Latitude']