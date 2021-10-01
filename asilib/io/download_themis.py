import requests
from datetime import datetime
from typing import List, Union
import dateutil.parser
import pathlib
import warnings

import asilib
import asilib.io.utils as utils

"""
This program contains the Time History of Events and Macroscale Interactions during 
Substorms (THEMIS) download functions that stream image and skymap data from the 
themis.ssl.berkeley.edu and data.phys.ucalgary.ca servers. The data is saved to 
asilib.config['ASI_DATA_DIR']/themis directory.
"""

IMG_BASE_URL = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/'
SKYMAP_BASE_URL = 'https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/skymaps/'

# Check and make a asilib.config['ASI_DATA_DIR']/themis/ directory if doesn't already exist.
themis_dir = pathlib.Path(asilib.config['ASI_DATA_DIR'], 'themis')
if not themis_dir.exists():
    themis_dir.mkdir(parents=True)


def download_themis_img(
    location_code: str,
    time: Union[datetime, str] = None,
    time_range: Union[datetime, str] = None,
    force_download: bool = False,
) -> List[pathlib.Path]:
    """
    This function downloads the THEMIS ASI image data given the day, location_code,
    and a flag to download a single hour file or the entire day. The images
    are saved to the asilib.config['ASI_DATA_DIR'] / 'themis' directory.

    Parameters
    ----------
    location_code: str
        The location_code to download the data from.
    time: datetime.datetime or str
        The date and time to download the data from. If day is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object.
    time_range: a list of len(2) of datetime.datetime or str
        Two time to download the data from. If day is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object.
    force_download: bool (optional)
        If True, download the file even if it already exists.

    Returns
    -------
    download_paths: list
        A list of pathlib.Path objects that contain the downloaded file
        path(s).

    Example
    -------
    from datetime import datetime

    import asilib

    | day = datetime(2017, 4, 13, 5)
    | station = 'LUCK'
    | asilib.download_themis_img(day, station)
    """
    if (time is None) and (time_range is None):
        raise AttributeError('Neither time or time_range is specified.')
    elif (time is not None) and (time_range is not None):
        raise AttributeError('Both time and time_range can not be simultaneously specified.')

    elif time is not None:
        time = utils._validate_time(time)
        download_path = _download_one_img_file(location_code, time, force_download)
        download_paths = [
            download_path
        ]  # List for constancy with the time_range code chunk output.

    elif time_range is not None:
        time_range = utils._validate_time_range(time_range)
        download_hours = utils._get_hours(time_range)
        download_paths = []

        for hour in download_hours:
            download_path = _download_one_img_file(location_code, hour, force_download)
            download_paths.append(download_path)

    return download_paths


def _download_one_img_file(location_code, time, force_download):
    """
    Download one hour-long file.
    """
    # Add the location/year/month url folders onto the url
    url = IMG_BASE_URL + f'{location_code.lower()}/{time.year}/{str(time.month).zfill(2)}/'

    search_pattern = f'{location_code.lower()}_{time.strftime("%Y%m%d%H")}'
    file_names = utils._search_hrefs(url, search_pattern=search_pattern)

    server_url = url + file_names[0]
    download_path = pathlib.Path(asilib.config['ASI_DATA_DIR'], 'themis', file_names[0])
    if force_download or (not download_path.is_file()):
        utils._stream_large_file(server_url, download_path)
    return download_path


def download_themis_skymap(station: str, force_download: bool = False):
    """
    Download all of the (skymap) IDL .sav file and save
    it to asilib.config['ASI_DATA_DIR']/themis/skymap/ directory.

    Parameters
    ----------
    station: str
        The station name, case insensitive
    force_download: bool (optional)
        If True, download the file even if it already exists.

    Returns
    -------
    None

    Example
    -------
    import asilib

    | station = 'LUCK'
    | asilib.download_themis_skymap(station)
    """
    # Create the skymap directory in data/themis/skymap/station
    save_dir = asilib.config['ASI_DATA_DIR'] / 'themis' / 'skymap' / station.lower()
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)
        print(f'Made directory at {save_dir}')

    url = SKYMAP_BASE_URL + f'{station.lower()}/'

    # Look for all of the skymap hyperlinks, go in each one of them, and
    # download the .sav file.
    skymap_folders_relative = utils._search_hrefs(url, search_pattern=station.lower())
    download_paths = []

    for skymap_folder in skymap_folders_relative:
        skymap_folder_absolute = url + skymap_folder

        # Lastly, research for the skymap .sav file.
        skymap_name = utils._search_hrefs(skymap_folder_absolute, search_pattern=f'.sav')[0]
        skymap_save_name = skymap_name.replace('-%2B', '')  # Replace the unicode '+'.

        # Download if force_download=True or the file does not exist.
        download_path = pathlib.Path(save_dir, skymap_save_name)
        download_paths.append(download_path)
        if force_download or (not download_path.is_file()):
            utils._stream_large_file(skymap_folder_absolute + skymap_name, download_path)
    return download_paths
