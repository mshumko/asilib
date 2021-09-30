import requests
from datetime import datetime
from typing import List, Union
import dateutil.parser
import pathlib
import warnings

import asilib
from asilib.io import download_rego

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
    day: Union[datetime, str],
    station: str,
    download_hour: bool = True,
    force_download: bool = False,
    test_flag: bool = False,
) -> List[pathlib.Path]:
    """
    This function downloads the THEMIS ASI image data given the day, station name,
    and a flag to download a single hour file or the entire day. The images
    are saved to the asilib.config['ASI_DATA_DIR'] / 'themis' directory.

    Parameters
    ----------
    day: datetime.datetime or str
        The date and time to download the data from. If day is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object.
    station: str
        The station id to download the data from.
    download_hour: bool (optinal)
        If True, will download only one hour of image data, otherwise it will
        download image data from the entire folder (typically a month)
        from URL = IMG_BASE_URL/station/year/month. This kwarg is useful if you
        need to download recently-uploaded data that is not fully processed
        and has the YYYYMMDD (not YYYYMMDDHH) date string in the filename.
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
    if isinstance(day, str):
        day = dateutil.parser.parse(day)
    # Add the station/year/month url folders onto the url
    url = IMG_BASE_URL + f'{station.lower()}/{day.year}/{str(day.month).zfill(2)}/'

    if download_hour:
        # Find an image file for the hour.
        search_pattern = f'{station.lower()}_{day.strftime("%Y%m%d%H")}'
        file_names = download_rego.search_hrefs(url, search_pattern=search_pattern)

        # Download file
        download_url = url + file_names[0]  # On the server
        download_path = pathlib.Path(
            asilib.config['ASI_DATA_DIR'], 'themis', file_names[0]
        )  # On the local machine.
        # Download if force_download=True or the file does not exist.
        if force_download or (not download_path.is_file()):
            download_rego.stream_large_file(download_url, download_path, test_flag=test_flag)
        return [download_path]
    else:
        # Otherwise find all of the image files for that station and UT hour.
        file_names = download_rego.search_hrefs(url)
        download_paths = []
        # Download files
        for file_name in file_names:
            download_url = url + file_name
            download_path = pathlib.Path(themis_dir, file_name)
            download_paths.append(download_path)
            # Download if force_download=True or the file does not exist.
            if force_download or (not download_path.is_file()):
                download_rego.stream_large_file(download_url, download_path, test_flag=test_flag)
        return download_paths


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
    skymap_folders_relative = download_rego.search_hrefs(url, search_pattern=station.lower())
    download_paths = []

    for skymap_folder in skymap_folders_relative:
        skymap_folder_absolute = url + skymap_folder

        # Lastly, research for the skymap .sav file.
        skymap_name = download_rego.search_hrefs(skymap_folder_absolute, search_pattern=f'.sav')[0]
        skymap_save_name = skymap_name.replace('-%2B', '')  # Replace the unicode '+'.

        # Download if force_download=True or the file does not exist.
        download_path = pathlib.Path(save_dir, skymap_save_name)
        download_paths.append(download_path)
        if force_download or (not download_path.is_file()):
            download_rego.stream_large_file(skymap_folder_absolute + skymap_name, download_path)
    return download_paths