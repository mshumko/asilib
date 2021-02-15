import requests
from datetime import datetime
from typing import List, Union
import dateutil.parser
import pathlib

from asilib import config
from asilib.download import download_rego

"""
This program contains the Time History of Events and Macroscale Interactions during 
Substorms (THEMIS) download functions that stream image and calibration data from the 
themis.ssl.berkeley.edu server and saves the files to the 
asilib.config.ASI_DATA_DIR/themis/ directory.
"""

IMG_BASE_URL = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/'
CAL_BASE_URL = 'http://themis.ssl.berkeley.edu/data/themis/thg/l2/asi/cal/'

# Check and make a config.ASI_DATA_DIR/themis/ directory if doesn't already exist.
if not pathlib.Path(config.ASI_DATA_DIR, 'themis').exists():
    pathlib.Path(config.ASI_DATA_DIR, 'themis').mkdir()


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
    are saved to the config.ASI_DATA_DIR / 'themis' directory.

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
        download image data from the entire day.
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

    day = datetime(2017, 4, 13, 5)
    station = 'LUCK'
    asilib.download_themis_img(day, station)
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
            config.ASI_DATA_DIR, 'themis', file_names[0]
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
            download_path = pathlib.Path(config.ASI_DATA_DIR, 'themis', file_name)
            download_paths.append(download_path)
            # Download if force_download=True or the file does not exist.
            if force_download or (not download_path.is_file()):
                download_rego.stream_large_file(download_url, download_path, test_flag=test_flag)
        return download_paths


def download_themis_cal(station: str, force_download: bool = False):
    """
    This function downloads the calibration cdf files for the
    station THEMIS ASI.

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

    station = 'LUCK'
    asilib.download_themis_cal(station)
    """
    # Create the calibration directory in data/themis/cal
    save_dir = config.ASI_DATA_DIR / 'themis' / 'cal'
    if not save_dir.is_dir():
        save_dir.mkdir()
        print(f'Made directory at {save_dir}')

    # Search all of the skymap files with the macthing station name.
    search_pattern = f'themis_skymap_{station.lower()}'
    file_names = download_rego.search_hrefs(CAL_BASE_URL, search_pattern=search_pattern)

    # Download the latest skymap file
    download_url = CAL_BASE_URL + file_names[-1]
    download_path = pathlib.Path(save_dir, file_names[-1])
    # Download if force_download=True or the file does not exist.
    if force_download or (not download_path.is_file()):
        download_rego.stream_large_file(download_url, download_path)
    return download_path
