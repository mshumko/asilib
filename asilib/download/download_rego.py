import requests
from datetime import datetime
from typing import List, Union
import dateutil.parser
import pathlib

from bs4 import BeautifulSoup

from asilib import config

"""
This program contains the Red-line Emission Geospace Observatory (REGO) download functions
that stream image data from the themis.ssl.berkeley.edu server and the calibration data 
from the ucalgary.ca server and saves the files to the asilib.config.ASI_DATA_DIR/rego/ 
directory.
"""

IMG_BASE_URL = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/reg/'
CAL_BASE_URL = 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/skymap/'

# Check and make a config.ASI_DATA_DIR/rego/ directory if doesn't already exist.
rego_dir = pathlib.Path(config.ASI_DATA_DIR, 'rego')
if not rego_dir.exists():
    rego_dir.mkdir()


def download_rego_img(
    day: Union[datetime, str],
    station: str,
    download_hour: bool = True,
    force_download: bool = False,
    test_flag: bool = False,
) -> List[pathlib.Path]:
    """
    The wrapper to download the REGO data given the day, station name,
    and a flag to download a single hour file or the entire day. The images
    are saved to the config.ASI_DATA_DIR / 'rego' directory.

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
    asilib.download_rego_img(day, station)
    """
    if isinstance(day, str):
        day = dateutil.parser.parse(day)
    # Add the station/year/month url folders onto the url
    url = IMG_BASE_URL + f'{station.lower()}/{day.year}/{str(day.month).zfill(2)}/'

    if download_hour:
        # Find an image file for the hour.
        search_pattern = f'{station.lower()}_{day.strftime("%Y%m%d%H")}'
        file_names = search_hrefs(url, search_pattern=search_pattern)

        # Download file
        download_url = url + file_names[0]  # On the server
        download_path = pathlib.Path(
            config.ASI_DATA_DIR, 'rego', file_names[0]
        )  # On the local machine.
        # Download if force_download=True or the file does not exist.
        if force_download or (not download_path.is_file()):
            stream_large_file(download_url, download_path, test_flag=test_flag)
        return [download_path]
    else:
        # Otherwise find all of the image files for that station and UT hour.
        file_names = search_hrefs(url)
        download_paths = []
        # Download files
        for file_name in file_names:
            download_url = url + file_name
            download_path = pathlib.Path(config.ASI_DATA_DIR, 'rego', file_name)
            download_paths.append(download_path)
            # Download if force_download=True or the file does not exist.
            if force_download or (not download_path.is_file()):
                stream_large_file(download_url, download_path, test_flag=test_flag)
        return download_paths


def download_rego_cal(station: str, force_download: bool = False) -> pathlib.Path:
    """
    Download the latest calibration (skymap) IDL .sav file and save
    it to config.ASI_DATA_DIR/rego/cal/ directory.

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
    asilib.download_rego_cal(station)
    """
    # Create the calibration directory in data/rego/cal
    save_dir = config.ASI_DATA_DIR / 'rego' / 'cal'
    if not save_dir.is_dir():
        save_dir.mkdir()
        print(f'Made directory at {save_dir}')

    url = CAL_BASE_URL + f'{station.lower()}/'

    # Look for all of the hyperlinks to the calibration file and download the
    # latest one.
    cal_time_tagged_hrefs = search_hrefs(url, search_pattern=station.lower())
    url = url + cal_time_tagged_hrefs[-1]  # Last href is the latest one.
    # Lastly, research for the skymap .sav file.
    cal_hrefs = search_hrefs(url, search_pattern=f'rego_skymap_{station.lower()}')
    cal_name = cal_hrefs[0].replace('-%2B', '')  # Replace the code for '+'.

    # Download if force_download=True or the file does not exist.
    download_path = pathlib.Path(save_dir, cal_name)
    if force_download or (not download_path.is_file()):
        stream_large_file(url + cal_hrefs[0], download_path)
    return download_path


def stream_large_file(url, save_path, test_flag: bool = False):
    """
    Streams a file from url to save_path. In requests.get(), stream=True
    sets up a generator to download a small chuck of data at a time,
    instead of downloading the entire file into RAM first.

    Parameters
    ----------
    url: str
        The URL to the file.
    save_path: str or pathlib.Path
        The local save path for the file.
    test_flag: bool (optional)
        If True, the download will halt after one 5 Mb chunk of data is
        downloaded.

    Returns
    -------
    None
    """
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get('content-length'))
    downloaded_bites = 0

    save_name = pathlib.Path(save_path).name

    megabyte = 1024 * 1024

    with open(save_path, 'wb') as f:
        for data in r.iter_content(chunk_size=5 * megabyte):
            f.write(data)
            if test_flag:
                return
            # Update the downloaded % in the terminal.
            downloaded_bites += len(data)
            download_percent = round(100 * downloaded_bites / file_size)
            download_str = "#" * (download_percent // 5)
            print(f'Downloading {save_name}: |{download_str:<20}| {download_percent}%', end='\r')
    print()  # Add a newline
    return


def search_hrefs(url: str, search_pattern: str = '') -> List[str]:
    """
    Given a url string, this function returns all of the
    hyper references (hrefs, or hyperlinks) if search_pattern=='',
    or a specific href that contains the search_pattern. If search_pattern
    is not found, this function raises a NotADirectoryError. The
    search is case-insensitive, and it doesn't return the '../' href.

    Parameters
    ----------
    url: str
        A url in string format
    search_pattern: str (optional)
        Find the exact search_pattern text contained in the hrefs.

    Returns
    -------
    hrefs: List(str)
        A list of hrefs that contain the search_pattern.
    """
    matched_hrefs = []

    request = requests.get(url)
    # request.status_code
    soup = BeautifulSoup(request.content, 'html.parser')

    for href in soup.find_all('a', href=True):
        if search_pattern.lower() in href['href'].lower():
            matched_hrefs.append(href['href'])
    if len(matched_hrefs) == 0:
        raise NotADirectoryError(
            f'The url {url} does not contain any hyper '
            f'references containing the search_pattern="{search_pattern}".'
        )
    return matched_hrefs
