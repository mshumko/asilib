# TODO: Remove when cleaning up Imager
"""
A set of utility functions for asilib.
"""
import dateutil.parser
import pathlib
from typing import List, Union
import copy
from datetime import timedelta, datetime

from bs4 import BeautifulSoup
import requests
import numpy as np


_time_type = Union[datetime, str]
_time_range_type = List[_time_type]


def _validate_time(time: _time_type) -> List[datetime]:
    """
    Validates tries to parse the time into datetime objects.
    """
    if isinstance(time, str):
        time = dateutil.parser.parse(time)
    elif isinstance(time, (int, float)):
        raise ValueError(f'Unknown time format, {time}')
    _is_valid_year(time)
    return time


def _validate_time_range(time_range: _time_range_type) -> List[datetime]:
    """
    Validates tries to parse the time_range into datetime objects.
    """
    if time_range is None:
        return None

    assert isinstance(
        time_range, (list, tuple, np.ndarray)
    ), "time_range must be a list, tuple, or np.ndarray."
    assert len(time_range) == 2, "time_range must be a list or a tuple with start and end times."

    time_range_parsed = []

    for t in time_range:
        if isinstance(t, str):
            time_range_parsed.append(dateutil.parser.parse(t))
        elif isinstance(t, (int, float)):
            raise ValueError(f'Unknown time format, {t}')
        else:
            time_range_parsed.append(t)

    for t in time_range_parsed:
        _is_valid_year(t)

    time_range_parsed.sort()
    return time_range_parsed


def _is_valid_year(time):
    """
    Provides a sanity check that the year is after 2000 and before the current year + 1.
    """
    year = time.year
    if year < 2000:
        raise ValueError(f'The passed year={year} must be greater than 2000.')
    elif year > datetime.now().year + 1:
        raise ValueError(f'The passed year={year} must be less than the current year + 1.')
    return


def _get_hours(time_range: _time_range_type) -> List[datetime]:
    """
    Helper function to figure out what date-hour times are between the times in time_range.
    This function is useful to figure out what hourly ASI files to download.
    """
    # Modify time_range. If time_range[0] is not at the top of the hour, we zero the minutes
    # seconds, and milliseconds. This helps with keeping the + 1 hour offsets aligned to the
    # start of the hour.
    current_hour = copy.copy(time_range[0].replace(minute=0, second=0, microsecond=0))
    hours = []

    # Not <= in while loop because we don't want to download the final hour if time_range[1] is,
    # for example, 05:00:00 [HH:MM:SS].
    while current_hour < time_range[1]:
        hours.append(current_hour)
        current_hour += timedelta(hours=1)
    return hours


def _stream_large_file(
    url: str, save_path: Union[pathlib.Path, str], test_flag: bool = False
) -> None:
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


def _search_hrefs(url: str, search_pattern: str = '.cdf') -> List[str]:
    """
    Given a url string, this function returns all of the
    hyper references (hrefs, or hyperlinks). If search_pattern is not
    specified, a default '.cdf' value is assumed and this function
    will return all hrefs with the CDF extension. If no hrefs containing
    search_pattern are found, this function raises a NotADirectoryError.
    The search is case-insensitive.

    Parameters
    ----------
    url: str
        A url in string format
    search_pattern: str (optional)
        Find the exact search_pattern text contained in the hrefs.
        By default all hrefs matching the extension ".cdf" are returned.

    Returns
    -------
    hrefs: List(str)
        A list of hrefs that contain the search_pattern.

    Raises
    ------
    NotADirectoryError
        If a hyper reference (a folder or a file) is not found on the
        server. This is raised if the data does not exist.
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
