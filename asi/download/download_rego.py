import requests
from datetime import datetime
from typing import List, Union
import dateutil.parser
# import argparse

from bs4 import BeautifulSoup

from asi import config

"""
This program contains functions to download the Red-line Emission Geospace 
Observatory (REGO) data from the https://data.phys.ucalgary.ca server.
"""

BASE_URL = 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/stream0/'

def search_hrefs(url: str, search_pattern: str ='') -> List[str]:
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
    soup = BeautifulSoup(request.content, 'html.parser')

    for href in soup.find_all('a', href=True):
        if (href.text != '../') and (search_pattern.lower() in href.text.lower()):
            matched_hrefs.append(href.text)
    if len(matched_hrefs) == 0:
        raise NotADirectoryError(f'The url {url} does not contain any hyper '
            f'references containing the search_pattern="{search_pattern}".')
    return matched_hrefs


def download(day: Union[datetime, str], station: str, download_minute: bool=True):
    """
    The wrapper to download the REGO data given the day, station name,
    and a flag to download a single minute file or the entire hour. The images
    are saved to the config.ASI_DATA_DIR / 'rego' directory. 

    Parameters
    ----------
    day: datetime.datetime or str
        The date and time to download the data from. If day is string, 
        dateutil.parser.parse will attempt to parse it into a datetime
        object.
    station: str
        The station id to download the data from.
    download_minute: bool (optinal)
        If True, will download only one minute of image data, otherwise it will
        download image data from the entire hour.

    Returns
    -------
    None
    """
    if isinstance(day, str):
        day = dateutil.parser.parse(day)
    # Add the year/month/day url folders onto the url
    url = BASE_URL + f'{day.year}/{str(day.month).zfill(2)}/{str(day.day).zfill(2)}/'

    # Find if the particular camera station was taking data on that day.
    station_url = search_hrefs(url, search_pattern=station.lower())
    # Append the station url directory and the UTC hour to the url.
    url +=  f'{station_url[0]}ut{str(day.hour).zfill(2)}/'

    if download_minute:
        # Find an image file for the one minute.
        file_names = search_hrefs(url, search_pattern=day.strftime('%Y%m%d_%H%M'))
        # Download file
        r = requests.get(url + file_names[0], allow_redirects=False)
        with open(config.ASI_DATA_DIR / 'rego' / file_names[0], 'wb') as f:
            f.write(r.content)
    else:
        # Otherwise find all of the image files for that station and UT hour.
        file_names = search_hrefs(url)
        # Download files
        for file_name in file_names:
            r = requests.get(url + file_name, allow_redirects=False)
            with open(config.ASI_DATA_DIR / 'rego' / file_name, 'wb') as f:
                f.write(r.content)
    return

day = datetime(2017, 4, 13, 5, 10)
station = 'LUCK'


# url = (f'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/stream0/'
#         f'{day.year}/{str(day.month).zfill(2)}/{str(day.day).zfill(2)}/')

# print(url)

# print(search_hrefs(url, search_pattern='luck'))

# r = requests.get(url)
# soup = BeautifulSoup(r.content, 'html.parser')

# hrefs = soup.find_all('a', href=True)
# print(hrefs)

# for href in hrefs:
#     if station.lower() in href.text:
#         station_url = href.text
#         break
#     else:
#         station_url = ''

# if station_url == '':
#     raise NotADirectoryError(f'The the directory for {station} not found in \n{url}.')

# # 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/stream0/2017/04/13/luck_rego-649/ut05/'

# url2 = url + station_url + f'ut{str(day.hour).zfill(2)}/'

# r = requests.get(url2)
# soup = BeautifulSoup(r.content, 'html.parser')

# hrefs = soup.find_all('a', href=True)


# date_str = day.strftime('%Y%m%d_%H%M')
# for href in hrefs:
#     # '20170413_0510'
#     if date_str in href.text:
#         station_url = href.text
#         print(station_url)
#         break
#     else:
#         station_url = ''

# url3 = url2 + station_url
# r = requests.get(url3, allow_redirects=True)
# with open(station_url, 'wb') as f:
#     f.write(r.content)