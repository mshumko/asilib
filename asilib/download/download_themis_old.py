# This script downloads the THEMIS asi data using the themisasi
# API.

import numpy as np
import pandas as pd
import pathlib
import urllib.request
from datetime import datetime
from bs4 import BeautifulSoup
import re

from asilib import config

frame_base_url = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/'
cal_base_url = 'http://themis.ssl.berkeley.edu/data/themis/thg/l2/asi/cal/'

def load_curtain_catalog(catalog_name):
    cat_path = dirs.CATALOG_DIR / catalog_name
    cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)
    return cat

def get_unique_stations(cat):
    """ Get a set of unique stations from the catalog. """
    nearby_stations = [i.split(' ') for i in cat.nearby_stations]
    flattened_stations = [item for sublist in nearby_stations for item in sublist]
    stations = list(set(flattened_stations))
    return stations

def download_asi_calibration_wrapper(cat, overwrite=False):
    """ 
    Wrapper to find all unique stations in the catalog and 
    download that data. 
    """
    stations = get_unique_stations(cat)
    for station in stations:
        download_asi_calibration(station, overwrite=overwrite)
    return

def download_asi_frames_wrapper(cat, overwrite=False):
    """ 
    Loops over the curtain catalog and downloads the 
    ASI image data 
    """
    for t, row in cat.iterrows():
        # Figure out if you need to loop over multiple stations that were
        nearby_stations = row.nearby_stations.split()
        
        # Loop over the stations (or just one station) and download the data.
        # Continue if the data does not exist.
        for station in nearby_stations:
            try:
                download_asi_frames(t, station, overwrite=overwrite)
            except urllib.error.HTTPError as err:
                if 'HTTP Error 404: Not Found' == str(err):
                    continue
                else:
                    raise
    return

def download_asi_calibration(station, overwrite=False):
    """
    Scrape the ASI calibration website and download all cdf 
    calibration files from the station.
    """
    # Find all cdf files with the station name
    html = urllib.request.urlopen(cal_base_url).read().decode('utf-8')
    # Scrape the HTML
    soup = BeautifulSoup(html, 'html.parser')
    # Extract all cdf files
    file_name_html = soup.findAll(href=re.compile("\.cdf$"))
    # Extract all hyperlinks (href) filenames with the station name.
    file_names = [file_name.get('href') for file_name in file_name_html 
                    if station.lower() in file_name.get('href')]
    # Download data
    for file_name in file_names:
        # Skip if overwite is false and file is already downloaded
        if not overwrite and pathlib.Path(dirs.ASI_DIR, file_name).is_file():
            print(f'Skipping {file_name}')
        else:
            print(f'Downloading {file_name}')
        urllib.request.urlretrieve(cal_base_url + file_name, 
                                dirs.ASI_DIR / file_name)
    return

def download_asi_frames(time, station, overwrite=False):
    """ 
    Download the ASI image data from a date + hour in the datetime 
    time object and a particular station.
    """
    station_url = (f'{station.lower()}/{time.year}/{time.strftime("%m")}/')
    file_name = f'thg_l1_asf_{station.lower()}_{time.strftime("%Y%m%d%H")}_v01.cdf'

    if not overwrite and pathlib.Path(dirs.ASI_DIR, file_name).is_file():
        print(f'Skipping {file_name}')
        return
    else:
        print(f'Downloading {file_name}')

    try:
        urllib.request.urlretrieve(frame_base_url + station_url + file_name, 
                                dirs.ASI_DIR / file_name)
    except urllib.error.HTTPError as err:
        if '404' in str(err):
            print(frame_base_url + station_url + file_name)
            raise
        else:
            raise
    return

if __name__ == '__main__':
    catalog_name = 'AC6_curtains_themis_asi_15deg.csv'
    cat = load_curtain_catalog(catalog_name)

    # Download the calibration data.
    download_asi_calibration_wrapper(cat)

    # Download the frame data
    download_asi_frames_wrapper(cat)
