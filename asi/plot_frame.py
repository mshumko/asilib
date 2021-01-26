import pathlib
from datetime import datetime
import dateutil.parser
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cdflib

import asi.download.download_rego as download_rego
import asi.download.download_themis as download_themis
import asi.config as config

def plot_frame(day: Union[datetime, str], mission: str, station: str, 
            force_download: bool=False, time_thresh_s: float=3) -> Union[datetime, np.ndarray]:
    """
    Plots the ASI frame from the mission (THEMIS or REGO), from a
    station on a day. If a file does not locally exist, it will attempt
    to download it.

    Parameters
    ----------
    day: datetime.datetime or str
        The date and time to download the data from. If day is string, 
        dateutil.parser.parse will attempt to parse it into a datetime
        object. Must contain the date and the UT hour.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the day argument in seconds. Will raise a ValueError if no 
        image time stamp is within the threshold. 

    Returns
    -------
    frame_time: datetime
        The frame timestamp.
    frame: np.ndarray
        A 2D array of the ASI image at the date-time nearest to the
        day argument.
    
    Example
    -------
    """
    # Try to convert day to datetime object if it is a string.
    if isinstance(day, str):
        day = dateutil.parser.parse(day)

    return

def load(day: Union[datetime, str], mission: str, station: str, 
            force_download: bool=False):
    """
    Loads the REGO or THEMIS ASI CDF file.

    Parameters
    ----------
    day: datetime.datetime or str
        The date and time to download the data from. If day is string, 
        dateutil.parser.parse will attempt to parse it into a datetime
        object. Must contain the date and the UT hour.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.

    Returns
    -------
    frame_time: datetime
        The frame timestamp.
    frame: np.ndarray
        A 2D array of the ASI image at the date-time nearest to the
        day argument.
    
    Example
    -------

    """
    # Try to convert day to datetime object if it is a string.
    if isinstance(day, str):
        day = dateutil.parser.parse(day)

    # Check if the REGO or THEMIS data is already saved locally.
    search_path = pathlib.Path(config.ASI_DATA_DIR, mission.lower())
    search_pattern = f'*{station.lower()}*{day.strftime("%Y%m%d%H")}*'
    matched_paths = list(search_path.rglob(search_pattern))
    # Try to download files if one is not found locally.
    if (len(matched_paths) == 0) and (mission.lower() == 'themis'):
        try:
            download_path = download_themis.download_themis_img(day, station)
        except NotADirectoryError:
            raise ValueError(f'THEMIS ASI data not found for station {station} on day {day.date()}')
    elif (len(matched_paths) == 0) and (mission.lower() == 'rego'):
        try:
            download_path = download_rego.download_rego_img(day, station)
        except NotADirectoryError:
            raise ValueError(f'REGO ASI data not found for station {station} on day {day.date()}')
    else:
        download_path = matched_paths[0]

    # If we made it here, we either found a local file, or downloaded one
    return cdflib.CDF(download_path)

if __name__ == '__main__':
    rego_data = load(datetime(2016, 10, 29, 4), 'REGO', 'GILL')