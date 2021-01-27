import pathlib
from datetime import datetime, timedelta
import dateutil.parser
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cdflib

import asi.download.download_rego as download_rego
import asi.download.download_themis as download_themis
import asi.config as config

def get_frame(time: Union[datetime, str], mission: str, station: str, 
            force_download: bool=False, time_thresh_s: float=3) -> Union[datetime, np.ndarray]:
    """
    Gets the ASI frame from the mission (THEMIS or REGO), from a
    station on a day. If a file does not locally exist, it will attempt
    to download it.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string, 
        dateutil.parser.parse will attempt to parse it into a datetime
        object. Must contain the date and the UT hour.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the time argument in seconds. Will raise a ValueError if no 
        image time stamp is within the threshold. 

    Returns
    -------
    frame_time: datetime
        The frame timestamp.
    frame: np.ndarray
        A 2D array of the ASI image at the date-time nearest to the
        time argument.
    
    Example
    -------
    time, frame = get_frame(datetime(2016, 10, 29, 4, 15), 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    cdf_obj = load(time, mission, station, force_download=force_download)

    if mission.lower() == 'rego':
        frame_key = f'clg_rgf_{station.lower()}'
        time_key  = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        raise NotImplementedError

    # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00) 
    # to datetime objects.
    epoch = np.array(cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key)))
    # Find the closest time stamp to time
    idx = np.where(
        (epoch >= time) & 
        (epoch < time + timedelta(seconds=time_thresh_s))
        )[0]
    assert len(idx) == 1, (f'{len(idx)} number of time stamps were found '
                        f'within {time_thresh_s} seconds of {time}.'
                        f'You can change the time_thresh_s kwarg to find a '
                        f'time stamp further away.')
    return epoch[idx[0]], cdf_obj.varget(frame_key)[idx[0], :, :]

def load(time: Union[datetime, str], mission: str, station: str, 
            force_download: bool=False):
    """
    Loads the REGO or THEMIS ASI CDF file.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string, 
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
    rego_data = load(datetime(2016, 10, 29, 4), 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    # Check if the REGO or THEMIS data is already saved locally.
    search_path = pathlib.Path(config.ASI_DATA_DIR, mission.lower())
    search_pattern = f'*{station.lower()}*{time.strftime("%Y%m%d%H")}*'
    matched_paths = list(search_path.rglob(search_pattern))
    # Try to download files if one is not found locally.
    if (len(matched_paths) == 0) and (mission.lower() == 'themis'):
        try:
            download_path = download_themis.download_themis_img(time, station, 
                            force_download=force_download)[0]
        except NotADirectoryError:
            raise ValueError(f'THEMIS ASI data not found for station {station} on day {time.date()}')
    elif (len(matched_paths) == 0) and (mission.lower() == 'rego'):
        try:
            download_path = download_rego.download_rego_img(time, station,
                            force_download=force_download)[0]
        except NotADirectoryError:
            raise ValueError(f'REGO ASI data not found for station {station} on day {time.date()}')
    else:
        download_path = matched_paths[0]

    # If we made it here, we either found a local file, or downloaded one
    return cdflib.CDF(download_path)

if __name__ == '__main__':
    time, frame = get_frame(datetime(2016, 10, 29, 4, 15), 'REGO', 'GILL')
