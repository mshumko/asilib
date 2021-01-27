import pathlib
from datetime import datetime, timedelta
import dateutil.parser
from typing import List, Union, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import cdflib

import asi.download.download_rego as download_rego
import asi.download.download_themis as download_themis
import asi.config as config

def plot_frame(time: Union[datetime, str], mission: str, station: str, 
            force_download: bool=False, time_thresh_s: float=3, 
            ax: plt.subplot=None, add_label: bool=True, color_map: str='hot',
            color_bounds: Union[List[float], None]=None, color_norm: str='log') -> plt.subplot:
    """
    Plots one ASI image frame given the mission (THEMIS or REGO), station, and 
    the day date-time parameters. If a file does not locally exist, the load()
    function will attempt to download it.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string, 
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the time argument in seconds. Will raise a ValueError if no 
        image time stamp is within the threshold. 
    ax: plt.subplot
        The subplot to plot the frame on. If None, this function will 
        create one.
    add_label: bool
        Flag to add the "mission/station/frame_time" text to the plot.
    color_map: str
        The matplotlib colormap to use. See 
        https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    color_bounds: List[float] or None
        The lower and upper values of the color scale. If None, will 
        automatically set it to low=1st_quartile and 
        high=min(3rd_quartile, 10*1st_quartile)
    color_norm: str
        Sets the 'lin' linear or 'log' logarithmic color normalization.

    Returns
    -------
    ax: plt.subplot
        Same subplot object if ax was passed as an argument, or a new
        subplot object contaning the frame.
    im: plt.imshow
        The plt.imshow object that can be used to add a colorbar.

    Example
    -------
    ax, im = plot_frame(datetime(2016, 10, 29, 4, 15), 'REGO', 'GILL', color_norm='log')
    plt.colorbar(im)
    plt.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    frame_time, frame = get_frame(time, mission, station, 
        force_download=force_download, time_thresh_s=time_thresh_s)
    
    # Figure out the color_bounds from the frame data.
    if color_bounds is None:
        lower, upper = np.quantile(frame, (0.25, 0.98))
        color_bounds = [lower, np.min([upper, lower*10])]

    if color_norm == 'log':
        norm=colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
    elif color_norm == 'lin':
        norm=colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    else:
        raise ValueError('color_norm must be either "log" or "lin".')

    im = ax.imshow(frame, cmap=color_map, norm=norm)
    ax.text(0, 0, f"{mission}/{station}\n{frame_time.strftime('%Y-%m-%d %H:%M:%S')}", 
            va='bottom', transform=ax.transAxes, color='white')
    return ax, im
    

def get_frame(time: Union[datetime, str], mission: str, station: str, 
            force_download: bool=False, time_thresh_s: float=3) -> Union[datetime, np.ndarray]:
    """
    Gets one ASI image frame given the mission (THEMIS or REGO), station, and 
    the day date-time parameters. If a file does not locally exist, this 
    function will attempt to download it.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string, 
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
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


def get_frames(time_range: List[Union[datetime, str]], mission: str, station: str, 
            force_download: bool=False) -> Union[datetime, np.ndarray]:
    """
    Gets multiple ASI image frames given the mission (THEMIS or REGO), station, and 
    the time_range date-time parameters. If a file does not locally exist, this 
    function will attempt to download it.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        frames. If either start or end time is a string, 
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
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
        The frame timestamps contained in time_range, inclduing the start 
        and end times.
    frame: np.ndarray
        An (nTime x nPixelRows x nPixelCols) array containing the ASI images
        for times contained in time_range.

    Example
    -------
    time_range = [datetime(2016, 10, 29, 4, 15), datetime(2016, 10, 29, 4, 20)]
    times, frames = get_frames(time_range, 'REGO', 'GILL')
    """
    # Run a few checks to make sure that the time_range parameter has length 2 and
    # convert time_range to datetime objects if it is a string.
    assert len(time_range) == 2, (f'len(time_range) = {len(time_range)} is not 2.')
    for i, t_i in enumerate(time_range):
        if isinstance(t_i, str):
            time_range[i] = dateutil.parser.parse(t_i)

    cdf_obj = load(time_range[0], mission, station, force_download=force_download)

    # Figure out the data keys to load.
    if mission.lower() == 'rego':
        frame_key = f'clg_rgf_{station.lower()}'
        time_key  = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        raise NotImplementedError

    # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00) 
    # to datetime objects.
    epoch = np.array(cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key)))
    # Find the time stamps in between time_range.
    idx = np.where((epoch >= time_range[0]) & (epoch <= time_range[1]))[0]
    assert len(idx), (f'{len(idx)} number of time stamps were found '
                      f'in time_range={time_range}')
    return epoch[idx], cdf_obj.varget(frame_key)[idx, :, :]


def load(time: Union[datetime, str], mission: str, station: str, 
            force_download: bool=False) -> cdflib.cdfread.CDF:
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
    # time, frame = get_frame(datetime(2016, 10, 29, 4, 15), 'REGO', 'GILL')
    # rego_data = load(datetime(2016, 10, 29, 4), 'REGO', 'GILL')
    # time_range: List[datetime] = [datetime(2016, 10, 29, 4, 15), datetime(2016, 10, 29, 4, 20)]
    # times, frames = get_frames(time_range, 'REGO', 'GILL')
    ax, im = plot_frame(datetime(2015, 4, 9, 7, 35, 6), 'REGO', 'FSMI', color_norm='log')
    plt.colorbar(im)
    plt.axis('off')
    plt.show()
