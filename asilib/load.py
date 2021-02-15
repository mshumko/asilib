import pathlib
from datetime import datetime, timedelta
import dateutil.parser
from typing import List, Union, Sequence, Tuple
from copy import copy

import pandas as pd
import numpy as np
import cdflib
import scipy.io
import matplotlib.pyplot as plt

from asilib.download import download_rego
from asilib.download import download_themis
from asilib import config


def load_img_file(
    time: Union[datetime, str], mission: str, station: str, force_download: bool = False
) -> cdflib.cdfread.CDF:
    """
    Loads the REGO or THEMIS ASI CDF file and downloads one if it doesn't
    exist on the local computer.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. Must contain the date and the UT hour.
    mission: str
        The mission id, can be either THEMIS or REGO.
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
    import asilib
    rego_data = asilib.load_img_file('2016-10-29T04', 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    # Download data if force_download == True:
    # Check if the REGO or THEMIS data is already saved locally.
    search_path = pathlib.Path(config.ASI_DATA_DIR, mission.lower())
    search_pattern = f'*{station.lower()}*{time.strftime("%Y%m%d%H")}*'
    matched_paths = list(search_path.rglob(search_pattern))
    # Try to download files if one is not found locally.

    if (len(matched_paths) == 1) and (not force_download):
        # If a local file was found and the user does not want to force the download.
        download_path = matched_paths[0]

    elif (len(matched_paths) == 1) and (force_download):
        # If a local file was found and the user does want to force the download.
        # These downloaders are guaranteed to find a matching file unless the server
        # lost the file.
        if mission.lower() == 'themis':
            download_path = download_themis.download_themis_img(
                time, station, force_download=force_download
            )[0]
        elif mission.lower() == 'rego':
            download_path = download_rego.download_rego_img(
                time, station, force_download=force_download
            )[0]

    elif len(matched_paths) == 0:
        # Now if no local files were found, try to download it.
        if mission.lower() == 'themis':
            try:
                download_path = download_themis.download_themis_img(
                    time, station, force_download=force_download
                )[0]
            except NotADirectoryError:
                raise FileNotFoundError(
                    f'THEMIS ASI data not found for station {station} on day {time.date()}'
                )
        elif mission.lower() == 'rego':
            try:
                download_path = download_rego.download_rego_img(
                    time, station, force_download=force_download
                )[0]
            except NotADirectoryError:
                raise FileNotFoundError(
                    f'REGO ASI data not found for station {station} on day {time.date()}'
                )
    else:
        raise ValueError(f"Not sure what happend here. I found {matched_paths} mathching paths.")

    # If we made it here, we either found a local file, or downloaded one
    return cdflib.CDF(download_path)


def load_cal_file(mission: str, station: str, force_download: bool = False) -> dict:
    """
    Loads the latest callibration file for the mission/station and downloads
    one if one is not found in the config.ASI_DATA_DIR/mission/cal/ folder.

    Parameters
    ----------
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.

    Returns
    -------
    cal_dict: dict
        A calibration dictionary with longitudes mapped from -180 to 180
        degrees.

    Example
    -------
    import asilib

    rego_cal = asilib.load_cal_file('REGO', 'GILL')
    """
    cal_dir = config.ASI_DATA_DIR / mission.lower() / 'cal'
    cal_paths = sorted(list(cal_dir.rglob(f'{mission.lower()}_skymap_{station.lower()}*')))

    # If no THEMIS cal files found, download the lastest one.
    if len(cal_paths) == 0 and mission.lower() == 'themis':
        cal_path = download_themis.download_themis_cal(station)
    # If no REGO cal files found, download the lastest one.
    elif len(cal_paths) == 0 and mission.lower() == 'rego':
        cal_path = download_rego.download_rego_cal(station)
    else:
        cal_path = cal_paths[-1]

    # Load the calibration file and convert it to a dictionary.
    cal_file = scipy.io.readsav(cal_path, python_dict=True)['skymap']
    cal_dict = {key: copy(cal_file[key][0]) for key in cal_file.dtype.names}
    # Map longitude from 0 - 360 to -180 - 180.
    cal_dict['SITE_MAP_LONGITUDE'] = np.mod(cal_dict['SITE_MAP_LONGITUDE'] + 180, 360) - 180
    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(cal_dict['FULL_MAP_LONGITUDE']))
    cal_dict['FULL_MAP_LONGITUDE'][valid_val_idx] = (
        np.mod(cal_dict['FULL_MAP_LONGITUDE'][valid_val_idx] + 180, 360) - 180
    )
    return cal_dict


def get_frame(
    time: Union[datetime, str],
    mission: str,
    station: str,
    force_download: bool = False,
    time_thresh_s: float = 3,
) -> Tuple[datetime, np.ndarray]:
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
    mission: str
        The mission id, can be either THEMIS or REGO.
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
    import asilib

    time, frame = asilib.get_frame('2016-10-29T04:15:00', 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    cdf_obj = load_img_file(time, mission, station, force_download=force_download)

    if mission.lower() == 'rego':
        frame_key = f'clg_rgf_{station.lower()}'
        time_key = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        frame_key = f'thg_asf_{station.lower()}'
        time_key = f'thg_asf_{station.lower()}_epoch'

    # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00)
    # to datetime objects.
    epoch = np.array(cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key)))
    # Find the closest time stamp to time
    idx = np.where((epoch >= time) & (epoch < time + timedelta(seconds=time_thresh_s)))[0]
    assert len(idx) == 1, (
        f'{len(idx)} number of time stamps were found '
        f'within {time_thresh_s} seconds of {time}. '
        f'You can change the time_thresh_s kwarg to find a '
        f'time stamp further away.'
    )
    return epoch[idx[0]], cdf_obj.varget(frame_key)[idx[0], :, :]


def get_frames(
    time_range: Sequence[Union[datetime, str]],
    mission: str,
    station: str,
    force_download: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
    mission: str
        The mission id, can be either THEMIS or REGO.
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
    from datetime import datetime

    import asilib

    time_range = [datetime(2016, 10, 29, 4, 15), datetime(2016, 10, 29, 4, 20)]
    times, frames = asilib.get_frames(time_range, 'REGO', 'GILL')
    """
    # Run a few checks to make sure that the time_range parameter has length 2 and
    # convert time_range to datetime objects if it is a string.
    assert len(time_range) == 2, f'len(time_range) = {len(time_range)} is not 2.'
    for i, t_i in enumerate(time_range):
        if isinstance(t_i, str):
            time_range[i] = dateutil.parser.parse(t_i)

    # Sort time_range if the user passed it in out of order.
    time_range = sorted(time_range)

    # Figure out the data keys to load.
    if mission.lower() == 'rego':
        frame_key = f'clg_rgf_{station.lower()}'
        time_key = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        frame_key = f'thg_asf_{station.lower()}'
        time_key = f'thg_asf_{station.lower()}_epoch'

    # Determine if we need to load in one hour or multiple hours worth of data.
    start_time_rounded = time_range[0].replace(minute=0, second=0, microsecond=0)
    end_time_rounded = time_range[1].replace(minute=0, second=0, microsecond=0)
    if start_time_rounded == end_time_rounded:
        # If the start/end date-hour are the same than load one data file, otherwise
        # load however many is necessary and concatinate the frames and times.
        cdf_obj = load_img_file(time_range[0], mission, station, force_download=force_download)

        # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00)
        # to datetime objects.
        epoch = np.array(cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key)))

        # Get the frames 3d array
        frames = cdf_obj.varget(frame_key)
    else:
        # If the time_range is across two or more hours
        epoch = np.array([])
        if mission.lower() == 'themis':
            frames = np.ones((0, 256, 256))
        elif mission.lower() == 'rego':
            frames = np.ones((0, 512, 512))
        else:
            raise NotImplementedError

        # The timedelta offset is needed to include the time_range[1] hour.
        hourly_date_times = pd.date_range(
            start=time_range[0], end=time_range[1] + pd.Timedelta(hours=1), freq='H'
        )
        for hour_date_time in hourly_date_times:
            cdf_obj = load_img_file(hour_date_time, mission, station, force_download=force_download)

            # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00)
            # to datetime objects.
            epoch = np.append(
                epoch, np.array(cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key)))
            )
            # Get the frames 3d array and concatenate.
            frames = np.concatenate((frames, cdf_obj.varget(frame_key)), axis=0)

    # Find the time stamps in between time_range.
    idx = np.where((epoch >= time_range[0]) & (epoch <= time_range[1]))[0]
    assert len(idx) > 0, (
        f'{len(idx)} number of time stamps were found ' f'in time_range={time_range}'
    )
    return epoch[idx], frames[idx, :, :]
