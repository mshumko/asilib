import numpy as np
import pandas as pd

from asilib.io.load import get_frames, load_cal, _validate_time_range

def keogram(time_range, mission, station, map_alt=None):
    """
    Makes a keogram pd.DataFrame along the central meridian.

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
    map_alt: int, optional
        The mapping altitude, in kilometers, used to index the mapped latitude in the 
        calibration data. If None, will plot pixel index for the y-axis.

    Returns
    -------
    keo: pd.DataFrame
        The 2d keogram with the time index. The columns are the geographic latitude
        if map_alt != None, otherwise it is the image pixel values (0-265) or (0-512).
    Raises
    ------
    AssertionError
        If map_alt does not equal the mapped altitudes in the calibration mapped values.
    """
    time_range = _validate_time_range(time_range)
    frame_times, frames = get_frames(time_range, mission, station)

    # Find the pixel at the center of the camera.
    center_pixel = int(frames.shape[1]/2)

    # Get the meridian from all of the frames.
    keo = frames[:, :, center_pixel]

    if map_alt is None:
        keogram_latitude = np.arange(frames.shape[1])  # Dummy index values.
    else:
        cal = load_cal(mission, station)
        assert map_alt in cal['FULL_MAP_ALTITUDE']/1000, \
            f'{map_alt} km is not in calibration altitudes: {cal["FULL_MAP_ALTITUDE"]/1000} km'
        alt_index = np.where(cal['FULL_MAP_ALTITUDE']/1000 == map_alt)[0][0]
        keogram_latitude = cal['FULL_MAP_LATITUDE'][alt_index, :, center_pixel]

        # Since keogram_latitude values are NaNs near the image edges, we want to filter
        # out those indices from keogram_latitude and keo.
        valid_lats = np.where(~np.isnan(keogram_latitude))[0]
        # The ::-1 reverses the latitude array to make them in ascending order.
        keogram_latitude = keogram_latitude[valid_lats][::-1]
        keo = keo[:, valid_lats]
    return pd.DataFrame(data=keo, index=frame_times, columns=keogram_latitude)