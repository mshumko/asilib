import numpy as np
import pandas as pd

from asilib.io.load import (
    load_image_generator,
    load_skymap,
    _validate_time_range,
    _create_empty_data_arrays,
)


def keogram(time_range, mission, station, map_alt=None):
    """
    Makes a keogram pd.DataFrame along the central meridian.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        images. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    map_alt: int, optional
        The mapping altitude, in kilometers, used to index the mapped latitude in the
        skymap data. If None, will plot pixel index for the y-axis.

    Returns
    -------
    keo: pd.DataFrame
        The 2d keogram with the time index. The columns are the geographic latitude
        if map_alt != None, otherwise it is the image pixel values (0-265) or (0-512).

    Raises
    ------
    AssertionError
        If map_alt does not equal the mapped altitudes in the skymap mapped values.
    """
    time_range = _validate_time_range(time_range)
    keo_times, keo = _create_empty_data_arrays(mission, time_range, 'keogram')
    image_generator = load_image_generator(time_range, mission, station)

    start_time_index = 0
    for file_image_times, file_images in image_generator:
        end_time_index = start_time_index + file_images.shape[0]
        keo[start_time_index:end_time_index, :] = file_images[
            :, :, keo.shape[1] // 2
        ]  # Slice the meridian
        keo_times[start_time_index:end_time_index] = file_image_times
        start_time_index += file_images.shape[0]

    # This code block removes any filler nan values if the ASI images were not sampled at the instrument
    # cadence throughout time_range.
    i_nan = np.where(~np.isnan(keo[:, 0]))[0]
    keo = keo[i_nan, :]
    keo_times = keo_times[i_nan]

    if map_alt is None:
        keogram_latitude = np.arange(keo.shape[1])  # Dummy index values for latitudes.
    else:
        skymap = load_skymap(mission, station, time_range[0])
        assert (
            map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
        ), f'{map_alt} km is not in skymap altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
        alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]
        keogram_latitude = skymap['FULL_MAP_LATITUDE'][alt_index, :, keo.shape[1] // 2]

        # keogram_latitude array are at the pixel edges. Remap it to the centers
        dl = keogram_latitude[1:] - keogram_latitude[:-1]
        keogram_latitude = keogram_latitude[0:-1] + dl / 2

        # Since keogram_latitude values are NaNs near the image edges, we want to filter
        # out those indices from keogram_latitude and keo.
        valid_lats = np.where(~np.isnan(keogram_latitude))[0]
        # The ::-1 reverses the latitude array to make them in ascending order.
        keogram_latitude = keogram_latitude[valid_lats][::-1]
        keo = keo[:, valid_lats]
    return pd.DataFrame(data=keo, index=keo_times, columns=keogram_latitude)
