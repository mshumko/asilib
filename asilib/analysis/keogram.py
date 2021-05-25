from asilib.io.load import get_frames, load_cal, _validate_time_range


def keogram(time_range, mission, station, map_alt=None):
    """
    Makes a keogram along the central meridian.

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
    """
    raise NotImplementedError
    return