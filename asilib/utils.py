"""
A set of utility functions for asilib.
"""
import dateutil.parser
import shutil
from typing import List, Union
from collections.abc import Iterable
import copy
from datetime import timedelta, datetime

import pandas as pd
import numpy as np


_time_type = Union[datetime, str]
_time_range_type = List[_time_type]


def validate_time(time: _time_type) -> List[datetime]:
    """
    Validates tries to parse the time into datetime objects.
    """
    if isinstance(time, str):
        time = dateutil.parser.parse(time)
    elif isinstance(time, (datetime, pd.Timestamp)):
        pass
    else:
        raise ValueError(f'Unknown time format, {time}')
    _validate_year(time)
    return time


def validate_time_range(time_range: _time_range_type) -> List[datetime]:
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
        _validate_year(t)

    time_range_parsed.sort()
    return time_range_parsed


def _validate_year(time):
    """
    Provides a sanity check that the year is after 2000 and before the current year + 1.
    """
    year = time.year
    if year < 2000:
        raise ValueError(f'The passed year={year} must be greater than 2000.')
    elif year > datetime.now().year:
        raise ValueError(f'The passed year={year} must be less than the current year + 1.')
    return


def get_filename_times(time_range: _time_range_type, dt='hours') -> List[datetime]:
    """
    Returns the dates and times within time_range and in dt steps. It returns times
    with the smaller components than dt set to zero, and increase by dt. This is useful
    to create a list of dates and times to load sequential files.

    time_range: list[datetime or str]
        A start and end time to calculate the file dates.
    dt: str
        The time difference between times. Can be 'days', 'hours', or 'minutes'.
    """
    time_range = validate_time_range(time_range)
    assert dt in ['days', 'hours', 'minutes'], "dt must be 'day', 'hour', or 'minute'."
    # First we need to appropriately zero time_range[0] so that the file that contains the
    # time_range[0] is returned. For example, if dt='hour' and time_range[0] is not at the
    # top of the hour, we zero the smaller time components. So
    # time_range[0] = '2010-01-01T10:29:32.000000' will be converted to '2010-01-01T10:00:00.000000'
    # to allow the
    zero_time_chunks = {'microsecond': 0, 'second': 0}
    # We don't need an if-statement if dt == 'minute'
    if dt == 'hours':
        zero_time_chunks['minute'] = 0
    elif dt == 'days':
        zero_time_chunks['minute'] = 0
        zero_time_chunks['hour'] = 0
    current_time = copy.copy(time_range[0].replace(**zero_time_chunks))
    times = []

    # Not <= in while loop because we don't want to download the final time if time_range[1] is,
    # exactly matches the end file name (you don't want to download the 11th hour if
    # time_range[1] is 'YYY-MM-DDT11:00:00').
    while current_time < time_range[1]:
        times.append(current_time)
        current_time += timedelta(**{dt: 1})
    return times


def progressbar(iterator: Iterable, iter_length: int = None, text: str = None):
    """
    A terminal progress bar.

    Parameters
    ----------
    iterator: Iterable
        The iterable that will be looped over.
    iter_length: int
        How many items the iterator loop over. If None, will calculate it
        using len(iterator).
    text: str
        Insert an optional text string in the beginning of the progressbar.
    """
    if text is None:
        text = ''
    else:
        text = text + ':'

    if iter_length is None:
        iter_length = len(iterator)

    try:
        for i, item in enumerate(iterator):
            i += 1  # So we end at 100%. Happy users!
            terminal_cols = shutil.get_terminal_size(fallback=(80, 20)).columns
            max_cols = int(terminal_cols - len(text) - 10)
            # Prevent a crash if the terminal window is narrower then len(text).
            if max_cols < 0:
                max_cols = 0

            percent = round(100 * i / iter_length)
            bar = "#" * int(max_cols * percent / 100)
            print(f'{text} |{bar:<{max_cols}}| {percent}%', end='\r')
            yield item
    finally:
        print()  # end with a newline.
