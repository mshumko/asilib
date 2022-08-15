from datetime import datetime

import pytest
import numpy as np

from asilib.imager import Imager

# TODO-Validation: Rewrite to raise an error when you call an Imager method, not in __init__.
# def test_single_image_time_inputs():
#     valid_times = [datetime.now(), datetime.now().isoformat()]
#     invalid_times = ['Today', '0', 1000]

#     image = np.zeros((10,10))

#     for valid_time in valid_times:
#         Imager({'Time':valid_time, 'image':image}, {}, {})

#     for invalid_time in invalid_times:
#         with pytest.raises(ValueError):
#             Imager({'Time':invalid_time, 'image':image}, {}, {})
#     return

# TODO-Validation: Rewrite to raise an error when you call an Imager method, not in __init__.
# def test_correct_data_keys():
#     time = datetime.now()
#     image = np.zeros((10,10))

#     valid_data_dicts = [
#         {'time':time, 'image':image},
#         {'start_time':5*[time], 'end_time':5*[time], 'loader':None, 'path':None}
#     ]

#     invalid_data_dicts = [
#         {'time':time},
#         {'image':image},
#         {'start_time':5*[time], 'end_time':5*[time]},
#         {'loader':None, 'path':None},
#         {'start_time':5*[time], 'loader':None, 'path':None}
#     ]

#     for data in valid_data_dicts:
#         Imager(data, {}, {})

#     for data in invalid_data_dicts:
#         with pytest.raises(AttributeError):
#             Imager(data, {}, {})
#     return
