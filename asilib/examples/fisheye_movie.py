from datetime import datetime

import asilib

asi_array_code = 'THEMIS'
location_code = 'FSMI'
time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 30))

asilib.animate_fisheye(asi_array_code, location_code, time_range, overwrite=True)
print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')
