from datetime import datetime

import asilib

time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
asi_array_code = 'THEMIS'
asi_location_code = 'FSMI'
# If you don't provide your own map, asilib will create a default map of Canada.
asilib.animate_map(asi_array_code, asi_location_code, time_range, 110, overwrite=True)

print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')