from datetime import datetime

import asilib.asi

location_code = 'FSMI'
time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 30))
asi = asilib.asi.themis(location_code, time_range=time_range)
asi.animate_fisheye()

print(f'Animation saved in {asilib.config["ASI_DATA_DIR"] / "animations" / asi.animation_name}')
