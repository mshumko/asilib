from datetime import datetime

import asilib.asi
import asilib.map

time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
location_code = 'FSMI'

asi = asilib.asi.themis(location_code, time_range=time_range, alt=110)

lat_bounds = (asi.meta['lat'] - 7, asi.meta['lat'] + 7)
lon_bounds = (asi.meta['lon'] - 20, asi.meta['lon'] + 20)
ax = asilib.map.create_map(lon_bounds=lon_bounds, lat_bounds=lat_bounds)

asi.animate_map(ax=ax)

print(f'Animation saved in {asilib.config["ASI_DATA_DIR"] / "animations" / asi.animation_name}')
