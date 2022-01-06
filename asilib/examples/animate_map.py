from datetime import datetime

import asilib

time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
asi_array_code = 'THEMIS'
location_code = 'FSMI'

# We need the skymap only to center the map on the projected image.
skymap = asilib.load_skymap(asi_array_code, location_code, time_range[0])
lat_bounds = (skymap['SITE_MAP_LATITUDE'] - 7, skymap['SITE_MAP_LATITUDE'] + 7)
lon_bounds = (skymap['SITE_MAP_LONGITUDE'] - 20, skymap['SITE_MAP_LONGITUDE'] + 20)

ax = asilib.create_cartopy_map(map_style='green', lon_bounds=lon_bounds, lat_bounds=lat_bounds)
asilib.animate_map(asi_array_code, location_code, time_range, 110, overwrite=True, ax=ax)

print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')
