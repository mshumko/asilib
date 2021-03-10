from datetime import datetime

import asilib

time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
asilib.plot_movie(time_range, 'THEMIS', 'FSMI')
print(f'Movie saved in {asilib.config.ASI_DATA_DIR / "movies"}')