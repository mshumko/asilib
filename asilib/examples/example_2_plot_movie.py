from datetime import datetime

import asilib

time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 30))
asilib.plot_movie(time_range, 'THEMIS', 'FSMI', overwrite=True)
print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')
