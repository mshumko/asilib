![Test python package](https://github.com/mshumko/aurora-asi-lib/workflows/Test%20python%20package/badge.svg) ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746447.svg)

# aurora-asi-lib
Easily download, plot, animate, and analyze aurora all sky imager (ASI) data. Currently the two supported camera systems (missions) are: 
* Red-line Emission Geospace Observatory (REGO)
* Time History of Events and Macroscale Interactions during Substorms (THEMIS).

[API Documentation](https://aurora-asi-lib.readthedocs.io/) / [Code on GitHub](https://github.com/mshumko/aurora-asi-lib)


Easily make ASI plots (example 1)!

![Aurora plot from example 1.](https://github.com/mshumko/aurora-asi-lib/blob/main/images/example_1.png?raw=true)

And movies! (example4; the track and mean ASI intensity plot is a little bit more work.)
![Aurora movie from example 4.](https://github.com/mshumko/aurora-asi-lib/blob/main/images/20170915_023400_023557_themis_rank.gif?raw=true)

Feel free to contact me and request that I add other ASI missions to `asilib`.

## Examples
Before you can run these examples, make sure that `asilib` is configured with the installation steps below. These examples, and more, are in the `asilib/examples/` folder.

### Example 1
This example uses asilib to plot one frame of a bright auroral arc.
```python
from datetime import datetime

import matplotlib.pyplot as plt

import asilib

# A bright auroral arc that was analyzed by Imajo et al., 2021 "Active 
# auroral arc powered by accelerated electrons from very high altitudes"
frame_time, ax, im = asilib.plot_frame(datetime(2017, 9, 15, 2, 34, 0), 'THEMIS', 'RANK', 
                    color_norm='log', force_download=False)
plt.colorbar(im)
ax.axis('off')
plt.show()
```

### Example 2
This example uses asilib to plot a 5 minute movie of an auroral arc brightening right as a meteor burns up at zenith!

```python
from datetime import datetime

import asilib

time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
asilib.plot_movie(time_range, 'THEMIS', 'FSMI')
print(f'Movie saved in {asilib.config.ASI_DATA_DIR / "movies"}')
```

### Example 3
This example is longer and it shows how to superpose a hypothetical satellite path through the THEMIS camera located at Rankin Inlet.

```python
from datetime import datetime

import numpy as np

from asilib import plot_movie_generator
from asilib import lla2azel
from asilib import load_cal


# ASI parameters
mission = 'THEMIS'
station = 'RANK'
time_range = (datetime(2017, 9, 15, 2, 34, 0), datetime(2017, 9, 15, 2, 36, 0))

# Load the calibration data.
cal_dict = load_cal(mission, station)

# Create the satellite track's latitude, longitude, altitude (LLA) coordinates.
# This is an imaginary north-south satellite track oriented to the east
# of the THEMIS/RANK station.
n = int((time_range[1] - time_range[0]).total_seconds() / 3)  # 3 second cadence.
lats = np.linspace(cal_dict["SITE_MAP_LATITUDE"] + 10, cal_dict["SITE_MAP_LATITUDE"] - 10, n)
lons = (cal_dict["SITE_MAP_LONGITUDE"] + 3) * np.ones(n)
alts = 500 * np.ones(n)
lla = np.array([lats, lons, alts]).T

# Map the satellite track to the station's azimuth and elevation coordinates as well as the
# image pixels
# The mapping is not along the magnetic field lines! You need to install IRBEM and then use
# asilib.trace_field_line().
sat_azel, sat_azel_pixels = lla2azel(mission, station, lla)

# Initiate the movie generator function.
movie_generator = plot_movie_generator(
    time_range, mission, station, azel_contours=True, overwrite=True
)

for i, (time, frame, ax, im) in enumerate(movie_generator):
    # Plot the entire satellite track
    ax.plot(sat_azel_pixels[:, 0], sat_azel_pixels[:, 1], 'red')
    # Plot the current satellite position.
    ax.scatter(sat_azel_pixels[i, 0], sat_azel_pixels[i, 1], c='red', marker='x', s=100)

    # Annotate the station and satellite info in the top-left corner.
    station_str = (
        f'{mission}/{station} '
        f'LLA=({cal_dict["SITE_MAP_LATITUDE"]:.2f}, '
        f'{cal_dict["SITE_MAP_LONGITUDE"]:.2f}, {cal_dict["SITE_MAP_ALTITUDE"]:.2f})'
    )
    satellite_str = f'Satellite LLA=({lla[i, 0]:.2f}, {lla[i, 1]:.2f}, {lla[i, 2]:.2f})'
    ax.text(0, 1, station_str + '\n' + satellite_str, va='top', 
            transform=ax.transAxes, color='red')
```

## Installation
To install this package as a user, run:

```shell
python3 -m pip install aurora-asi-lib
```

To install this package as a developer, run:

```shell
git clone git@github.com:mshumko/aurora-asi-lib.git
cd aurora-asi-lib
python3 -m pip install -r requirements.txt # or
python3 -m pip install -e .
```


In either case, you'll need to configure your system paths to tell `asilib` (the import name) where to save the ASI data and movies. Run ```python3 -m asilib config``` to set up the data directory where the image, calibration, and movie files will be saved. Your settings will be stored in `config.py`. If you configure `asilib`, but don't specify a data directory, a default directory in `~/asilib-data` will be created if it doesn't exist.

### ffmpeg dependency
To make  movies you'll also need to install the ffmpeg library.
 - **Ubuntu**: ```apt install ffmpeg```
 - **Mac**: ```brew install ffmpeg```

__NOTES__
- If you get the "ERROR: Could not build wheels for pymap3d which use PEP 517 and cannot be installed directly" error when installing, you need to upgrade your pip, setuptools, and wheel libaries via ```python3 -m pip install --upgrade pip setuptools wheel```.