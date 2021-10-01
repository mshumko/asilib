![Test python package](https://github.com/mshumko/aurora-asi-lib/workflows/Test%20python%20package/badge.svg) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746447.svg)](https://doi.org/10.5281/zenodo.4746446)

# aurora-asi-lib
Easily download, plot, animate, and analyze aurora all sky imager (ASI) data. Currently the two supported camera systems (missions) are: 
* Red-line Emission Geospace Observatory (REGO)
* Time History of Events and Macroscale Interactions during Substorms (THEMIS).

[API Documentation](https://aurora-asi-lib.readthedocs.io/) / [Code on GitHub](https://github.com/mshumko/aurora-asi-lib)


Easily make ASI plots (example 1)!

![Aurora plot from example 1.](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/example_1.png?raw=true)

And movies! (example4; the track and mean ASI intensity plot is a little bit more work.)
![Aurora movie from example 4.](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/20170915_023400_023557_themis_rank.gif?raw=true)

Feel free to contact me and request that I add other ASI missions to `asilib`.

## Examples
Before you can run these examples, make sure that `asilib` is configured with the installation steps below. These examples, and more, are in the `asilib/examples/` folder.

### Example 1
This example uses asilib to plot one image of a bright auroral arc.
```python
from datetime import datetime

import matplotlib.pyplot as plt

import asilib

# A bright auroral arc that was analyzed by Imajo et al., 2021 "Active 
# auroral arc powered by accelerated electrons from very high altitudes"
image_time, image, ax, im = asilib.plot_image(datetime(2017, 9, 15, 2, 34, 0), 'THEMIS', 'RANK', 
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
import matplotlib.pyplot as plt

import asilib


# ASI parameters
mission = 'THEMIS'
station = 'RANK'
time_range = (datetime(2017, 9, 15, 2, 32, 0), datetime(2017, 9, 15, 2, 35, 0))

fig, ax = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={'height_ratios':[4, 1]}, 
                        constrained_layout=True)

# Load the calibration data. This is only necessary to create a fake satellite track.
skymap_dict = asilib.load_skymap(mission, station, time_range[0])

# Create the fake satellite track coordinates: latitude, longitude, altitude (LLA).
# This is a north-south satellite track oriented to the east of the THEMIS/RANK 
# station.
n = int((time_range[1] - time_range[0]).total_seconds() / 3)  # 3 second cadence.
lats = np.linspace(skymap_dict["SITE_MAP_LATITUDE"] + 5, skymap_dict["SITE_MAP_LATITUDE"] - 5, n)
lons = (skymap_dict["SITE_MAP_LONGITUDE"]-0.5) * np.ones(n)
alts = 110 * np.ones(n)
lla = np.array([lats, lons, alts]).T

# Map the satellite track to the station's azimuth and elevation coordinates and
# image pixels. NOTE: the mapping is not along the magnetic field lines! You need
# to install IRBEM and then use asilib.lla2footprint() before 
# lla2azel() is called.
sat_azel, sat_azel_pixels = asilib.lla2azel(mission, station, time_range[0], lla)

# Initiate the movie generator function. Any errors with the data will be raised here.
movie_generator = asilib.plot_movie_generator(
    time_range, mission, station, azel_contours=True, overwrite=True,
    ax=ax[0]
)

# Use the generator to get the images and time stamps to estimate mean the ASI
# brightness along the satellite path and in a (10x10 km) box.
image_data = movie_generator.send('data')

# Calculate what pixels are in a box_km around the satellite, and convolve it
# with the images to pick out the ASI intensity in that box.
area_box_mask = asilib.equal_area(mission, station, time_range[0], lla, box_km=(20, 20))
asi_brightness = np.nanmean(image_data.images*area_box_mask, axis=(1,2))
area_box_mask[np.isnan(area_box_mask)] = 0  # To play nice with plt.contour()

for i, (time, image, _, im) in enumerate(movie_generator):
    # Note that because we are drawing moving data: ASI image in ax[0] and 
    # the ASI time series + a vertical bar at the image time in ax[1], we need
    # to redraw everything at every iteration.
     
    # Clear ax[1] (ax[0] cleared by asilib.plot_movie_generator())
    ax[1].clear()
    # Plot the entire satellite track
    ax[0].plot(sat_azel_pixels[:, 0], sat_azel_pixels[:, 1], 'red')
    ax[0].contour(area_box_mask[i, :, :], levels=[0.99], colors=['yellow'])
    # Plot the current satellite position.
    ax[0].scatter(sat_azel_pixels[i, 0], sat_azel_pixels[i, 1], c='red', marker='o', s=50)

    # Plot the time series of the mean ASI intensity along the satellite path
    ax[1].plot(image_data.time, asi_brightness)
    ax[1].axvline(time, c='k') # At the current image time.

    # Annotate the station and satellite info in the top-left corner.
    station_str = (
        f'{mission}/{station} '
        f'LLA=({skymap_dict["SITE_MAP_LATITUDE"]:.2f}, '
        f'{skymap_dict["SITE_MAP_LONGITUDE"]:.2f}, {skymap_dict["SITE_MAP_ALTITUDE"]:.2f})'
    )
    satellite_str = f'Satellite LLA=({lla[i, 0]:.2f}, {lla[i, 1]:.2f}, {lla[i, 2]:.2f})'
    ax[0].text(0, 1, station_str + '\n' + satellite_str, va='top', 
            transform=ax[0].transAxes, color='red')
    ax[1].set(xlabel='Time', ylabel='Mean ASI intensity [counts]')

print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')
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


In either case, you'll need to configure your system paths to tell `asilib` (the import name) where to save the ASI data and movies. Run ```python3 -m asilib config``` to set up the data directory where the image, skymap, and movie files will be saved. Your settings will be stored in `config.py`. If you configure `asilib`, but don't specify a data directory, a default directory in `~/asilib-data` will be created if it doesn't exist.

### ffmpeg dependency
To make  movies you'll also need to install the ffmpeg library.
 - **Ubuntu**: ```apt install ffmpeg```
 - **Mac**: ```brew install ffmpeg```

__NOTES__
- If you get the "ERROR: Could not build wheels for pymap3d which use PEP 517 and cannot be installed directly" error when installing, you need to upgrade your pip, setuptools, and wheel libaries via ```python3 -m pip install --upgrade pip setuptools wheel```.
