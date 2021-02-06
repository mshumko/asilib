# asi-lib
Easily download, plot, animate, and analyze aurora all sky imager (ASI) data. Currently the two supported camera systems (missions) are: 
* Red-line Emission Geospace Observatory (REGO)
* Time History of Events and Macroscale Interactions during Substorms (THEMIS).

Feel free to contact me to add other ASI missions to this library.

## Examples
Before you can run these examples you will need to configure the ASI data and movie directory via ```python3 -m asilib config``` (see the installation steps below).

### Example 1
This example uses asilib to visualize a bright auroral arc.
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
This example uses asilib to make a 5 minute movie of an auroral arc brightening right as a meteor burns up at zenith!

```python
from datetime import datetime

import asilib

time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
asilib.plot_movie(time_range, 'THEMIS', 'FSMI')
```

## Installation
Run these terminal commands to install the dependencies into a virtual environment and configure the data paths:

```shell
# cd into the asi-lib (not the child asilib) directory
python3 -m venv env
source env/bin/activate

python3 -m pip install .  # (don't forget the .)
#  or 
pip3 install -r requirements.txt
```

You'll need to configure your system paths to tell asilib where to save the ASI data and movies. Run ```python3 -m asilib config``` to set up the data directory where the image, calibration, and movie files will be saved. Your settings wll be stored in `config.py`

### ffmpeg dependency
To make  movies you'll also need to install the ffmpeg library.
 - **Ubuntu**: ```apt install ffmpeg```
 - **Mac**: ```brew install ffmpeg```

## User Guide
**Note** These are top-level descriptions: The full parameter list and an example for each function is accessible via the built-in ```help()``` function. 

**Note** The longitude units are converted from 0->360 to -180->180 degrees in the calibration files.

### Download ASI Data
To download ASI data, the programs in the ```asi-lib/download/``` search for and download the THEMIS and REGO image and calibration data.

* `asilib.download_themis_img()` and `asilib.download_rego_img()`: downloads the THEMIS and REGO images in the common data format (CDF) files.
* `asilib.download_themis_cal()` and `asilib.download_rego_cal()`: downloads the THEMIS and REGO images in the common data format (CDF) files.

### Load ASI Data
There are a few data loading functions that automaticaly call the download programs if a file is not found on the local computer or the user explicitly passes ```force_download = True``` to force the download. These functions are in `config.py`.

* `asilib.load_img_file()`: Returns an `cdflib.CDF()` file object for an ASI
file specified by the date-time, mission, and station. See the [cdflib](https://github.com/MAVENSDC/cdflib) documentaion for the CDF interface.
* `asilib.load_cal_file()`: Returns an dictionary containing the latest calibration data from a specified mission/station. Be aware that the longitude is mapped from 0 to 360 to -180 to 180 degrees.
* `asilib.get_frame()`: Given a mission/station and a date-time, this function calls `asilib.load_img_file()` and returns the time stamp and one image (frame) with a time stamp within ```time_thresh_s = 3``` seconds (optional kwarg), otherwise an AssertionError is raised if a ASI time stamp is not found.
* `asilib.get_frames()`: Given a mission/station and a date-time ```time_range```, this function calls `asilib.load_img_file()` and returns an array of time stamps and images observed at times inside the ```time_range```.

### Plot ASI Data
There are two modules that plot a single frame or a series of frames.

* `asilib.plot_frame()`: Given a mission/station and a date-time arguments, this function calls `asilib.get_frame()` and plots one ASI frame. By default, the color map is black-white for THEMIS and black-red for REGO, the color scale is logarthmic, and color map limits are automatically set as ```(25th percentile, min(98th percentile, 10x25th percentile))```. This ensures a good dynamic range for each frame. The subplot object, the frame time, and the ```plt.imshow()``` objects are returned so the user can add to the subplot.

* `asilib.plot_movie()`: Similar to `asilib.plot_frame()`, given a mission/station and a ```time_range``` arguments, this function calls `asilib.get_frames()` and plots one multiple ASI frames and saves them to ```/data/movies/```. Movie file creation, such as an `mp4` or `gif`, is not implemented yet because I have not found a movie writer that is available between Windows/Linux/Mac.

* `plot_movie_generator()` TBD

* `plot_collage()`: Similar to `asilib.plot_movie()` in that the arguments are the same, but this function returns a collage of images with the time stamps annotated.

### Mapping satellite position to the skyfield
* `asilib.map_skyfield()`: maps the satellite coordinates from LLA (latitude, longitude, altitudes) to the ASI image x and y pixel indices. This function relies on the azimuth and elevation calibration files that can be downloaded via `asilib.load_cal_file()`. This function does **not** map the satellite position along the magnetic field line, that is done by `map_along_magnetic_field.py` and requires IRBEM-Lib to be installed (beyond the scope of this user guide).
* `map_along_magnetic_field.py`: magnetically maps the satellite LLA coordinates with time stamps to a specified altitude. The hemisphere of the mapping can be: same, opposite, northern, or southern. 

## Testing
Each module has a corresponding `test_module.py` module in ```asilib/tests/```. Run these tests to confirm that the downloading, loading, plotting, and mapping functions work correctly. If a test fails, please submit an Issue. To help me fix the bug, please run the unit tests in verbose mode, i.e. ```python3 test_module.py -v```.
