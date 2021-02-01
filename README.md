# aurora_asi
This project downloads and analyzes the aurora all sky imager (ASI) data. The two supported camera systems (missions) are: Red-line Emission Geospace Observatory (REGO) and Time History of Events and Macroscale Interactions during Substorms (THEMIS).

## Installation
Run these shell commands to install the dependencies into a virtual environment and configure the data paths:

```
# cd into the top directory
python3 -m venv env
source env/bin/activate

python3 -m pip install -e .  # (don't forget the .)
#  or 
pip3 install -r requirements.txt
```

## User Guide
This package needs to be configured before you download or analyze ASI data. After the package is installed, run ```python3 -m asi config``` to set up the data directory where the image, calibration, and movie files will be saved.

**Note** These are top-level descriptions: The full parameter list and an example for each function is accessible via the built-in ```help()``` function. 

### Download ASI Data
To download ASI data, the programs in the ```asi/download/``` search for and download the THEMIS and REGO image and calibration data.

* `asi.download_themis_img()` and `asi.download_rego_img()`: downloads the THEMIS and REGO images in the common data format (CDF) files.
* `asi.download_themis_cal()` and `asi.download_rego_cal()`: downloads the THEMIS and REGO images in the common data format (CDF) files.

### Load ASI Data
There are a few data loading functions that automaticaly call the download programs if a file is not found on the local computer or the user explicitly passes ```force_download = True``` to force the download. These functions are in `config.py`.

* `load_img_file()`: Returns an `cdflib.CDF()` file object for an ASI
file specified by the date-time, mission, and station. See the [cdflib](https://github.com/MAVENSDC/cdflib) documentaion for the CDF interface.
* `load_cal_file()`: Returns an dictionary containing the latest calibration data from a specified mission/station. Be aware that the longitude is mapped from 0 to 360 to -180 to 180 degrees.
* `get_frame()`: Given a mission/station and a date-time, this function calls `load_img_file()` and returns the time stamp and one image (frame) with a time stamp within ```time_thresh_s = 3``` seconds (optional kwarg), otherwise an AssertionError is raised if a ASI time stamp is not found.
* `get_frames()`: Given a mission/station and a date-time ```time_range```, this function calls `load_img_file()` and returns an array of time stamps and images observed at times inside the ```time_range```.


### Plot ASI Data
There are two modules that plot a single aurora frame or 

- Mention that the map_skyfield.py maps the (latitude, longitude, altitudes) coordinates and not the mapped coordinates!  