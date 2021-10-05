"""
This example script creates a movie with the satellite track superposed.
"""
from datetime import datetime

import numpy as np

import asilib
from asilib import plot_movie_generator
from asilib import lla2azel
from asilib import load_skymap

# ASI parameters
asi_array_code = 'THEMIS'
location_code = 'RANK'
time_range = (datetime(2017, 9, 15, 2, 34, 0), datetime(2017, 9, 15, 2, 36, 0))

# Load the skymap data.
skymap_dict = load_skymap(asi_array_code, location_code, time_range[0])

# Create the satellite track's latitude, longitude, altitude (LLA) coordinates.
# This is an imaginary north-south satellite track oriented to the east
# of the THEMIS/RANK imager.
n = int((time_range[1] - time_range[0]).total_seconds() / 3)  # 3 second cadence.
lats = np.linspace(skymap_dict["SITE_MAP_LATITUDE"] + 10, skymap_dict["SITE_MAP_LATITUDE"] - 10, n)
lons = (skymap_dict["SITE_MAP_LONGITUDE"] + 3) * np.ones(n)
alts = 500 * np.ones(n)
lla = np.array([lats, lons, alts]).T

# Map the satellite track to the imager's azimuth and elevation coordinates as well as the
# image pixels
# The mapping is not along the magnetic field lines! You need to install IRBEM and then use
# asilib.lla2footprint().
sat_azel, sat_azel_pixels = lla2azel(asi_array_code, location_code, time_range[0], lla)

# Initiate the movie generator function.
movie_generator = plot_movie_generator(
    time_range, asi_array_code, location_code, azel_contours=True, overwrite=True
)

for i, (time, image, ax, im) in enumerate(movie_generator):
    # Plot the entire satellite track
    ax.plot(sat_azel_pixels[:, 0], sat_azel_pixels[:, 1], 'red')
    # Plot the current satellite position.
    ax.scatter(sat_azel_pixels[i, 0], sat_azel_pixels[i, 1], c='red', marker='x', s=100)

    # Annotate the location_code and satellite info in the top-left corner.
    location_code_str = (
        f'{asi_array_code}/{location_code} '
        f'LLA=({skymap_dict["SITE_MAP_LATITUDE"]:.2f}, '
        f'{skymap_dict["SITE_MAP_LONGITUDE"]:.2f}, {skymap_dict["SITE_MAP_ALTITUDE"]:.2f})'
    )
    satellite_str = f'Satellite LLA=({lla[i, 0]:.2f}, {lla[i, 1]:.2f}, {lla[i, 2]:.2f})'
    ax.text(
        0,
        1,
        location_code_str + '\n' + satellite_str,
        va='top',
        transform=ax.transAxes,
        color='red',
    )

print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')
