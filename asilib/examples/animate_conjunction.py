"""
This example script creates a movie with the satellite track superposed.
"""
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import asilib
import asilib.asi


# ASI parameters
location_code = 'RANK'
alt = 110  # km
time_range = (datetime(2017, 9, 15, 2, 32, 0), datetime(2017, 9, 15, 2, 35, 0))

fig, ax = plt.subplots(
    3, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [4, 1, 1]}, constrained_layout=True
)

asi = asilib.asi.themis(location_code, time_range=time_range, alt=alt)

# Create the fake satellite track coordinates: latitude, longitude, altitude (LLA).
# This is a north-south satellite track oriented to the east of the THEMIS/RANK
# imager.
n = int((time_range[1] - time_range[0]).total_seconds() / 3)  # 3 second cadence.
lats = np.linspace(asi.meta["lat"] + 5, asi.meta["lat"] - 5, n)
lons = (asi.meta["lon"] - 0.5) * np.ones(n)
alts = alt * np.ones(n)  # Altitude needs to be the same as the skymap.
sat_lla = np.array([lats, lons, alts]).T
# Normally the satellite time stamps are not the same as the ASI.
# You may need to call Conjunction.interp_sat() to find the LLA coordinates
# at the ASI timestamps.
sat_time = asi.data.time

conjunction_obj = asilib.Conjunction(asi, (sat_time, sat_lla))

# Map the satellite track to the imager's azimuth and elevation coordinates and
# image pixels. NOTE: the mapping is not along the magnetic field lines! You need
# to install IRBEM and then use conjunction.lla_footprint() before
# calling conjunction_obj.map_azel.
sat_azel, sat_azel_pixels = conjunction_obj.map_azel()

# Calculate the auroral intensity near the satellite and mean intensity within a 10x10 km area.
nearest_pixel_intensity = conjunction_obj.intensity(box=None)
area_intensity = conjunction_obj.intensity(box=(10, 10))
area_mask = conjunction_obj.equal_area(box=(10, 10))

# Need to change masked NaNs to 0s so we can plot the rectangular area contours.
area_mask[np.where(np.isnan(area_mask))] = 0

# Initiate the animation generator function.
gen = asi.animate_fisheye_gen(
    ax=ax[0], azel_contours=True, overwrite=True, cardinal_directions='NE'
)

for i, (time, image, _, im) in enumerate(gen):
    # Plot the entire satellite track, its current location, and a 20x20 km box
    # around its location.
    ax[0].plot(sat_azel_pixels[:, 0], sat_azel_pixels[:, 1], 'red')
    ax[0].scatter(sat_azel_pixels[i, 0], sat_azel_pixels[i, 1], c='red', marker='o', s=50)
    ax[0].contour(area_mask[i, :, :], levels=[0.99], colors=['yellow'])

    if 'vline1' in locals():
        vline1.remove()  # noqa: F821
        vline2.remove()  # noqa: F821
        text_obj.remove()  # noqa: F821
    else:
        # Plot the ASI intensity along the satellite path
        ax[1].plot(sat_time, nearest_pixel_intensity)
        ax[2].plot(sat_time, area_intensity)
    vline1 = ax[1].axvline(time, c='b')
    vline2 = ax[2].axvline(time, c='b')

    # Annotate the location_code and satellite info in the top-left corner.
    location_code_str = (
        f'THEMIS/{location_code} '
        f'LLA=({asi.meta["lat"]:.2f}, '
        f'{asi.meta["lon"]:.2f}, {asi.meta["alt"]:.2f})'
    )
    satellite_str = f'Satellite LLA=({sat_lla[i, 0]:.2f}, {sat_lla[i, 1]:.2f}, {sat_lla[i, 2]:.2f})'
    text_obj = ax[0].text(
        0,
        1,
        location_code_str + '\n' + satellite_str,
        va='top',
        transform=ax[0].transAxes,
        color='red',
    )
    ax[1].set(ylabel='ASI intensity\nnearest pixel [counts]')
    ax[2].set(xlabel='Time', ylabel='ASI intensity\n10x10 km area [counts]')

print(f'Animation saved in {asilib.config["ASI_DATA_DIR"] / "animations" / asi.animation_name}')
