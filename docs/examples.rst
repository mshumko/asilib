========
Examples
========

This example gallery using the best practices and illustrates functionality throughout `asilib`. 

Fisheye Lens View of an Arc
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./_static/fisheye_image_arc.png
    :alt: A fisheye lens view of an auroral arc.
    :width: 75%

    A bright auroral arc that was analyzed by Imajo et al. 2021 "Active auroral arc powered by accelerated electrons from very high altitudes"

.. code:: python

    from datetime import datetime

    import matplotlib.pyplot as plt

    import asilib

    asi_array_code = 'THEMIS'
    location_code = 'RANK'
    time = datetime(2017, 9, 15, 2, 34, 0)

    # A bright auroral arc that was analyzed by Imajo et al., 2021 "Active
    # auroral arc powered by accelerated electrons from very high altitudes"
    image_time, image, ax, im = asilib.plot_fisheye(
        asi_array_code, location_code, time, color_norm='log', force_download=False
    )
    plt.colorbar(im)
    ax.axis('off')
    plt.show()


STEVE projected onto a map
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./_static/map_steve.png
    :alt: STEVE mapped onto a map.
    :width: 75%

    Maps an image of STEVE (the thin band). Reproduced from http://themis.igpp.ucla.edu/nuggets/nuggets_2018/Gallardo-Lacourt/fig2.jpg Note: cartopy takes a few moments to make the necessary coordinate transforms.

.. code:: python

    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    import asilib

    # Create a custom map subplot from a satellite's perspective.
    sat_lat = 54
    sat_lon = -112
    sat_altitude_km = 500

    fig = plt.figure(figsize=(7, 7))
    projection = ccrs.NearsidePerspective(sat_lon, sat_lat, satellite_height=1000*sat_altitude_km)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.add_feature(cfeature.LAND, color='green')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(linestyle=':')

    image_time, image, skymap, ax, p = asilib.plot_map(
            'THEMIS', 'ATHA', datetime(2010, 4, 5, 6, 7, 0), 110, ax=ax
        )
    plt.tight_layout()
    plt.show()


Auroral arc projected onto a map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./_static/map_arc.png
    :alt: A breakup of an auroral arc projected onto a map of North America.
    :width: 75%

    The first breakup of an auroral arc during a substorm analyzed by Donovan et al. 2008 "Simultaneous THEMIS in situ and auroral observations of a small
    substorm"

.. code:: python

    from datetime import datetime

    import matplotlib.pyplot as plt

    import asilib

    time = datetime(2007, 3, 13, 5, 8, 45)
    asi_array_code = 'THEMIS'
    location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
    map_alt = 110
    min_elevation = 2

    # At this time asilib doesn't have an intuitive way to map multiple ASI images, so you need
    # to plot the first imager, and reuse the retuned subplot map to plot the other images.
    image_time, image, skymap, ax, pcolormesh_obj = asilib.plot_map(
        asi_array_code, location_codes[0], time, map_alt, 
        map_style='green', min_elevation=min_elevation)

    for location_code in location_codes[1:]:
        asilib.plot_map(asi_array_code, location_code, time, map_alt, ax=ax, min_elevation=min_elevation)

    ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
    plt.show()


Example 3: A keogram of STEVE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./_static/keogram_steve.png
    :alt: A keogram of STEVE.
    :width: 75%

    A keogram with a STEVE event that moved towards the equator. This event was analyzed in Gallardo-Lacourt et al. 2018 "A statistical analysis of STEVE"

.. code:: python

    import matplotlib.pyplot as plt

    import asilib

    asi_array_code = 'REGO'
    location_code = 'LUCK'
    time_range = ['2017-09-27T07', '2017-09-27T09']
    map_alt_km = 230

    fig, ax = plt.subplots(figsize=(8, 6))
    ax, im = asilib.plot_keogram(
        asi_array_code,
        location_code,
        time_range,
        ax=ax,
        map_alt=map_alt_km,
        color_bounds=(300, 800),
    )
    plt.colorbar(im, label='Intensity')
    ax.set_xlabel('UTC')
    ax.set_ylabel(f'Emission Latitude [deg] at {map_alt_km} km')
    plt.tight_layout()
    plt.show()

Keogram of a field line resonance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./_static/keogram_flr.png
    :alt: A keogram of a field line resonance.
    :width: 75%

    A field line resonance studied in: Gillies, D. M., Knudsen, D., Rankin, R., Milan, S., & Donovan, E. (2018). A statistical survey of the 630.0‐nm optical signature of periodic auroral arcs resulting from magnetospheric field line resonances. Geophysical Research Letters, 45(10), 4648-4655.

.. code:: python

    import matplotlib.pyplot as plt

    import asilib

    asi_array_code = 'REGO'
    location_code = 'GILL'
    time_range = ['2015-02-02T10', '2015-02-02T11']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax, im = asilib.plot_keogram(
        asi_array_code,
        location_code,
        time_range,
        ax=ax,
        map_alt=230,
        pcolormesh_kwargs={'cmap': 'Greys_r'},
    )
    plt.xlabel('Time')
    plt.ylabel('Geographic Latitude [deg]')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


Fisheye Movie
^^^^^^^^^^^^^

.. raw:: html

    <iframe width="75%" height="500"
    src="_static/20150326_060700_062957_themis_fsmi.mp4"; frameborder="0"
    allowfullscreen></iframe>

.. code:: python

    from datetime import datetime

    import asilib

    asi_array_code = 'THEMIS'
    location_code = 'FSMI'
    time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 30))

    asilib.animate_fisheye(asi_array_code, location_code, time_range, overwrite=True)
    print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')


ASI-satellite conjunction movie
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
A comprehensive example that maps a hypothetical satellite track to an image and calculates the mean ASI intensity in a 20x20 km box around the satellite's 100 km altitude footprint.

The `asilib` functionality used here: 

* `asilib.animate_fisheye_generator().send()` to get all the images and image times
* `asilib.lla2azel()` to map the satelites latitude, longitude, altitude (LLA) coordinates to the imager's azimuth and elevation (values and nearest image pixels).
* `asilib.equal_area()` to create a masked array of pixels within a X by Y km sized box at the emission altitude. The masked array is `np.nan` outside of the box and 1 inside.
    
.. raw:: html

    <iframe width="100%", height="900px"
    src="_static/20170915_023300_023457_themis_rank.mp4"
    allowfullscreen></iframe>

.. code:: python

    from datetime import datetime

    import numpy as np
    import matplotlib.pyplot as plt

    import asilib


    # ASI parameters
    asi_array_code = 'THEMIS'
    location_code = 'RANK'
    time_range = (datetime(2017, 9, 15, 2, 32, 0), datetime(2017, 9, 15, 2, 35, 0))

    fig, ax = plt.subplots(
        2, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [4, 1]}, constrained_layout=True
    )

    # Load the skymap calibration data. This is only necessary to create a fake satellite track.
    skymap_dict = asilib.load_skymap(asi_array_code, location_code, time_range[0])

    # Create the fake satellite track coordinates: latitude, longitude, altitude (LLA).
    # This is a north-south satellite track oriented to the east of the THEMIS/RANK
    # imager.
    n = int((time_range[1] - time_range[0]).total_seconds() / 3)  # 3 second cadence.
    lats = np.linspace(skymap_dict["SITE_MAP_LATITUDE"] + 5, skymap_dict["SITE_MAP_LATITUDE"] - 5, n)
    lons = (skymap_dict["SITE_MAP_LONGITUDE"] - 0.5) * np.ones(n)
    alts = 110 * np.ones(n)
    lla = np.array([lats, lons, alts]).T

    # Map the satellite track to the imager's azimuth and elevation coordinates and
    # image pixels. NOTE: the mapping is not along the magnetic field lines! You need
    # to install IRBEM and then use asilib.lla2footprint() before
    # lla2azel() is called.
    sat_azel, sat_azel_pixels = asilib.lla2azel(asi_array_code, location_code, time_range[0], lla)

    # Initiate the movie generator function. Any errors with the data will be raised here.
    movie_generator = asilib.animate_fisheye_generator(
        asi_array_code, location_code, time_range, azel_contours=True, overwrite=True, ax=ax[0]
    )

    # Use the generator to get the images and time stamps to estimate mean the ASI
    # brightness along the satellite path and in a (20x20 km) box.
    image_data = movie_generator.send('data')

    # Calculate what pixels are in a box_km around the satellite, and convolve it
    # with the images to pick out the ASI intensity in that box.
    area_box_mask = asilib.equal_area(
        asi_array_code, location_code, time_range[0], lla, box_km=(20, 20)
    )
    asi_brightness = np.nanmean(image_data.images * area_box_mask, axis=(1, 2))
    area_box_mask[np.isnan(area_box_mask)] = 0  # To play nice with plt.contour()

    for i, (time, image, _, im) in enumerate(movie_generator):
        # Note that because we are drawing different data in each frame (a unique ASI 
        # image in ax[0] and the ASI time series + a guide in ax[1], we need
        # to redraw everything at every iteration.

        ax[1].clear() # ax[0] cleared by asilib.animate_fisheye_generator()
        # Plot the entire satellite track, its current location, and a 20x20 km box 
        # around its location.
        ax[0].plot(sat_azel_pixels[:, 0], sat_azel_pixels[:, 1], 'red')
        ax[0].scatter(sat_azel_pixels[i, 0], sat_azel_pixels[i, 1], c='red', marker='o', s=50)
        ax[0].contour(area_box_mask[i, :, :], levels=[0.99], colors=['yellow'])

        # Plot the time series of the mean ASI intensity along the satellite path
        ax[1].plot(image_data.time, asi_brightness)
        ax[1].axvline(time, c='k')

        # Annotate the location_code and satellite info in the top-left corner.
        location_code_str = (
            f'{asi_array_code}/{location_code} '
            f'LLA=({skymap_dict["SITE_MAP_LATITUDE"]:.2f}, '
            f'{skymap_dict["SITE_MAP_LONGITUDE"]:.2f}, {skymap_dict["SITE_MAP_ALTITUDE"]:.2f})'
        )
        satellite_str = f'Satellite LLA=({lla[i, 0]:.2f}, {lla[i, 1]:.2f}, {lla[i, 2]:.2f})'
        ax[0].text(
            0,
            1,
            location_code_str + '\n' + satellite_str,
            va='top',
            transform=ax[0].transAxes,
            color='red',
        )
        ax[1].set(xlabel='Time', ylabel='Mean ASI intensity [counts]')

    print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "movies"}')