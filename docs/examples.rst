========
Examples
========

This example gallery using the best practices and illustrates functionality throughout `asilib`. 

Example 1: Fisheye Lens View of an Arc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./_static/example_1.png

    A bright auroral arc that was analyzed by Imajo et al., 2021 "Active auroral arc powered by accelerated electrons from very high altitudes"

.. code:: python

    from datetime import datetime

    import matplotlib.pyplot as plt

    import asilib

    frame_time, frame, ax, im = asilib.plot_frame(datetime(2017, 9, 15, 2, 34, 0), 
        'THEMIS', 'RANK', color_norm='log', force_download=False)
    plt.colorbar(im)
    ax.axis('off')
    plt.show()

Example 2: A keogram
^^^^^^^^^^^^^^^^^^^^
.. figure:: ./_static/example_2.png

    A keogram with a STEVE event that moved towards the equator. This event was analyzed in Gallardo-Lacourt et al. 2018 "A statistical analysis of STEVE"

.. code:: python

    import matplotlib.pyplot as plt

    import asilib

    mission='REGO'
    station='LUCK'
    map_alt_km = 230

    fig, ax = plt.subplots(figsize=(8, 6))
    ax, im = asilib.plot_keogram(['2017-09-27T07', '2017-09-27T09'], mission, station, 
                    ax=ax, map_alt=map_alt_km, color_bounds=(300, 800))
    plt.colorbar(im, label='Intensity')
    ax.set_xlabel('UTC')
    ax.set_ylabel(f'Emission Latitude [deg] at {map_alt_km} km')
    plt.tight_layout()
    plt.show()

Example 2: Fisheye Movie
^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <iframe width="560" height="315"
    src="_static/20150326_060700_062957_themis_fsmi.mp4"; frameborder="0"
    allowfullscreen></iframe>


.. .. video:: ./_static/20150326_060700_062957_themis_fsmi.mp4
..     :width: 500
..     :height: 300
..     :autoplay: