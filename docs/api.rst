=============
API Reference
=============

`asilib` saves all of the ASI image files, skymap calibration files, and movies to 

.. code-block:: python

   asilib.config['ASI_DATA_DIR'] 

By default this directory is set to `~/asilib-data/`, but you can configure the paths using the prompt opened by

.. code-block:: shell

   python3 -m asilib config


The below functions can be imported and called using either of the following ways:

- .. code-block:: python

    import asilib
    asilib.load_image(...)

- .. code-block:: python

    from asilib.io.load import load_image
    load_image(...)

The former option is possible because these functions are all imported by default. However, this may change in a future release, so absolute import (shown in the latter example) is preferred.

.. note::
    `asilib` is hierarchically structured so that the `plot` functions call the `load` functions that then call the `download` functions if a file does not exist locally, or if `force_download=True`. Therefore, **you don't normally need to call the download functions unless you need to download data in bulk.**

.. note::
    The longitude units are converted from 0->360 to -180->180 degrees in the skymap calibration files.

Function Summary
================

.. autosummary::
   :nosignatures:

   asilib.io.download.download_image
   asilib.io.download.download_skymap
   asilib.io.load.load_skymap
   asilib.io.load.load_image
   asilib.io.load.load_image_generator
   asilib.plot.plot_keogram.plot_keogram 
   asilib.plot.plot_fisheye.plot_fisheye
   asilib.plot.plot_map.plot_map
   asilib.plot.animate_fisheye.animate_fisheye
   asilib.plot.animate_fisheye.animate_fisheye_generator
   asilib.plot.animate_map.animate_map
   asilib.plot.animate_map.animate_map_generator
   asilib.analysis.keogram.keogram
   asilib.analysis.map.lla2azel
   asilib.analysis.map.lla2footprint
   asilib.analysis.equal_area.equal_area


Download
========

.. automodule:: asilib.io.download
   :members: download_image, download_skymap
   :undoc-members:
   :show-inheritance:

Load
====
The following functions are very useful if you want to work with the raw image and skymap data without dealing without explicitly downloading them.

.. automodule:: asilib.io.load
   :members: load_skymap, load_image, load_image_generator
   :undoc-members:
   :show-inheritance:

Plot
====

.. automodule:: asilib.plot.plot_keogram
   :members: plot_keogram
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.plot.plot_fisheye
   :members: plot_fisheye
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.plot.plot_map
   :members: plot_map, create_cartopy_map
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.plot.animate_fisheye
   :members: animate_fisheye, animate_fisheye_generator
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.plot.animate_map
   :members: animate_map, animate_map_generator
   :undoc-members:
   :show-inheritance:


Analysis
========

.. automodule:: asilib.analysis.keogram
   :members: keogram
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.analysis.map
   :members: lla2azel, lla2footprint
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.analysis.equal_area
   :members: equal_area
   :undoc-members:
   :show-inheritance: