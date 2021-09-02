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
    asilib.get_frame(...)

- .. code-block:: python

    from asilib.io.load import get_frame
    get_frame(...)

The former option is possible because these functions are all imported by default. However, this may change in a future release, so absolute import (shown in the latter example) is preferred.

.. note::
    `asilib` is hierarchically structured so that the `plot` functions call the `load` functions that then call the `download` functions if a file does not exist locally, or if `force_download=True`. Therefore, **you don't normally need to call the download functions unless you need to download data in bulk.**

.. note::
    The longitude units are converted from 0->360 to -180->180 degrees in the skymap calibration files.

Summary
=======

.. autosummary::
    asilib.io.download_themis.download_themis_img
    asilib.io.download_themis.download_themis_cal
    asilib.io.download_rego.download_rego_img
    asilib.io.download_rego.download_rego_cal
    asilib.io.load.load_skymap
    asilib.io.load.get_frame
    asilib.io.load.get_frames
    asilib.plot.plot_keogram.plot_keogram 
    asilib.plot.plot_frame.plot_frame
    asilib.plot.plot_map.plot_map
    asilib.plot.plot_movie.plot_movie
    asilib.plot.plot_movie.plot_movie_generator
    asilib.analysis.keogram.keogram
    asilib.analysis.map.lla2azel
    asilib.analysis.map.lla2footprint
    asilib.analysis.equal_area.equal_area

Imager
======

.. note::

    COMING SOON: All of the asilib functionality will soon be combined into an Imager() class.

Download
========

.. automodule:: asilib.io.download_themis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.io.download_rego
   :members: download_rego_img, download_rego_cal
   :undoc-members:
   :show-inheritance:

Load
====
The following functions are very useful if you want to work with the raw image and skymap data without dealing without explicitly downloading them.

.. automodule:: asilib.io.load
   :members: load_skymap, get_frame, get_frames
   :undoc-members:
   :show-inheritance:

Plot
====

.. automodule:: asilib.plot.plot_keogram
   :members: plot_keogram
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.plot.plot_frame
   :members: plot_frame
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.plot.plot_map
   :members: plot_map
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.plot.plot_movie
   :members: plot_movie, plot_movie_generator
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