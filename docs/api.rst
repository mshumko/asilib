=============
API Reference
=============

The below functions can be imported and called using the following two preferred ways:

- .. code-block:: python

    import asilib
    asilib.get_frame(...)

- .. code-block:: python

    from asilib.io.load import get_frame
    get_frame(...)

The former option is possible because these functions are all imported by default. However, this may change in a future release, so absolute import shown in the latter example is preferred.

.. note::
    `asilib` is hierarchically structured so that the `plot` functions call the `load` functions that then call the `download` functions if a file does not exist locally, or if `force_download=True`. Therefore, **you don't normally need to call the download functions unless you need to download data in bulk.**

Summary
=======

.. autosummary::
    asilib.io.download_themis.download_themis_img
    asilib.io.download_themis.download_themis_cal
    asilib.io.download_rego.download_rego_img
    asilib.io.download_rego.download_rego_cal
    asilib.io.load.load_img
    asilib.io.load.load_cal
    asilib.io.load.get_frame
    asilib.io.load.get_frames
    asilib.plot.plot_keogram.plot_keogram 
    asilib.plot.plot_frame.plot_frame
    asilib.plot.plot_movie.plot_movie
    asilib.plot.plot_movie.plot_movie_generator

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
The following functions are very useful if you want to work with the raw image and calibration data without dealing without explicitly downloading them.

.. automodule:: asilib.io.load
   :members: load_img, load_cal, get_frame, get_frames
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

.. automodule:: asilib.plot.plot_movie
   :members: plot_movie, plot_movie_generator
   :undoc-members:
   :show-inheritance:

Analysis
========