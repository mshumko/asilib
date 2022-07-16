====================
Imager API Reference
====================

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

   asilib.imager.Imager
   asilib.conjunction.Conjunction
   asilib.array.themis.themis
   asilib.array.lamp_emccd.lamp
   asilib.array.lamp_phantom.lamp


Imager
======

.. automodule:: asilib.imager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.conjunction
   :members:
   :undoc-members:
   :show-inheritance:

Supported ASI systems (arrays)
==============================

THEMIS
^^^^^^
.. automodule:: asilib.array.themis
   :members:
   :undoc-members:
   :show-inheritance:

LAMP
^^^^

.. automodule:: asilib.array.lamp_emccd
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.array.lamp_phantom
   :members:
   :undoc-members:
   :show-inheritance: