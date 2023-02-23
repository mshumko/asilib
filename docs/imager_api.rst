====================
Imager API Reference
====================

.. warning::
   The Imager API is under active development. 

`asilib` saves all of the ASI image files, skymap calibration files, and movies to 

.. code-block:: python

   asilib.config['ASI_DATA_DIR'] 

By default this directory is set to `~/asilib-data/`, but you can configure the paths using the prompt opened by

.. code-block:: shell

   python3 -m asilib config


.. note::
    The longitude units are converted from [0, 360] to [-180, 180] degrees in the skymap calibration files.

Summary
=======

.. autosummary::
   :nosignatures:

   asilib.imager.Imager
   asilib.conjunction.Conjunction
   asilib.map
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


Geographic Maps
===============

.. automodule:: asilib.map
   :members: create_map, create_cartopy_map, create_simple_map
   :undoc-members:
   :show-inheritance: