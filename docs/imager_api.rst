.. _api:

====================
Imager API Reference
====================

`asilib` saves all of the ASI image files, skymap calibration files, and movies to 

.. code-block:: python

   asilib.config['ASI_DATA_DIR'] 

By default this directory is set to `~/asilib-data/`, but you can configure the paths using the prompt opened by

.. code-block:: shell

   python3 -m asilib config


.. note::
    The longitude units are converted from [0, 360] to [-180, 180] degrees in the skymap calibration files.

ASI arrays
==========

.. _themis_asi:

Time History of Events and Macroscale Interactions during Substorms (THEMIS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: asilib.asi.themis
   :members:
   :undoc-members:
   :show-inheritance:

.. _rego_asi:

Red-line Emission Geospace Observatory (REGO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: asilib.asi.rego
   :members:
   :undoc-members:
   :show-inheritance:

.. _trex_asi:

Transition Region Explorer (TREx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: asilib.asi.trex
   :members:
   :undoc-members:
   :show-inheritance:

Imager Interface
================

.. automodule:: asilib.imager
   :members:
   :undoc-members:
   :show-inheritance:


Conjunctions with Satellites
============================
.. automodule:: asilib.conjunction
   :members:
   :undoc-members:
   :show-inheritance:

Geographic Maps
===============

.. automodule:: asilib.map
   :members: create_map, create_cartopy_map, create_simple_map
   :undoc-members:
   :show-inheritance: