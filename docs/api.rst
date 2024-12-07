.. _api:

=============
API Reference
=============

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
.. automodule:: asilib.asi
   :members: themis, themis_info, themis_skymap
   :undoc-members:
   :show-inheritance:

.. _rego_asi:

Red-line Emission Geospace Observatory (REGO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: asilib.asi
   :members: rego, rego_info, rego_skymap
   :undoc-members:
   :show-inheritance:

.. _trex_asi:

Transition Region Explorer (TREx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: asilib.asi
   :members: trex_rgb, trex_rgb_info, trex_rgb_skymap, trex_nir, trex_nir_skymap, trex_nir_info
   :undoc-members:
   :show-inheritance:

.. _mango_asi:

Mid-latitude All-sky-imaging Network for Geophysical Observations (MANGO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: asilib.asi
   :members: mango, mango_info
   :undoc-members:
   :show-inheritance:

Class Interface
===============

.. automodule:: asilib
   :members:
   :undoc-members:
   :show-inheritance:

Geographic Maps
===============

.. automodule:: asilib.map
   :members: create_map, create_cartopy_map, create_simple_map
   :undoc-members:
   :show-inheritance: