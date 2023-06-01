===========
Get Started
=========== 

Install
-------

Installing aurora-asi-lib is as simple as:

.. code-block:: shell

   python3 -m pip install aurora-asi-lib 


Anaconda
^^^^^^^^

`aurora-asi-lib` can also be installed with pip inside Anaconda. In a new environment install scipy first and then install asilib using the above instructions. 


.. note::
   - By default, aurora-asi-lib saves the ASI data, movie images, and movies in the `~/asilib-data/` directory. To override the default directory, run aurora-asi-lib as a module, `python3 -m asilib config`. See the Configuration section below for more details.

   - If you get the "`ERROR: Could not build wheels for pymap3d which use PEP 517 and cannot be installed directly`" error when installing, you need to upgrade your pip, setuptools, and wheel libaries via ```python3 -m pip install --upgrade pip setuptools wheel```.

Dependencies
^^^^^^^^^^^^
There are three optional dependencies that you may want to install if you want to use certain `asilib` functions.

+----------------+------------------------------+--------------------------------------+
| **Dependency** | **Purpose**                  | **asilib methods**                   |
+----------------+---------+--------------------+--------------------------------------+
| ffmpeg         | Animating Images             | | asilib.Imager.animate_fisheye()    |
|                |                              | | asilib.Imager.animate_map()        |
|                |                              | | asilib.Imager.animate_fisheye_gen()|
|                |                              | | asilib.Imager.animate_map_gen()    |
+----------------+------------------------------+--------------------------------------+
| IRBEM          | Magnetic field footprint     | asilib.Conjunction.lla_footprint()   |
+----------------+------------------------------+--------------------------------------+
| cartopy        | Projecting images onto a map | asilib.map.create_map()*             |
+----------------+------------------------------+--------------------------------------+

*\*create_map() will fallback to a simple map function if cartopy is not installed.*

Configuration
^^^^^^^^^^^^^
aurora-asi-lib writes the data and movie files to the `asilib.config['ASI_DATA_DIR']` directory. By default `ASI_DATA_DIR` is pointed at `~/asilib-data` and it is configurable. To configure `ASI_DATA_DIR`, and other asilib settings, run `python3 -m asilib config` and answer the prompts. The prompt answer in [brackets] is the default if you don't enter anything.

Core Concepts
-------------
The core of the user interface is the :py:meth:`~asilib.imager.Imager` class. It is invoked using an *entry function* such as :py:meth:`asilib.asi.themis`, :py:meth:`asilib.asi.rego`, or :py:meth:`asilib.asi.trex.trex_nir`.

The entry function downloads the necessary image and skymap files, and passes the skymap arrays and image file paths to :py:meth:`~asilib.imager.Imager`. The entry function also specifies a loader function that loads one file given it's path. :py:meth:`~asilib.imager.Imager` uses these paths to load data as needed---also referred to as the "lazy mode"---to maintain a low money usage (necessary if working with high speed ASIs or simultaneously with multiple ASIs). If memory is not an issue, you can load all of the ASI data at once---also referred to as the "greedy mode".

Once initiated, :py:meth:`~asilib.imager.Imager` exposes an intuitive user API to load, plot, animate, and analyze ASI data.

The architecture described so far is illustrated in the flowchart below.

.. figure:: ./_static/imager_flowchart.png
    :alt: asilib Imager architecture.

asilib also implements two classes to extend :py:meth:`~asilib.imager.Imager`. First, :py:meth:`~asilib.conjunction.Conjunction` finds and calculates auroral intensity near a satellite's footprint. Second, asilib.Imagers() plots and animates images from multiple asilib.Imager() instances, useful for example, for creating mosaics.

:py:meth:`~asilib.conjunction.Conjunction`: Often ASI observations need to be combined with in-situ measurements such as low Earth orbiting satellites.

Examples
--------

Tutorial
--------