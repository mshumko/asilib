===========
Get Started
=========== 

Installation
------------

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
There are three optional dependencies that you may want to install if you want to use certain `asilib` functions. See the dependency table below, followed by limited instructions on how to install these dependencies. Finally, see their official documentation for the comprehensive installation instructions.

+----------------+--------------------------------------+
| **Dependency** | **asilib methods**                   |
+----------------+--------------------------------------+
| ffmpeg         | | asilib.Imager.animate_fisheye()    |
|                | | asilib.Imager.animate_map()        |
|                | | asilib.Imager.animate_fisheye_gen()|
|                | | asilib.Imager.animate_map_gen()    |
+----------------+--------------------------------------+
| IRBEM          | asilib.Conjunction.lla_footprint()   |
+----------------+--------------------------------------+
| cartopy        | asilib.map.create_map()*             |
+----------------+--------------------------------------+

*\*create_map() will fallback to a simple map function if cartopy is not installed.*

ffmpeg
======
To make movies.

- **Linux**: ```apt install ffmpeg```
- **Mac**: ```brew install ffmpeg```

See their `main page <https://ffmpeg.org/download.html>`_ for further instructions.

IRBEM
=====
Necessary to map along magnetic field lines. You'll need to download (or clone) the library `source code <https://github.com/PRBEM/IRBEM>`_, and then execute these two steps:
- Compile the fortran code (`make...all` and `make...install` commands)
- `cd` into the python directory and execute `python3 -m pip install .`

Configuration
-------------
aurora-asi-lib writes the data and movie files to the `asilib.config['ASI_DATA_DIR']` directory. By default `ASI_DATA_DIR` is pointed at `~/asilib-data` and it is configurable. To configure `ASI_DATA_DIR`, and other asilib settings, run `python3 -m asilib config` and answer the prompts. The prompt answer in [brackets] is the default if you don't enter anything.

asilib Internals
----------------
TODO: Describe how asilib works.