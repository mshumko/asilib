============
Installation
============
Installing aurora-asi-lib is as simple as:

.. code-block:: shell

   python3 -m pip install aurora-asi-lib 


Anaconda
^^^^^^^^

`aurora-asi-lib` can also be installed with Anaconda, however the steps are more contrived; See their  `official documentation`_ for more details. In short, you need to make an environment to install `aurora-asi-lib` and then use Anaconda to install `pandas`, and `cartopy`. Then run the above pip command to install `aurora-asi-lib` and its remaining dependencies.

.. _official documentation: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment


.. note::
   - By default, aurora-asi-lib saves the ASI data, movie images, and movies in the `~/asilib-data/` directory. To override the default directory, run aurora-asi-lib as a module, `python3 -m asilib config`. See the Configuration section below for more details.

   - If you get the "`ERROR: Could not build wheels for pymap3d which use PEP 517 and cannot be installed directly`" error when installing, you need to upgrade your pip, setuptools, and wheel libaries via ```python3 -m pip install --upgrade pip setuptools wheel```.

Dependencies
^^^^^^^^^^^^
There are three optional dependencies that you may want to install if you want to use certain `asilib` functions. See the dependency table below, followed by limited instructions on how to install these dependencies. Finally, see their official documentation for the comprehensive installation instructions.

+----------------+--------------------------------+
| **Dependency** | **asilib functions**           |
+----------------+--------------------------------+
| ffmpeg         | | asilib.make_movie()          |
|                | | asilib.make_movie_generator()|
+----------------+--------------------------------+
| IRBEM          | asilib.lla2footprint()         |
+----------------+--------------------------------+
| cartopy        | asilib.plot_map()              |
+----------------+--------------------------------+

ffmpeg
======
To make movies.

- **Linux**: ```apt install ffmpeg```
- **Mac**: ```brew install ffmpeg```

See their `main page`_ for further instructions.

.. _main page: https://ffmpeg.org/download.html

IRBEM
=====
Necessary to map along magnetic field lines. You'll need to download (or clone) the library `source code`_, and then execute these two steps:
- Compile the fortran code (`make...all` and `make...install` commands)
- `cd` into the python directory and execute `python3 -m pip install .`

.. _source code: https://github.com/PRBEM/IRBEM

cartopy
=======
To project ASI images onto a map you need to install the cartopy dependencies, followed by cartopy itself. As installing cartopy dependencies tend to be complex, see their `install`_ page for more details.

.. note::
   As of 24 January 2022, `cartopy` versions >= 0.20.0 only work with `PROJ` 8.0.0 or later. Linux `apt` does not have this version available yet so you'll have to build it from source.

.. _install: https://scitools.org.uk/cartopy/docs/latest/installing.html#installing

**Linux**

.. code-block:: shell

   sudo apt-get install python3-dev
   sudo apt-get install libproj-dev proj-data proj-bin  
   sudo apt-get install libgeos-dev  

**Mac**

.. code-block:: shell

   brew install proj geos
   pip3 install --upgrade pyshp
   pip3 install shapely --no-binary shapely

Configuration
^^^^^^^^^^^^^
aurora-asi-lib writes the data and movie files to the `asilib.config['ASI_DATA_DIR']` directory. By default `ASI_DATA_DIR` is pointed at `~/asilib-data` and it is configurable. To configure `ASI_DATA_DIR`, and other asilib settings, run `python3 -m asilib config` and answer the prompts. The prompt answer in [brackets] is the default if you don't enter anything.

As you probably figured out, the asilib configuration data is contained in the `asilib.config` dictionary that currently contains:

=============    ===========
Parameter        Description
=============    ===========
ASILIB_DIR       asilib code directory (mainly used for testing)
ASI_DATA_DIR     asilib data directory
=============    ===========

.. _IRBEM-lib: https://github.com/PRBEM/IRBEM