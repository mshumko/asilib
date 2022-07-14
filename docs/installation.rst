============
Installation
============
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

+----------------+--------------------------------+
| **Dependency** | **asilib functions**           |
+----------------+--------------------------------+
| ffmpeg         | | asilib.make_movie()          |
|                | | asilib.make_movie_generator()|
+----------------+--------------------------------+
| IRBEM          | asilib.lla2footprint()         |
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