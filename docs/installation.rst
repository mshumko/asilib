============
Installation
============
Installing aurora-asi-lib is as simple as:

.. code-block:: shell

   python3 -m pip install aurora-asi-lib 

.. note::
   - By default, aurora-asi-lib saves the ASI data, movie frames, and movies in the `~/asilib-data/` directory. To override the default directory, run aurora-asi-lib as a module, `python3 -m asilib config`. See the Configuration section below for more details.

   - If you get the "`ERROR: Could not build wheels for pymap3d which use PEP 517 and cannot be installed directly`" error when installing, you need to upgrade your pip, setuptools, and wheel libaries via ```python3 -m pip install --upgrade pip setuptools wheel```.

Dependencies
^^^^^^^^^^^^

ffmpeg
------
To make movies you'll also need to install the ffmpeg library to make movies

- **Ubuntu**: ```apt install ffmpeg```
- **Mac**: ```brew install ffmpeg```

cartopy
-------
**Once asilib implements asi maps:** to make maps you will need to install cartopy dependencies. On linux the following commands will install these dependencies.

.. code-block:: shell

   sudo apt-get install python3-dev
   sudo apt-get install libproj-dev proj-data proj-bin  
   sudo apt-get install libgeos-dev  

Configuration
^^^^^^^^^^^^^
aurora-asi-lib writes the data and movie files to the `asilib.config['ASI_DATA_DIR']` directory. By default `ASI_DATA_DIR` is pointed at `~/asilib-data` and it is configurable. To configure `ASI_DATA_DIR`, and other asilib settings, run `python3 -m asilib config` and answer the prompts. The prompt answer in [brackets] is the default if you don't enter anything.

As you probably figured out, the asilib configuration data is contained in the `asilib.config` dictionary that currently contains:

- `ASILIB_DIR`: asilib code directory (mainly used for testing)
- `ASI_DATA_DIR`: asilib data directory
- `IRBEM_WARNING`: to toggle warnings when the IRBEM-lib_ library is not installed.

=============    ===========
Parameter        Description
=============    ===========
ASILIB_DIR       asilib code directory (mainly used for testing)
ASI_DATA_DIR     asilib data directory
IRBEM_WARNING    warn when the IRBEM-lib_ library is not installed
MEMORY_USE       (NOT IMPLEMENTED!) warn when the `ASI_DATA_DIR` directory memory size exceeds a threshold 
=============    ===========

.. _IRBEM-lib: https://github.com/PRBEM/IRBEM