============
Installation
============
Installing aurora-asi-lib is as simple as:

.. code-block:: shell

   python3 -m pip install aurora-asi-lib 

.. note::
   By default, aurora-asi-lib saves the ASI data, movie frames, and movies in the `~/asilib-data/` directory. To override the default directory, run aurora-asi-lib as a module, `python3 -m asilib init`.

ffmpeg dependency
-----------------
To make movies you'll also need to install the ffmpeg library to make movies

- **Ubuntu**: ```apt install ffmpeg```
- **Mac**: ```brew install ffmpeg```