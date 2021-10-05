![Test python package](https://github.com/mshumko/aurora-asi-lib/workflows/Test%20python%20package/badge.svg) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746447.svg)](https://doi.org/10.5281/zenodo.4746446)

# aurora-asi-lib
Easily download, plot, animate, and analyze aurora all sky imager (ASI) data. Currently the two supported camera systems (arrays) are: 
* Red-line Emission Geospace Observatory (REGO)
* Time History of Events and Macroscale Interactions during Substorms (THEMIS).

[API Documentation](https://aurora-asi-lib.readthedocs.io/) / [Code on GitHub](https://github.com/mshumko/aurora-asi-lib)


Easily make ASI plots (example 1)!

![Aurora plot from example 1.](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/example_1.png?raw=true)

And movies! (example4; the track and mean ASI intensity plot is a little bit more work.)
![Aurora movie from example 4.](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/20170915_023400_023557_themis_rank.gif?raw=true)

Feel free to contact me and request that I add other ASI arrays to `asilib`.

## Installation
To install this package as a user, run:

```shell
python3 -m pip install aurora-asi-lib
```

To install this package as a developer, run:

```shell
git clone git@github.com:mshumko/aurora-asi-lib.git
cd aurora-asi-lib
python3 -m pip install -r requirements.txt # or
python3 -m pip install -e .
```


In either case, you'll need to configure your system paths to tell `asilib` (the import name) where to save the ASI data and movies. Run ```python3 -m asilib config``` to set up the data directory where the image, skymap, and movie files will be saved. Your settings will be stored in `config.py`. If you configure `asilib`, but don't specify a data directory, a default directory in `~/asilib-data` will be created if it doesn't exist.

### ffmpeg dependency
To make  movies you'll also need to install the ffmpeg library.
 - **Ubuntu**: ```apt install ffmpeg```
 - **Mac**: ```brew install ffmpeg```

__NOTES__
- If you get the "ERROR: Could not build wheels for pymap3d which use PEP 517 and cannot be installed directly" error when installing, you need to upgrade your pip, setuptools, and wheel libaries via ```python3 -m pip install --upgrade pip setuptools wheel```.
