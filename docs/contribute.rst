Contribute
==========

We welcome community support in the form of feature request, bug reports, or directly contributing to asilib.

Bug Reports and Feature Requests
--------------------------------
If you find a bug or you have a feature request, feel free to let us know via `GitHub's issues <https://github.com/mshumko/aurora-asi-lib/issues/new/choose>`_. Please select the appropriate issue template to help the developers quickly understand your problem or request, and determine what changes to implement in asilib. Lastly, before suggesting a feature, please read the :ref:`scope` section below to decide if the feature is aligned with the asilib goals.

Installation
------------

To standardize the development packages, run the following commands to reproduce the development environment locally:

.. code-block:: shell

   git clone git@github.com:mshumko/aurora-asi-lib.git
   cd aurora-asi-lib
   python3 -m pip install -r requirements.txt

To develop the docs, you must install Sphinx to your operating system. For linux the command is 

.. code-block:: shell

    apt-get install python3-sphinx

Adding a new ASI
----------------
You can add a new ASI to `asilib` by writing a `wrapper` function that creates and returns an `asilib.Imager` instance. As you read the following interface descriptions, you're welcome to see an example in the `asilib/asi/fake_asi.py` module that contains a `fake_asi()` wrapper function. 

The `asilib.Imager` interface consists of three dictionaries:

1. `data`,

2. `skymap`, and

3. `meta`.

The `data` dictionary provides information on when and how to load ASI images. The four required keys are: 
- `path`---list of image file paths represented as `pathlib.Path`, that `asilib.Imager` will load. 
- `start_time`---list specifying the time of the first image in each file in `path`,
- `end_time`---list specifying the time of the last image in each file in `path`,
- `loader`---function with a `path` argument that returns time stamps represented as `datetime.datetime` and images represented as a `np.array`. `asilib.Imager` will call the `loader` when it needs to load one or more images.

Lastly, `data` must contain either a `time` or `time_range` keys. These tell `asilib.Imager` what image to load, or what time range the user requested (in general, the `time_range` will not correspond to `start_time[0]` and `end_time[-1]`).

The `skymap` dictionary provides information on how to orient and map images onto a geographic map. See the code snippet below for the required key-value pairs. 

.. TODO: Describe the dimensions of the image and skymap arrays.

.. code-block:: python

    skymap = {
            'lat':np.array(...),  # Latitude of pixel vertices.
            'lon':np.array(...),  # Longitude of pixel vertices. In the (-180->180) degree range.
            'alt':float,  # The mapping altitude in km.
            'el':np.array(...),   # The elevation of each pixel.
            'az':np.array(...),   # The azimuth of each pixel.
            'path':pathlib.Path(...),  # The path to the skymap file.
        }

The `meta` dictionary provides information about the ASI. See the code snippet below for the required key-value pairs. 

.. code-block:: python

    meta = {
        'array': str,  # The ASI array name
        'location': str,  # The ASI location name.
        'lat': float,  # Latitude in units of degrees.
        'lon': float, # Longitude in units of degrees. In the (-180->180) degree range.
        'alt': float,  # Imager altitude in units of km.
        'cadence': 3,  # Imager cadence in units of seconds.
        'resolution': (int, int),  # Imager pixel resolution.
    }

Tests
-----
At a bare minimum, your asi loader function needs to include an example in its docstring. Furthermore, this example should also be wrapped up in a test.

See the `matplotlib docs <https://matplotlib.org/stable/devel/testing.html#writing-an-image-comparison-test>`_ on how to create and test functions that create images.

Examples
--------
TODO: Add guidance


.. _scope:

Scope
-----
TODO: Add