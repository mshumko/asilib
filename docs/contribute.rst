Contribute
==========

We welcome community support in the form of feature request, bug reports, or directly contributing to asilib.

Bug Reports and Feature Requests
--------------------------------
If you find a bug or you have a feature request, feel free to let us know via `GitHub's issues <https://github.com/mshumko/asilib/issues/new/choose>`_. Please select the appropriate issue template to help the developers quickly understand your problem or request, and determine what changes to implement in asilib. Lastly, before suggesting a feature, please read the :ref:`scope` section below to decide if the feature is aligned with the asilib goals.

Installation
------------

To standardize the development packages, run the following commands to reproduce the development environment locally:

.. code-block:: shell

   git clone git@github.com:mshumko/asilib.git
   cd asilib
   python3 -m pip install -r requirements.txt

To develop the docs, you must install Sphinx to your operating system. For linux the command is 

.. code-block:: shell

    apt-get install python3-sphinx

.. _contribute_asi:

Adding A New ASI
----------------
You can add a new ASI to `asilib` by writing a `wrapper` function that creates and returns an `asilib.Imager` instance. As you read the following interface descriptions, you're welcome to see an example in the `asilib/asi/fake_asi.py` module that contains a `fake_asi()` wrapper function. 

The `asilib.Imager` interface consists of four dictionaries:

- `file_info`,
- `skymap`,
- `meta`, and
- `plot_settings` (optional) 


file_info dictionary
^^^^^^^^^^^^^^^^^^^^

The `file_info` dictionary provides information on when and how to load ASI images. See the two code snippets below for the required key-value pairs for loading one or multiple images.


**One Image**

.. code-block:: python

    file_info = {
        # The time to load the image
        'time': datetime.datetime(),
        # Specify the path the relevant image file. List length is 1.
        'path': List[pathlib.Path],  
        # The time of the first image in each file in `path`. List length is 1.
        'start_time': List[datetime.datetime()],
        # The time of the last image in each file in `path`. List length is 1.
        'end_time': List[datetime.datetime()],
        # The function that takes an image path and returns an `np.array()` of 
        # `datetime.datetime()` time stamps and an `np.array()` with images. The 
        # first dimension of both arrays must correspond to the number of time 
        # stamps (1 if there is only one image per file).
        'loader': callable,
    }

The function specified by the `loader` key is called by `asilib.Imager` when it needs to call the images. This type of function is often called a callback function. 

**Multiple Images**

.. code-block:: python

    file_info = {
        # The start and end times to load the images. The Tuple length is 2.
        'time_range': Tuple[datetime.datetime()],  
        # The paths to all relevant image file. List length is N.
        'path': List[pathlib.Path],
        # The time of the first image in each file in `path`. List length is N.
        'start_time': List[datetime.datetime()],
        # The time of the last image in each file in `path`. List length is N.
        'end_time': List[datetime.datetime()],
        # The function that takes an image path and returns time stamps represented
        # as `datetime.datetime()` and images represented as a `np.array()`.
        'loader': callable,
    }

The reason that `asilib` needs both the `time_range`, as well as `start_time` and `end_time` is that in general, the `time_range` will not correspond to `start_time[0]` and `end_time[-1]`.

Skymap Dictionary
^^^^^^^^^^^^^^^^^

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

Meta Dictionary
^^^^^^^^^^^^^^^
The `meta` dictionary provides information about the ASI. See the code snippet below for the required key-value pairs. 

.. code-block:: python

    meta = {
        'array': str,  # The ASI array name
        'location': str,  # The ASI location name.
        'lat': float,  # Latitude in units of degrees.
        'lon': float, # Longitude in units of degrees. In the (-180->180) degree range.
        'alt': float,  # Imager altitude in units of km.
        'cadence': float,  # Imager cadence in units of seconds.
        'resolution': (int, int),  # Imager pixel resolution.
    }

Plot Settings
^^^^^^^^^^^^^
An optional dictionary that customizes the `asilib.Imager`'s plot settings.

.. code-block:: python

    plot_settings = {
        # REGO colormap goes from black to red.
        'color_map': matplotlib.colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r']),
        'color_norm': 'log',
        # A function that takes in an image and returns the (vmin, vmax) values passed into matplotlib.
        'color_bounds': callable 
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