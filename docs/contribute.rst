Contribute
==========

We welcome community support in the form of feature request, bug reports, or directly contributing to asilib.

Bug Reports and Feature Requests
--------------------------------
If you find a bug or you have a feature request, feel free to let us know via `GitHub's issues <https://github.com/mshumko/aurora-asi-lib/issues/new/choose>`_. Please select the appropriate issue template to help the developers quickly understand your problem or request, and determine what changes to implement in asilib.

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

Adding an ASI array
-------------------

Three dictionaries must be passed into asilib.Imager:
1. data,
2. skymap, and
3. meta.

These dictionaries are described below.
TODO: Describe how the Imager API works.

Tests
-----
At a bare minimum, your asi loader function needs to include an example in its docstring. Furthermore, this example should also be wrapped up in a test.

See the `matplotlib docs <https://matplotlib.org/stable/devel/testing.html#writing-an-image-comparison-test>`_ on how to create and test functions that create images.

Examples
--------
TODO: Add guidance