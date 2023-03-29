Contribute
==========

We added the following guidance if you'd like to contribute to asilib.

Installation
-----------

To standardize the development packages, run the following commands to reproduce the development environment locally:

.. code-block:: shell

   git clone git@github.com:mshumko/aurora-asi-lib.git
   cd aurora-asi-lib
   python3 -m pip install -r requirements.txt # or
   python3 -m pip install -e .

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

Tests
-----
At a bare minimum, your asi loader function needs to include an example in its docstring. Furthermore, this example should also be wrapped up in a test.

See the matplotlib docs on how to create and test functions that create images.
https://matplotlib.org/stable/devel/testing.html#writing-an-image-comparison-test

Examples
--------