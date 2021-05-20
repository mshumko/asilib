============
Installation
============
Installing aurora-asi-lib is as simple as:

.. code-block:: shell

   python3 -m pip install aurora-asi-lib

.. note::
    By default, aurora-asi-lib saves the ASI data, movie frames, and movies in the `~/asilib-data/` directory. To override the default directory, run aurora-asi-lib as a module, `python3 -m asilib init`.


Developer Installation
^^^^^^^^^^^^^^^^^^^^^^

To install this package as a developer, run:

.. code-block:: shell

   git clone git@github.com:mshumko/aurora-asi-lib.git
   cd aurora-asi-lib
   python3 -m pip install -r requirements.txt # or
   python3 -m pip install -e .

To develop the docs, you must install Sphinx to your operating system. For linux the command is 

.. code-block:: shell

    apt-get install python3-sphinx