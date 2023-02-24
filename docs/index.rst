.. image:: ./_static/aurora-asi-lib_logo.png
  :alt: Alternative text

**Last Built**: |today| | **Version**: |version| | **Source**: `github`_ | **Archive**: `zenodo`_.

.. _github: https://github.com/mshumko/aurora-asi-lib
.. _zenodo: https://doi.org/10.5281/zenodo.4746446

Easily download, plot, animate, and analyze auroral all sky imager (ASI) data.

Supported ASI arrays
--------------------

- Red-line Emission Geospace Observatory (REGO)
- Time History of Events and Macroscale Interactions during Substorms (THEMIS).

.. note::
   While this package is named `aurora-asi-lib`, import it using the name `asilib`.

The two ways to interact with `asilib` is via the mature function Application Program Interface (API) or the experimental Imager API. If you're new to `asilib` use the function API for now.

.. toctree::
   :maxdepth: 2
   :caption: asilib:

   installation
   examples
   tutorial
   function_api
   imager_api

.. toctree::
   :maxdepth: 2
   :caption: DEVELOPMENT:

   developer_installation

..
   This is a comment block

   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
