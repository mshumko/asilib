=============
API Reference
=============

The below functions can be imported and called using the following two preferred ways:

- .. code-block:: python

    import asilib
    asilib.download_themis_img(...)

- .. code-block:: python

    from asilib.io.download_themis import download_themis_img
    download_themis_img(...)

The former option is possible because these functions are all imported by default. However, this may change in a future release, so absolute import (former example) is preferred.

Given Python's flexibility, there are many alternative ways to import asilib with varying degrees of typing.


.. note::

    `asilib` is hierarchically structured so that the `load_` functions call the `download_` functions internally if a file does not exist locally, or `force_download=True`. Therefore, **you don't normally need to call the `download_` functions unless you need to download data in bulk.**


Imager
======

.. note::

    All of the asilib functionality will soon be combined into an Imager() class.

Download
^^^^^^^^

.. automodule:: asilib.io.download_themis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: asilib.io.download_rego
   :members: download_rego_img, download_rego_cal
   :undoc-members:
   :show-inheritance: