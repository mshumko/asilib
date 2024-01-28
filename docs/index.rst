.. image:: ./_static/aurora-asi-lib_logo.png
  :alt: Alternative text

**Last Built**: |today| | **Version**: |version| | **Source**: `github`_ | **Archive**: `zenodo`_.

.. _github: https://github.com/mshumko/aurora-asi-lib
.. _zenodo: https://doi.org/10.5281/zenodo.4746446

aurora-asi-lib is an open source package providing data access and analysis tools for the world's all-sky imager (ASI) data.

The purpose of this project is to combine data from numerous observational ASI arrays into a single unified framework and is thus not associated with the development and operations of all sky cameras, or the curation of ASI datasets. All data is publicly available and is provided as-is. Please give appropriate credit and coordinate with instrument teams with regards to data issues and/or interpretation. See the :ref:`acknowledgements` section for more information.

Supported ASI arrays
--------------------   
- :ref:`rego_asi`,
- :ref:`themis_asi`,
- :ref:`trex_asi`

.. note::
   While this package is named `aurora-asi-lib`, import it using the name `asilib`.

.. figure:: ./_static/global_coverage.png
   :alt: A geographic map showing the spatial coverage (field of view rings) of all imagers supported by aurora-asi-lib.
   :width: 75%

.. figure:: ./_static/collage.png
   :alt: Top four panels are a collage showing an image from a THEMIS and REGO ASI in the fisheye and mapped formats. The bottom panel is a THEMIS ASI keogram from this time interval. 
   :width: 75%

.. _acknowledgements:
Acknowledgements
----------------
asilib is not associated with the development and operations of all sky cameras, or the curation of ASI datasets. All data accessed by asilib is publicly available from the home institution responsible for the instrumentation. We recommend data users coordinate with instrument teams with regards to data issues and/or interpretation. Users are responsible to appropriately acknowledge the data sources they utilize. Required acknowledgements are contained in the descriptions of each instrument network.

If asilib significantly contributed to your research, and you would like to acknowledge it in your academic publication, please consider including the asilib developers as co-authors, and/or citing the following paper:

- Shumko M, Chaddock D, Gallardo-Lacourt B, Donovan E, Spanswick EL, Halford AJ, Thompson I and Murphy KR (2022), AuroraX, PyAuroraX, and aurora-asi-lib: A user-friendly auroral all-sky imager analysis framework. Front. Astron. Space Sci. 9:1009450. doi: 10.3389/fspas.2022.1009450

.. toctree::
   :maxdepth: 2
   :caption: asilib:

   get_started
   examples
   tutorials
   imager_api
   function_api
   contribute
