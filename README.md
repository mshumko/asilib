![Test python package](https://github.com/mshumko/asilib/workflows/Test%20python%20package/badge.svg) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746447.svg)](https://doi.org/10.5281/zenodo.4746446)

# asilib
asilib is an open source package providing data access and analysis tools for the world's all-sky imager (ASI) data.

> [!NOTE]  
> The asilib code on PyPI moved from [aurora-asi-lib](https://pypi.org/project/aurora-asi-lib/) to [asilib](https://pypi.org/project/asilib/). To update to version >=0.22.0, run
> 1. `python3 -m pip uninstall aurora-asi-lib` and
> 2. `python3 -m pip install asilib`.

The purpose of this project is to combine data from numerous observational ASI arrays into a single unified framework and is thus not associated with the development and operations of all sky cameras, or the curation of ASI datasets. All data is publicly available and is provided as-is. Please give appropriate credit and coordinate with instrument teams with regards to data issues and/or interpretation. See the [Acknowledgements](https://aurora-asi-lib.readthedocs.io/en/latest/index.html#acknowledgements) section for more information. Currently the supported camera systems (arrays) are: 
* [Red-line Emission Geospace Observatory (REGO)](https://aurora-asi-lib.readthedocs.io/en/latest/api.html#themis-asi)
* [Time History of Events and Macroscale Interactions during Substorms (THEMIS)](https://aurora-asi-lib.readthedocs.io/en/latest/api.html#module-asilib.asi)
* [Transition Region Explorer (TREx)](https://aurora-asi-lib.readthedocs.io/en/latest/api.html#trex-asi)
* [Mid-latitude All-sky-imaging Network for Geophysical Observations (MANGO)](https://aurora-asi-lib.readthedocs.io/en/latest/api.html#mango-asi)

[Documentation](https://aurora-asi-lib.readthedocs.io/) | [Code on GitHub](https://github.com/mshumko/asilib) | [PyPI archive](https://pypi.org/project/asilib/) | [Zenodo archive](https://doi.org/10.5281/zenodo.4746446)

![A geographic map showing the spatial coverage (field of view rings) of all imagers supported by asilib.](https://github.com/mshumko/asilib/blob/main/docs/_static/global_coverage.png?raw=true)

![An asilib collage showing fisheye images, mapped images, and a keogram from the THEMIS and REGO imagers at RANK.](https://github.com/mshumko/asilib/blob/main/docs/_static/collage.png?raw=true)

[And you can animate images & conjunctions!](https://aurora-asi-lib.readthedocs.io/en/latest/basics_tutorial.html#Satellite-conjunction)

See more examples in the [online documentation](https://aurora-asi-lib.readthedocs.io/en/latest/examples.html) 

Your contributions and ideas are always welcome. The easiest way is to submit [Issues](https://github.com/mshumko/asilib/issues) or [Pull Requests](https://github.com/mshumko/asilib/pulls).

# Acknowledgments
Are in the [Acknowledgements](https://aurora-asi-lib.readthedocs.io/en/latest/index.html#acknowledgements) section of the online documentation.
