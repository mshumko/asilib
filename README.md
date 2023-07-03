![Test python package](https://github.com/mshumko/aurora-asi-lib/workflows/Test%20python%20package/badge.svg) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746447.svg)](https://doi.org/10.5281/zenodo.4746446)

# aurora-asi-lib
Your one stop to working with the world's extensive arrays of auroral all-sky imagers (ASI). Currently the supported camera systems (arrays) are: 
* [Red-line Emission Geospace Observatory (REGO)](https://aurora-asi-lib.readthedocs.io/en/latest/imager_api.html#rego-asi)
* [Time History of Events and Macroscale Interactions during Substorms (THEMIS)](https://aurora-asi-lib.readthedocs.io/en/latest/imager_api.html#module-asilib.asi.themis)
* [Transition Region Explorer (TREx)](https://aurora-asi-lib.readthedocs.io/en/latest/imager_api.html#module-asilib.asi.trex)

[Documentation](https://aurora-asi-lib.readthedocs.io/) | [Code on GitHub](https://github.com/mshumko/aurora-asi-lib) | [PyPI archive](https://pypi.org/project/aurora-asi-lib/) | [Zenodo archive](https://doi.org/10.5281/zenodo.4746446)

![An asilib collage showing fisheye images, mapped images, and a keogram from the THEMIS and REGO imagers at RANK.](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/collage.png?raw=true)

[And you can animate images & conjunctions!](https://aurora-asi-lib.readthedocs.io/en/latest/basics_tutorial.html#Satellite-conjunction)

See more examples in the [online documentation](https://aurora-asi-lib.readthedocs.io/en/latest/examples.html) 

Feel free to contact me and request that I add other ASI arrays to `asilib`.

# Acknowledgments
If asilib significantly contributed to your research, and you would like to acknowledge it in your academic publication, we suggest citing the following paper:

- Shumko M, Chaddock D, Gallardo-Lacourt B, Donovan E, Spanswick EL, Halford AJ, Thompson I and Murphy KR (2022), AuroraX, PyAuroraX, and aurora-asi-lib: A user-friendly auroral all-sky imager analysis framework. Front. Astron. Space Sci. 9:1009450. doi: 10.3389/fspas.2022.1009450

This library will not be possible without 1) everyone involved with designing, building, and maintaining all-sky imaging systems, and 2) everyone who contributed to the dependencies used by `asilib`. Some of the dependencies include:
- numpy: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
- Scipy: Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
- aacgm2: Angeline Burrell for the Python source and Shepherd, S. G. (2014), Altitude‐adjusted corrected geomagnetic coordinates: Definition and functional approximations, Journal of Geophysical Research: Space Physics, 119, 7501–7521, doi:10.1002/2014JA020264. 
- pandas: Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, gfyoung, Sinhrks, Adam Klein, Matthew Roeschke, Simon Hawkins, Jeff Tratner, Chang She, William Ayd, Terji Petersen, Marc Garcia, Jeremy Schendel, Andy Hayden, MomIsBestFriend, … Mortada Mehyar. (2020). pandas-dev/pandas: Pandas 1.0.3 (v1.0.3). Zenodo. https://doi.org/10.5281/zenodo.3715232
- cartopy: Phil Elson, Elliott Sales de Andrade, Greg Lucas, Ryan May, Richard Hattersley, Ed Campbell, Andrew Dawson, Stephane Raynaud, scmc72, Bill Little, Alan D. Snow, Kevin Donkers, Byron Blay, Peter Killick, Nat Wilson, Patrick Peglar, lbdreyer, Andrew, Jon Szymaniak, … Mark Hedley. (2022). SciTools/cartopy: v0.20.2 (v0.20.2). Zenodo. https://doi.org/10.5281/zenodo.5842769
- IRBEM: Boscher, D., Bourdarie, S., O'Brien, P., Guild, T., & Shumko, M. (2012). IRBEM-lib library.