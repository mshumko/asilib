![Test python package](https://github.com/mshumko/aurora-asi-lib/workflows/Test%20python%20package/badge.svg) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746447.svg)](https://doi.org/10.5281/zenodo.4746446)

# aurora-asi-lib
Easily download, plot, animate, and analyze aurora all sky imager (ASI) data. Currently the two supported camera systems (arrays) are: 
* Red-line Emission Geospace Observatory (REGO)
* Time History of Events and Macroscale Interactions during Substorms (THEMIS).

[API Documentation](https://aurora-asi-lib.readthedocs.io/) | [Code on GitHub](https://github.com/mshumko/aurora-asi-lib) | [PyPI archive](https://pypi.org/project/aurora-asi-lib/)


Easily make ASI fisheye lens plots (example 1).

![Aurora plot from example 1.](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/fisheye_image_arc.png?raw=true)

Or project the image onto a map
![An ASI image projected onto a map](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/map_arc.png?raw=true)

And make a keogram
![A keogram of a field line resonance](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/keogram_flr.png?raw=true)

And you can make movies
![Aurora movie from example 4.](https://github.com/mshumko/aurora-asi-lib/blob/main/docs/_static/20170915_023400_023557_themis_rank.gif?raw=true)

See more examples in the [online documentation](https://aurora-asi-lib.readthedocs.io/en/latest/examples.html) 

Feel free to contact me and request that I add other ASI arrays to `asilib`.

# Acknowledgments
This library will not be possible without 1) everyone involved with designing, building, and maintaining all-sky imaging systems, and 2) everyone who contributed to the dependencies used by `asilib`. Some of the dependencies include:
- numpy: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
- Scipy: Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
- aacgm2: Angeline Burrell for the Python source and Shepherd, S. G. (2014), Altitude‐adjusted corrected geomagnetic coordinates: Definition and functional approximations, Journal of Geophysical Research: Space Physics, 119, 7501–7521, doi:10.1002/2014JA020264. 
- pandas: Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, gfyoung, Sinhrks, Adam Klein, Matthew Roeschke, Simon Hawkins, Jeff Tratner, Chang She, William Ayd, Terji Petersen, Marc Garcia, Jeremy Schendel, Andy Hayden, MomIsBestFriend, … Mortada Mehyar. (2020). pandas-dev/pandas: Pandas 1.0.3 (v1.0.3). Zenodo. https://doi.org/10.5281/zenodo.3715232
- cartopy: Phil Elson, Elliott Sales de Andrade, Greg Lucas, Ryan May, Richard Hattersley, Ed Campbell, Andrew Dawson, Stephane Raynaud, scmc72, Bill Little, Alan D. Snow, Kevin Donkers, Byron Blay, Peter Killick, Nat Wilson, Patrick Peglar, lbdreyer, Andrew, Jon Szymaniak, … Mark Hedley. (2022). SciTools/cartopy: v0.20.2 (v0.20.2). Zenodo. https://doi.org/10.5281/zenodo.5842769
- IRBEM: Boscher, D., Bourdarie, S., O'Brien, P., Guild, T., & Shumko, M. (2012). IRBEM-lib library.