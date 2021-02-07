import warnings
import pathlib

# Check that the package has been configured and config.py file exists.
here = pathlib.Path(__file__).parent.resolve()
if not pathlib.Path(here / 'config.py').is_file():
    warnings.warn('config.py file with the ASI data directory not found. '
                'Did you run "python3 -m asi config"?')
else:
    # Import download programs.
    from asilib.download.download_rego import download_rego_img, download_rego_cal
    from asilib.download.download_themis import download_themis_img, download_themis_cal

    from asilib.load import load_img_file
    from asilib.load import load_cal_file

    from asilib.plot_frame import plot_frame
    from asilib.plot_movie import plot_movie, plot_movie_generator

    # Import the magnetic field mapping 
    from asilib.project_lla_to_skyfield import lla_to_skyfield
    from asilib.map_along_magnetic_field import map_along_magnetic_field