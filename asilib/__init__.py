import warnings
import pathlib

__version__ = '0.1.4'

# Check that the package has been configured and config.py file exists.
here = pathlib.Path(__file__).parent.resolve()
if not pathlib.Path(here / 'config.py').is_file():
    warnings.warn(
        'config.py file with the ASI data directory not found. '
        'Did you run "python3 -m asilib config"?'
    )
else:
    # Import download programs.
    from asilib.download.download_rego import download_rego_img, download_rego_cal
    from asilib.download.download_themis import download_themis_img, download_themis_cal

    # Import the loading functions.
    from asilib.load import load_img_file
    from asilib.load import load_cal_file
    from asilib.load import get_frame
    from asilib.load import get_frames

    # Import the plotting and animating functions.
    from asilib.plot_frame import plot_frame
    from asilib.plot_movie import plot_movie, plot_movie_generator

    # Import the skyfield and magnetic field mapping functions.
    from asilib.project_lla_to_skyfield import lla_to_skyfield

    try:
        from asilib.map_along_magnetic_field import map_along_magnetic_field
    except ImportError:
        warnings.warn(
            "The IRBEM-Lib magnetic field library is not installed so "
            "asilib.map_along_magnetic_field() won't work."
        )
