import warnings
import pathlib
import importlib.util

__version__ = '0.2.2'

# Check that the package has been configured and config.py file exists.
here = pathlib.Path(__file__).parent.resolve()
if not pathlib.Path(here / 'config.py').is_file():
    warnings.warn(
        'config.py file with the ASI data directory not found. '
        'Did you run "python3 -m asilib config"?'
    )
else:
    # Import download programs.
    from asilib.io.download_rego import download_rego_img, download_rego_cal
    from asilib.io.download_themis import download_themis_img, download_themis_cal

    # Import the loading functions.
    from asilib.io.load import load_img_file
    from asilib.io.load import load_cal_file
    from asilib.io.load import get_frame
    from asilib.io.load import get_frames

    # Import the plotting and animating functions.
    from asilib.plot_frame import plot_frame
    from asilib.plot_movie import plot_movie, plot_movie_generator

    # Import the skyfield and magnetic field mapping functions.
    from asilib.utils.project_lla_to_skyfield import lla_to_skyfield

    # If the IRBEM module exists, import map_along_magnetic_field.
    # This is better than a try-except block because the ImportError
    # exception can be raised from another map_along_magnetic_field 
    # dependency but we want to specifically check for IRBEM and let
    # it crash if something else is wrong.
    if importlib.util.find_spec('IRBEM'):
        from asilib.utils.map_along_magnetic_field import map_along_magnetic_field
    else:
        warnings.warn(
            "The IRBEM magnetic field library is not installed and is "
            "a dependency of asilib.map_along_magnetic_field()."
        )
