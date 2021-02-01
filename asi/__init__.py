# Currently the download scripts can be accessed by 
# asi.download_themis.download_asi_frames,
# but I want to just have asi.download_themis() function.
# from .download import download_themis
import warnings
import pathlib

# Check that the package has been configured and config.py file exists.
here = pathlib.Path(__file__).parent.resolve()
if not pathlib.Path(here / 'config.py').is_file():
    print('config.py file with the ASI data directory not found. '
                'Did you run "python3 -m asi config"?')
else:
    # Import download programs.
    from asi.download.download_rego import download_rego_img, download_rego_cal
    from asi.download.download_themis import download_themis_img, download_themis_cal

    from asi.load import load_img_file
    from asi.load import load_cal_file

    from asi.plot_frame import plot_frame
    from asi.plot_movie import plot_movie

    # Import the magnetic field mapping 
    # from asi import lla2azel
    # from asi import map_satellite