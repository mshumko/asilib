"""
The `Pulsating aurora (PsA) project <http://www.psa-research.org>`_ operated high-speed ground-based cameras in the northern Scandinavia and Alaska(in Norway, Sweden, Finland, and Alaska) during the 2016-current years to observe rapid modulation of PsA. These ground-based observations will be compared with the wave and particle data from the ERG satellite, which launched in 2016, in the magnetosphere to understand the connection between the non-linear processes in the magnetosphere and periodic variation of PsA on the ground. Before using this data, please refer to the `rules of the road document <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/rules-of-the-road_psa-pwing.pdf>`_ for data caveats and other prudent considerations. The DOIs of the cameras are introduced in the rules of the road document online. When you write a paper using data from these cameras, please indicate the corresponding DOIs of the cameras that you used for your analyses. You can find the animations and keograms `online <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi>`_.
"""
import pathlib

import pandas as pd

import asilib
import asilib.utils as utils


image_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/raw/'
skymap_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/raw/fovd/azm_ele/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'psa_emccd'

def psa_emccd(
        location_code: str,
        time: utils._time_type = None,
        time_range: utils._time_range_type = None,
        alt: int = 110,
        redownload: bool = False,
        missing_ok: bool = True,
        load_images: bool = True,
        imager=asilib.Imager,
        ) -> asilib.Imager:
    """
    Create an Imager instance of the Pulsating Aurora project's EMCCD ASI.

    Parameters
    ----------
    location_code: str
        The ASI's location code, in either the "C#" format or the full name (e.g., "Tromsoe"), case insensitive
    time: str or datetime.datetime
        A time to look for the ASI data at. Either time or time_range
        must be specified (not both or neither).
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    alt: int
        The reference skymap altitude, in kilometers.
    redownload: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).
    load_images: bool
        Create an Imager object without images. This is useful if you need to
        calculate conjunctions and don't need to download or load unnecessary data.
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A PSA Project ASI instance with the time stamps, images, skymaps, and metadata.
    """
    location_code = location_code.upper()

    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_image_dir = local_base_dir / 'images' / location_code.lower()

    file_info = {}
    skymap = {}

    meta = {
        'array': 'PSA_EMCCD',
        'location': location_code,
        # 'lat': float(_skymap['SITE_MAP_LATITUDE']),
        # 'lon': float(_skymap['SITE_MAP_LONGITUDE']),
        # 'alt': float(_skymap['SITE_MAP_ALTITUDE']) / 1e3,
        'cadence': 1/100,
        'resolution':(255, 255),
        }
    return imager(file_info, meta, skymap)

def psa_emccd_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the PSA_EMCCD ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of PSA_EMCCD imager names and locations.
    """

    return