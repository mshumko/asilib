"""
The `Pulsating Aurora (PsA) project <http://www.psa-research.org>`_ operated high-speed ground-based cameras in the northern Scandinavia and Alaska(in Norway, Sweden, Finland, and Alaska) during the 2016-current years to observe rapid modulation of PsA. These ground-based observations will be compared with the wave and particle data from the ERG satellite, which launched in 2016, in the magnetosphere to understand the connection between the non-linear processes in the magnetosphere and periodic variation of PsA on the ground. Before using this data, please refer to the `rules of the road document <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/rules-of-the-road_psa-pwing.pdf>`_ for data caveats and other prudent considerations. The DOIs of the cameras are introduced in the rules of the road document online. When you write a paper using data from these cameras, please indicate the corresponding DOIs of the cameras that you used for your analyses. You can find the animations and keograms `online <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi>`_.
"""
from datetime import datetime
import pathlib
import bz2

import numpy as np
import pandas as pd

import asilib
import asilib.utils as utils
import asilib.io.download as download


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
    if (len(location_code) != 2) and location_code[0] != 'C':
        location_df = psa_emccd_info()
        location_df['name_uppercase'] = location_df['name'].str.upper()
        row = location_df.loc[location_df['name_uppercase']==location_code]
        if row.shape[0] != 1:
            raise ValueError(
                f'{location_code=} is invalid. Try one of these: '
                f'{location_df["location_code"].to_numpy()} or '
                f'{location_df["name"].to_numpy()}'
            )
    pass

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
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'psa_project']
    return df.reset_index(drop=True)

def psa_emccd_skymap(location_code, time, redownload):
    """
    Load a PSA EMCCD ASI skymap file.

    Parameters
    ----------
    location_code: str
        The four character location name.
    time: str or datetime.datetime
        A ISO-formatted time string or datetime object. Must be in UT time.
    redownload: bool
        Redownload all skymaps.
    """
    time = utils.validate_time(time)
    local_dir = local_base_dir / 'skymaps' / location_code.lower()
    local_dir.mkdir(parents=True, exist_ok=True)
    skymap_top_url = skymap_base_url + location_code.lower() + '/'

    # if redownload:
    #     # Delete any existing skymap files.
    #     local_skymap_paths = pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
    #     for local_skymap_path in local_skymap_paths:
    #         os.unlink(local_skymap_path)
    #     local_skymap_paths = _download_all_skymaps(
    #         location_code, skymap_top_url, local_dir, redownload=redownload
    #     )

    # else:
    #     local_skymap_paths = sorted(
    #         pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
    #     )
    #     # TODO: Add a check to periodically redownload the skymap data, maybe once a month?
    #     if len(local_skymap_paths) == 0:
    #         local_skymap_paths = _download_all_skymaps(
    #             location_code, skymap_top_url, local_dir, redownload=redownload
    #         )

    # skymap_filenames = [local_skymap_path.name for local_skymap_path in local_skymap_paths]
    # skymap_file_dates = []
    # for skymap_filename in skymap_filenames:
    #     date_match = re.search(r'\d{8}', skymap_filename)
    #     skymap_file_dates.append(datetime.strptime(date_match.group(), '%Y%m%d'))

    # # Find the skymap_date that is closest and before time.
    # # For reference: dt > 0 when time is after skymap_date.
    # dt = np.array([(time - skymap_date).total_seconds() for skymap_date in skymap_file_dates])
    # dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    # if np.all(~np.isfinite(dt)):
    #     # Edge case when time is before the first skymap_date.
    #     closest_index = 0
    #     warnings.warn(
    #         f'The requested skymap time={time} for THEMIS-{location_code.upper()} is before first '
    #         f'skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
    #     )
    # else:
    #     closest_index = np.nanargmin(dt)
    # skymap_path = local_skymap_paths[closest_index]
    # skymap = _load_skymap(skymap_path)
    return skymap

def _load_image_file(path):
    with bz2.open(path, "rb") as f:
        content = f.read()
    return

if __name__ == '__main__':
    asi = psa_emccd(
        'test', 
        time_range=(
            datetime(2019, 3, 1, 18, 30, 0),
            datetime(2019, 3, 1, 20, 0, 0)
            )
        )
    pass