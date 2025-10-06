"""
The `Pulsating Aurora (PsA) project <http://www.psa-research.org>`_ operated high-speed ground-based cameras in the northern Scandinavia and Alaska(in Norway, Sweden, Finland, and Alaska) during the 2016-current years to observe rapid modulation of PsA. These ground-based observations will be compared with the wave and particle data from the ERG satellite, which launched in 2016, in the magnetosphere to understand the connection between the non-linear processes in the magnetosphere and periodic variation of PsA on the ground. Before using this data, please refer to the `rules of the road document <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/rules-of-the-road_psa-pwing.pdf>`_ for data caveats and other prudent considerations. The DOIs of the cameras are introduced in the rules of the road document online. When you write a paper using data from these cameras, please indicate the corresponding DOIs of the cameras that you used for your analyses. You can find the animations and keograms `online <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi>`_.
"""
from datetime import datetime, timedelta
from typing import Union, Iterable, List
import warnings
import pathlib
import bz2
import os
import re

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

import asilib
import asilib.skymap
import asilib.utils as utils
import asilib.download as download


image_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/raw/'
skymap_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/raw/fovd/'
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
    location_code = _verify_location(location_code)

    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_image_dir = local_base_dir / 'images' / location_code.lower()

    if load_images:
        # Download and find image data
        file_paths = _get_raw_files(
            location_code,
            time,
            time_range,
            image_base_url,
            local_image_dir,
            redownload,
            missing_ok,
        )

        start_times = len(file_paths) * [None]
        end_times = len(file_paths) * [None]
        for i, file_path in enumerate(file_paths):
            date_match = re.search(r'\d{8}_\d{4}', file_path.name)
            start_times[i] = datetime.strptime(date_match.group(), '%Y%m%d_%H%M')
            end_times[i] = start_times[i] + timedelta(minutes=1)
        file_info = {
            'path': file_paths,
            'start_time': start_times,
            'end_time': end_times,
            'loader': _load_image_file,
        }
    else:
        file_info = {
            'path': [],
            'start_time': [],
            'end_time': [],
            'loader': [],
        }
    if time_range is not None:
        file_info['time_range'] = time_range
        _time = time_range[0]
    else:
        file_info['time'] = time
        _time = time

    skymap = psa_emccd_skymap(location_code, _time, redownload, alt)

    meta = {
        'array': 'PSA_EMCCD',
        'location': location_code,
        # TODO: Get site locations from Keisuke.
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

def psa_emccd_skymap(location_code, time, redownload, alt):
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

    if redownload:
        # Delete any existing skymap files.
        local_skymap_paths = pathlib.Path(local_dir).rglob(f'{location_code}*.txt')
        for local_skymap_path in local_skymap_paths:
            os.unlink(local_skymap_path)
        local_skymap_paths = _download_all_skymaps(
            location_code, skymap_base_url, local_dir, redownload=redownload
        )

    else:
        local_skymap_paths = sorted(
            pathlib.Path(local_dir).rglob(f'{location_code}*.txt')
        )
        if len(local_skymap_paths) == 0:
            local_skymap_paths = _download_all_skymaps(
                location_code, skymap_base_url, local_dir, redownload=redownload
            )

    skymap_filenames = [local_skymap_path.name for local_skymap_path in local_skymap_paths]
    skymap_file_dates = []
    for skymap_filename in skymap_filenames:
        date_match = re.search(r'\d{8}', skymap_filename)
        skymap_file_dates.append(datetime.strptime(date_match.group(), '%Y%m%d'))

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date.
    dt = np.array([(time - skymap_date).total_seconds() for skymap_date in skymap_file_dates])
    dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    if np.all(~np.isfinite(dt)):
        # Edge case when time is before the first skymap_date.
        closest_index = 0
        warnings.warn(
            f'The requested skymap time={time} for psa_emccd imager {location_code.upper()} is '
            f'before first skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    closest_date = skymap_file_dates[closest_index]
    # Find the the az and el skymaps (2), and the lat and lon skymaps (2*n_altitudes).
    valid_skymap_idx = np.where(np.array(skymap_file_dates) == closest_date)[0]
    valid_skymap_paths = np.array(local_skymap_paths)[valid_skymap_idx]
    skymap = _load_skymap(valid_skymap_paths, alt)
    return skymap

def _verify_location(location_str):
    """
    Locate and verify that the location code (C#) or full name is valid.
    """
    location_code = location_str.upper()
    location_df = psa_emccd_info()
    location_df['name_uppercase'] = location_df['name'].str.upper()
    if (len(location_code) != 2) and location_code[0] != 'C':
        # Assume its the full name
        row = location_df.loc[location_df['name_uppercase']==location_code]
        if row.shape[0] != 1:
            raise ValueError(
                f'{location_code=} is invalid. Try one of these: '
                f'{location_df["location_code"].to_numpy()} or '
                f'{location_df["name"].to_numpy()}'
            )
    else:
        # Assume it is a Camera code.
        row = location_df.loc[location_df['location_code']==location_code]
        if row.shape[0] != 1:
            raise ValueError(
                    f'{location_code=} is invalid. Try one of these: '
                    f'{location_df["location_code"].to_numpy()} or '
                    f'{location_df["name"].to_numpy()}'
                )
    return row['location_code'].to_numpy()[0]

def _download_all_skymaps(location_code, url, save_dir, redownload):
    save_paths = []

    for folder in ['azm_ele/', 'gla_glo/']:
        d = download.Downloader(url + folder)
        ds = d.ls(f'{location_code}*.txt')

        for d_i in ds:
            save_paths.append(d_i.download(save_dir, redownload=redownload))
    return save_paths

def _load_skymap(valid_skymap_paths, alt):
    for valid_skymap_path in valid_skymap_paths:
        if 'az.txt' in valid_skymap_path.name:
            az_skymap_path = valid_skymap_path
        if 'el.txt' in valid_skymap_path.name:
            el_skymap_path = valid_skymap_path
        if f'{alt}_lat.txt' in valid_skymap_path.name:
            lat_skymap_path = valid_skymap_path
        if f'{alt}_long.txt' in valid_skymap_path.name:
            lon_skymap_path = valid_skymap_path

    # TODO: Verify that the altitude is available.
    az_skymap = np.genfromtxt(az_skymap_path)
    el_skymap = np.genfromtxt(el_skymap_path)
    lat_skymap = np.genfromtxt(lat_skymap_path)
    lon_skymap = np.genfromtxt(lon_skymap_path)
    lat_skymap[lat_skymap == -999] = np.nan
    lon_skymap[lon_skymap == -999] = np.nan
    return {'az':az_skymap, 'el':el_skymap, 'lat':lat_skymap, 'lon':lon_skymap, 'alt':alt}

def _get_raw_files(
    location_code: str,
    time: datetime,
    time_range: Iterable[datetime],
    base_url: str,
    save_dir: Union[str, pathlib.Path],
    redownload: bool,
    missing_ok: bool,
) -> List[pathlib.Path]:
    """
    Look for and download 1-minute RAW files.

    Parameters
    ----------
    location_code:str
        The C# location code.
    time: datetime.datetime
        Time to download one file. Either time or time_range must be specified,
        but not both.
    time_range: Iterable[datetime]
        An iterable with a start and end time. Either time or time_range must be
        specified, but not both.
    base_url: str
        The starting URL to search for file.
    save_dir: str or pathlib.Path
        The parent directory where to save the data to.
    redownload: bool
        Download data even if the file is found locally. This is useful if data
        is corrupt.
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).

    Returns
    -------
    list(pathlib.Path)
        Local paths to each PGM file that was successfully downloaded.
    """
    if (time is None) and (time_range is None):
        raise ValueError('time or time_range must be specified.')
    elif (time is not None) and (time_range is not None):
        raise ValueError('both time and time_range can not be simultaneously specified.')

    if redownload:
        # Option 1/4: Download one minute of data regardless if it is already saved
        if time is not None:
            return [
                _download_one_raw_file(location_code, time, base_url, save_dir, redownload)
            ]

        # Option 2/4: Download the data in time range regardless if it is already saved.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='minutes')
            file_paths = []
            for file_time in file_times:
                try:
                    file_paths.append(
                        _download_one_raw_file(
                            location_code, file_time, base_url, save_dir, redownload
                        )
                    )
                except (FileNotFoundError, AssertionError) as err:
                    if missing_ok and (
                        ('does not contain any hyper references containing' in str(err))
                        or ('Only one href is allowed' in str(err))
                    ):
                        continue
                    raise
            return file_paths
    else:
        # Option 3/4: Download one minute of data if it is not already saved.
        if time is not None:
            file_search_str = f'{location_code}_{time:%Y%m%d_%H%M}.raw.bz2'
            local_file_paths = list(pathlib.Path(save_dir).rglob(file_search_str))
            if len(local_file_paths) == 1:
                return local_file_paths
            else:
                return [
                    _download_one_raw_file(
                        location_code, time, base_url, save_dir, redownload
                    )
                ]

        # Option 4/4: Download the data in time range if they don't exist locally.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='minutes')
            file_paths = []
            for file_time in file_times:
                file_search_str = (
                    f'{location_code}_{file_time:%Y%m%d_%H%M}.raw.bz2'
                )
                local_file_paths = list(pathlib.Path(save_dir).rglob(file_search_str))
                if len(local_file_paths) == 1:
                    file_paths.append(local_file_paths[0])
                else:
                    try:
                        file_paths.append(
                            _download_one_raw_file(
                                location_code, file_time, base_url, save_dir, redownload
                            )
                        )
                    except (FileNotFoundError, AssertionError) as err:
                        if missing_ok and (
                            ('does not contain any hyper references containing' in str(err))
                            or ('Only one href is allowed' in str(err))
                        ):
                            continue
                        raise
            return file_paths


def _download_one_raw_file(
    location_code: str,
    time: datetime,
    base_url: str,
    save_dir: Union[str, pathlib.Path],
    redownload: bool,
) -> pathlib.Path:
    """
    Download one raw file.

    Parameters
    ---------- 
    location_code: str
        The C# location code.
    time: str or datetime.datetime
        A time to look for the ASI data at.
    base_url: str
        The starting URL to search for file.
    save_dir: str or pathlib.Path
        The parent directory where to save the data to.
    redownload: bool
        Will redownload an existing file.

    Returns
    -------
    pathlib.Path
        Local path to file.
    """
    start_url = base_url + f'cam{location_code[-1]}/{time.year}/{time.month:02}/{time.day:02}/'
    d = download.Downloader(start_url)
    # Find the unique directory
    matched_downloaders = d.ls(f'{location_code}_{time:%Y%m%d_%H%M}.raw.bz2')
    assert len(matched_downloaders) == 1
    # Search that directory for the file and download it.
    return matched_downloaders[0].download(save_dir, redownload=redownload, stream=True)


def _load_image_file(path):
    """
    Translated from IDL with the help of ChatGPT.

    https://ergsc.isee.nagoya-u.ac.jp/psa-pwing/pub/raw/soft/psa_routines.pro
    """
    n = 0
    n_max = 200 # TODO: Calc this from the cadence.
    with bz2.BZ2File(path, 'rb') as f:
        while n < n_max:
            _data_block = ebireaded_ym(f)
            if _data_block[0] == 2000:
                # Image data
                _image = _data_block[4]
                # plt.imshow(_image, vmin=2000, vmax=2700); plt.show()
            elif _data_block[0] == 1001:
                # Pixel Resolution
                x, y = _data_block[1], _data_block[2]
                assert x==y==256, f'The image dimensions should be 256x256 but got {x} and {y}.'
            elif _data_block[0] == 1002:
                # timestamp
                _time = _data_block[3]
            n+=1
    return

def ebireaded_ym(f):
    """
    Read one image-related block from the PSA binary file.
    
    Parameters
    ----------
    f : file-like object
        Binary file handle opened in 'rb' mode.

    Returns
    -------
    tuple : (itag, x, y, imgname, dat)
        itag    : int - record type
        x, y    : int or None - image dimensions if available
        imgname : str or None - image name if available
        dat     : numpy.ndarray or None - image data if available
    """
    # Defaults
    x = None
    y = None
    imgname = None
    dat = None

    # Read record tag
    tag_bytes = f.read(4)
    if not tag_bytes or len(tag_bytes) < 4:
        return None  # EOF
    iTag = np.frombuffer(tag_bytes, dtype='<i4')[0]  # IDL LONG = 4-byte signed int

    if iTag == 1000:
        # Header: size + raw bytes
        iSize = np.frombuffer(f.read(4), dtype='<i4')[0]
        tmp = bytearray(f.read(iSize))
        # In IDL, tmp is just a byte array; here we ignore content for now

    elif iTag == 1001:
        # Header with depth and dimensions
        iSize = np.frombuffer(f.read(4), dtype='<i4')[0]
        iDepth = np.frombuffer(f.read(4), dtype='<i4')[0]
        x = np.frombuffer(f.read(4), dtype='<i4')[0]
        y = np.frombuffer(f.read(4), dtype='<i4')[0]

    elif iTag == 1002:
        # Filename
        iSize = np.frombuffer(f.read(4), dtype='<i4')[0]
        cname = np.frombuffer(f.read(iSize), dtype=np.uint8)
        # Build string until null byte
        chars = []
        for b in cname:
            if b == 0:
                break
            chars.append(chr(b))
        imgname = ''.join(chars)

    elif iTag == 2000:
        # Image data
        x = y = 256
        iSize = np.frombuffer(f.read(4), dtype='<i4')[0]
        if x is None or y is None:
            raise ValueError("Image dimensions (x, y) must be set before reading data.")
        dat = np.frombuffer(f.read(x * y * 2), dtype='<u2').reshape((y, x))

    elif iTag == 9999:
        # End or placeholder
        pass

    else:
        raise ValueError(f"Unknown record tag: {iTag}")

    return iTag, x, y, imgname, dat


if __name__ == '__main__':
    # https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi?year=2017&month=03&day=07&jump=Plot
    asi = psa_emccd(
        'C1', 
        # time=datetime(2017, 3, 7, 19, 35, 0),
        time_range=(datetime(2017, 3, 7, 19, 0, 0), datetime(2017, 3, 7, 20, 0, 0)),
        redownload=False
        )
    asi.plot_fisheye()