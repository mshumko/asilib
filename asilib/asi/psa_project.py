"""
The `Pulsating Aurora (PsA) project <http://www.psa-research.org>`_ operated high-speed ground-based cameras in the northern Scandinavia and Alaska(in Norway, Sweden, Finland, and Alaska) during the 2016-current years to observe rapid modulation of PsA. These ground-based observations will be compared with the wave and particle data from the ERG satellite, which launched in 2016, in the magnetosphere to understand the connection between the non-linear processes in the magnetosphere and periodic variation of PsA on the ground. Before using this data, please refer to the `rules of the road document <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/rules-of-the-road_psa-pwing.pdf>`_ for data caveats and other prudent considerations. The DOIs of the cameras are introduced in the rules of the road document online. When you write a paper using data from these cameras, please indicate the corresponding DOIs of the cameras that you used for your analyses. You can find the summary animations and keograms `online <https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi>`_. If there is an animation or keogram online but no data for a time period, contact the PsA team (e.g., Y. Miyoshi or K. Hosokawa) to retrieve the data from cold storage.
"""
from datetime import datetime, timedelta
from typing import Union, Iterable, List
import functools
import warnings
import pathlib
import bz2
import os
import copy
import re

import scipy.io
import numpy as np
import pandas as pd

import asilib
import asilib.utils as utils
import asilib.download as download


image_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/raw/'
skymap_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/raw/fovd/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'psa_project'

# These paths are for the LAMP project, which was also part of the PSA project, but the imagers
# are different.
lamp_image_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-pwing/pub/raw/lamp/sav_img/'
lamp_skymap_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-pwing/pub/raw/lamp/sav_fov/'
lamp_local_base_dir = asilib.config['ASI_DATA_DIR'] / 'psa_project'

def psa_project(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    downsample_factor: int = 1,
    redownload: bool = False,
    missing_ok: bool = True,
    load_images: bool = True,
    acknowledge: bool = True,
    imager=asilib.Imager,
    ) -> asilib.Imager:
    """
    Create an Imager instance of the Pulsating Aurora project's EMCCD ASIs.

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
    downsample_factor: int
        The factor by which to downsample the images. For example, a value of
        10 will reduce the image cadence by a factor of 10. For C1 after 2017-01-26,
        and C2, C6, C7, the frame rate is 100 fps, so a downsample_factor=10 will 
        yield 10 fps. Other imagers are 10 fps by default, so downsample_factor=10
        will yield 1 fps.
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
    acknowledge: bool
        If True, prints the acknowledgment statement for PsA Project.
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A PSA Project ASI instance with the time stamps, images, skymaps, and metadata.

    Examples
    --------
    >>> from asilib.asi import psa_project
    >>> # https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi?year=2017&month=03&day=07&jump=Plot
    >>> asi = psa_project(
    >>>    'C1', 
    >>>    time=datetime(2017, 3, 7, 19, 35, 0),
    >>>    redownload=False
    >>>    )
    >>> _, ax = plt.subplots(figsize=(8, 8))
    >>> ax.xaxis.set_visible(False)
    >>> ax.yaxis.set_visible(False)
    >>> plt.tight_layout()
    >>> # We use the origin kwarg to shift the origin of the cardinal direction.
    >>> asi.plot_fisheye(ax=ax, origin=(0.9, 0.1))
    >>> plt.show()

    >>> # Note that this example will take a while to download data and run.
    >>> from asilib.asi import psa_project
    >>> # https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi?year=2017&month=03&day=07&jump=Plot
    >>> asi = psa_project(
    >>>    'C1',
    >>>    time_range=(datetime(2017, 3, 7, 19, 0, 0), datetime(2017, 3, 7, 20, 0, 0)),
    >>>    redownload=False,
    >>>    downsample_factor=100  # corresponds to 1 fps for C1 after 2017-01-26.
    >>>    )
    >>> _, ax = plt.subplots(figsize=(8, 8))
    >>> ax.xaxis.set_visible(False)
    >>> ax.yaxis.set_visible(False)
    >>> plt.tight_layout()
    >>> # We use the origin kwarg to shift the origin of the cardinal direction.
    >>> asi.animate_fisheye(ax=ax, ffmpeg_params={'framerate':100}, origin=(0.9, 0.1))

    >>> # Animate the Tromsø ASI on a map.
    >>> # Note that this example will take a while to download data and run.
    >>> # Change the time_range if needed to run quicker.
    >>> from datetime import datetime
    >>> 
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> import asilib.asi
    >>> import asilib.map
    >>> 
    >>> # https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi?year=2017&month=03&day=07&jump=Plot
    >>> asi = asilib.asi.psa_project(
    >>>     'C1', 
    >>>     time_range=(datetime(2017, 3, 7, 19, 0, 0), datetime(2017, 3, 7, 20, 0, 0)),
    >>>     redownload=False,
    >>>     downsample_factor=100  # 1 fps for C1 after 2017-01-26.
    >>>     )
    >>> fig = plt.figure(figsize=(6, 6))
    >>> ax = asilib.map.create_map(
    >>>     lon_bounds=(asi.meta['lon']-10, asi.meta['lon']+10),
    >>>     lat_bounds=(asi.meta['lat']-5, asi.meta['lat']+5),
    >>>     fig_ax=(fig, 111)
    >>>     )
    >>> plt.tight_layout()
    >>> asi.animate_map(ax=ax, ffmpeg_params={'framerate':100})
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
            'loader': functools.partial(_load_image_file, downsample_factor=downsample_factor),
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

    skymap = psa_project_skymap(location_code, _time, redownload, alt)
    max_el_idx = np.unravel_index(np.argmax(skymap['el']), skymap['el'].shape)

    meta = {
        'array': 'psa_project',
        'location': location_code,
        'lat': skymap['lat'][*max_el_idx],
        'lon': skymap['lon'][*max_el_idx],
        'alt': 0,
        'cadence': downsample_factor/_fps(file_info['path'][0]),
        'resolution':(255, 255),
        'acknowledgment':(
            'The Pulsating Aurora (PsA) project (http://www.psa-research.org) operated high-speed '
            'ground-based cameras in the northern Scandinavia and Alaska(in Norway, Sweden, '
            'Finland, and Alaska) during the 2016-current years to observe rapid modulation of '
            'PsA. These ground-based observations will be compared with the wave and particle data'
            'from the ERG satellite, which launched in 2016, in the magnetosphere to understand '
            'the connection between the non-linear processes in the magnetosphere and periodic '
            'variation of PsA on the ground. Before using this data, please refer to the rules of '
            'the road document at https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/pub/rules-of-the-road_psa-pwing.pdf '
            'for data caveats and other prudent considerations. The DOIs of the cameras are '
            'introduced in the rules of the road document online. When you write a paper using '
            'data from these cameras, please indicate the corresponding DOIs of the cameras that '
            'you used for your analyses. You can find the animations and keograms online '
            '(https://ergsc.isee.nagoya-u.ac.jp/psa-gnd/bin/psa.cgi). If there is an animation '
            'or keogram online but no data for a time period, contact the PsA team (e.g., '
            'Y. Miyoshi or K. Hosokawa) to retrieve the data from cold storage.'
        )
        }
    plot_settings={
        'color_bounds':(2_000, 3_000),
        'label_fontsize': 20,
    }
    if acknowledge and ('psa_project' not in asilib.config['ACKNOWLEDGED_ASIS']):
        print(meta['acknowledgment'])
        asilib.config['ACKNOWLEDGED_ASIS'].append('psa_project')
    return imager(file_info, meta, skymap, plot_settings=plot_settings)

def psa_project_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the psa_project ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of psa_project imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'psa_project']
    return df.reset_index(drop=True)

def psa_project_skymap(location_code, time, redownload, alt):
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
            f'The requested skymap time={time} for psa_project imager {location_code.upper()} is '
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
    location_df = psa_project_info()
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

    # The [::-1, :] flips the skymap vertically so north is up.
    az_skymap = np.genfromtxt(az_skymap_path)[::-1, :]
    el_skymap = np.genfromtxt(el_skymap_path)[::-1, :]
    lat_skymap = np.genfromtxt(lat_skymap_path)[::-1, :]
    lon_skymap = np.genfromtxt(lon_skymap_path)[::-1, :]
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

def _fps(path):
    """
    Determine the frames per second of the PSA EMCCD camera from the filename.
    """
    path = pathlib.Path(path)
    m = re.search(r'C(\d)_(\d{8})_(\d{4})', path.name)
    if m is None:
        raise ValueError(f"Cannot parse camera/date from filename: {path.name}")
    camid = int(m.group(1))
    date_str = m.group(2)

    date_long = int(date_str)
    if camid == 1:
        fps = 10
        if date_long >= 20170126:
            fps = 100
    elif camid in (2, 6, 7):
        fps = 100
    else:
        fps = 10
    return fps

def _load_image_file(path, downsample_factor):
    """
    Translated from IDL with the help of ChatGPT.

    https://ergsc.isee.nagoya-u.ac.jp/psa-pwing/pub/raw/soft/psa_routines.pro
    """
    fps = _fps(path)
    img_i = 0
    time_i = 0
    number_of_images = 60 * fps

    if number_of_images % downsample_factor != 0:
        raise ValueError(
            f'downsample_factor={downsample_factor} must evenly divide the number of'
            f'images={number_of_images} in each file.'
        )

    raw_images = np.full((number_of_images+1, 256, 256), 0, dtype=np.uint16)
    raw_times = np.full((number_of_images+1,), np.datetime64("NaT"), dtype=object)

    with bz2.BZ2File(path, 'rb') as f:
        while f:
            _data_block = _ebireaded_ym(f)
            if _data_block is None:
                break  # EOF
            elif _data_block[0] == 2000:
                # Image data
                raw_images[img_i, ...] = _data_block[4]
                img_i+=1
            elif _data_block[0] == 1001:
                # Pixel Resolution
                x, y = _data_block[1], _data_block[2]
                assert x==y==256, f'The image dimensions should be 256x256 but got {x} and {y}.'
            elif _data_block[0] == 1002:
                # timestamp
                _time_raw_str = _data_block[3]
                time_chunks = _time_raw_str.split('_')
                if len(time_chunks) == 5:
                    # New style of datetimes
                    raw_times[time_i] = datetime.strptime(
                        f'{time_chunks[1]}_{time_chunks[2]}_{time_chunks[3]}',
                        '%Y%m%d_%H%M%S_%f'
                        )
                    time_i+=1
                else:
                    raise NotImplementedError(
                        f"Please submit a GitHub Issue about the old timestamp "
                        f"format that hasn't been implemented yet for file {path.name}."
                        )

    raw_times = raw_times[:time_i]
    raw_images = raw_images[:time_i, :, :]

    if downsample_factor > 1:
        # Trim to whole number of groups and average each non-overlapping group
        n_frames = raw_images.shape[0]
        n_groups = n_frames // downsample_factor

        if n_groups > 0:
            last_downsample_frame = n_groups * downsample_factor
            raw_times = raw_times[:last_downsample_frame]
            raw_images = raw_images[:last_downsample_frame, :, :]
            # reshape to (n_groups, group_size, img_H, img_W) and average over group_size
            raw_images = raw_images.reshape(
                n_groups, downsample_factor, raw_images.shape[1], raw_images.shape[2]
                ).mean(axis=1).astype(np.uint16)
            # keep the first timestamp of each group
            raw_times = raw_times[::downsample_factor]
        else:
            raise ValueError(
                f"downsample_factor={downsample_factor} is too large for "
                f"the number of frames={n_frames} in the file."
            )
    return raw_times, raw_images[:, ::-1, :]

def _ebireaded_ym(f):
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

def psa_project_lamp(
        location_code:str, 
        time: utils._time_type=None,
        time_range: utils._time_range_type=None,
        redownload=False, 
        missing_ok:bool=True, 
        alt:int=90, 
        downsample_factor:int=1
        ) -> asilib.Imager:
    """
    Create an Imager instance of the Pulsating Aurora ground-based EMCCD ASI in support of the LAMP 
    sounding rocket flight in March 2022.

    Parameters
    ----------
    location_code: str
        The ASI's location code (four letters).
    time: str or datetime.datetime
        A time to look for the ASI data at. Either time or time_range
        must be specified (not both or neither).
    time_range: list
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    redownload: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).
    alt: int
        The mapping altitude.
    downsample_factor: int
        The factor by which to downsample the images. For example, a value of
        10 will reduce the image cadence by a factor of 10. The LAMP EMCCD ASIs
        are 100 fps by default, so downsample_factor=10 will yield 10 fps.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A PsA Project LAMP ASI instance with the time stamps, images, skymaps, and metadata.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.gridspec
    >>>
    >>> import asilib.map
    >>> from asilib.asi.psa_project import psa_project_lamp
    >>>
    >>> asi = psa_project_lamp(
    >>>     'vee',
    >>>     time=datetime(2022, 3, 5, 11, 0),
    >>>     redownload=False,
    >>> )
    >>>
    >>> fig = plt.figure(figsize=(8, 4))
    >>> gs = matplotlib.gridspec.GridSpec(1, 2, fig)
    >>> ax = fig.add_subplot(gs[0,0])
    >>> bx = asilib.map.create_map(
    >>>     lon_bounds=(asi.meta['lon']-8, asi.meta['lon']+8),
    >>>     lat_bounds=(asi.meta['lat']-4, asi.meta['lat']+4),
    >>>     fig_ax=(fig, gs[0,1])
    >>>     )
    >>> ax.axis('off')
    >>> asi.plot_fisheye(ax=ax)
    >>> asi.plot_map(ax=bx)
    >>> plt.show()

    >>> # Animate a few minutes of LAMP PKF data.
    >>> from asilib.asi.psa_project import psa_project_lamp
    >>> import dateutil.parser
    >>>
    >>> alt = 110  # km
    >>> time_range_str = ('2022-03-05T14:52', '2022-03-05T14:56')
    >>> time_range = (dateutil.parser.parse(time_range_str[0]), dateutil.parser.parse(time_range_str[1]))
    >>>
    >>> asi = psa_project_lamp('pkf', time_range=time_range, alt=alt, downsample_factor=100)
    >>> asi.animate_fisheye(color_bounds=asi.auto_color_bounds(), overwrite=True)
    """
    if location_code.lower() == 'vee':
        meta = {
            'array': 'LAMP',
            'location': 'VEE',
            'lat': 67.0139,
            'lon': -146.4186,
            'alt': 0.174,
            'cadence': 0.01,
            'resolution': (256, 256),
        }
    elif location_code.lower() == 'pkf':
        meta = {
            'array': 'LAMP',
            'location': 'PKF',
            'lat': 65.1256,
            'lon': -147.4919,
            'alt': 0.213,
            'cadence': 0.01,
            'resolution': (256, 256),
        }
    else:
        raise NotImplementedError

    _skymap = psa_project_lamp_skymap(location_code, alt, redownload)
    skymap = {
        'lat': _skymap['gla'],
        'lon': _skymap['glo'],
        'alt': alt,
        'el': _skymap['ele'],
        'az': _skymap['azm'],
        'path': _skymap['path'],
    }

    file_paths = _get_lamp_file_paths(location_code, time, time_range, redownload, missing_ok)

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
        'loader': functools.partial(_lamp_reader, downsample_factor=downsample_factor),
    }
    if time_range is not None:
        file_info['time_range'] = time_range
    else:
        file_info['time'] = time

    plot_settings={
        'color_bounds':(2_000, 3_000),
        'label_fontsize': 20,
    }
    return asilib.Imager(file_info, meta, skymap, plot_settings=plot_settings)


def _get_lamp_file_paths(location_code, time, time_range, redownload, missing_ok):
    """
    Get the local file paths for the LAMP EMCCD ASI images, and download them
    if they do not exist locally.
    """
    if (time is None) and (time_range is None):
        raise ValueError('time or time_range must be specified.')
    elif (time is not None) and (time_range is not None):
        raise ValueError('both time and time_range can not be simultaneously specified.')

    local_dir = lamp_local_base_dir / 'images' / f'lamp_{location_code.lower()}'

    # Find one image file.
    if time is not None:
        time = utils.validate_time(time)
        filename = f'{location_code.lower()}_{time.strftime("%Y%m%d_%H%M")}.sav'
        file_paths = list(pathlib.Path(local_dir).rglob(filename))

        if (len(file_paths) == 0) or redownload:
            d = download.Downloader(lamp_image_base_url + f'{filename}')
            file_path = d.download(local_dir, redownload=redownload, stream=True)
            return [file_path]

    # Find multiple image files.
    if time_range is not None:
        time_range = utils.validate_time_range(time_range)
        file_times = utils.get_filename_times(time_range, dt='minutes')
        file_paths = []

        for file_time in file_times:
            filename = f'{location_code.lower()}_{file_time.strftime("%Y%m%d_%H%M")}.sav'
            _matched_file_paths = list(pathlib.Path(local_dir).rglob(filename))

            if redownload:
                d = download.Downloader(lamp_image_base_url + f'/{filename}')
                file_paths.append(d.download(local_dir, redownload=redownload, stream=True))
            else:
                if len(_matched_file_paths) == 1:
                    file_paths.append(_matched_file_paths[0])

                elif len(_matched_file_paths) == 0:
                    d = download.Downloader(lamp_image_base_url + f'/{filename}')
                    try:
                        file_paths.append(d.download(local_dir, redownload=redownload, stream=True))
                    except (FileNotFoundError, AssertionError, ConnectionError) as err:
                        if missing_ok and (
                            ('does not contain any hyper references containing' in str(err))
                            or ('Only one href is allowed' in str(err))
                            or ('error response' in str(err))
                        ):
                            continue
                        raise
                else:
                    raise ValueError(f'{len(_matched_file_paths)} files found.')
    return file_paths


def _lamp_reader(file_path, downsample_factor=1):
    """
    Reads a LAMP EMCCD .sav file and returns the times and images.
    """
    sav_data = scipy.io.readsav(str(file_path), python_dict=True)
    images = np.moveaxis(sav_data['img'], 2, 0)
    times = np.array(
        [
            datetime(y, mo, d, h, m, s, 1000 * ms)
            for y, mo, d, h, m, s, ms in zip(
                sav_data['yr'],
                sav_data['mo'],
                sav_data['dy'],
                sav_data['hh'],
                sav_data['mm'],
                sav_data['ss'],
                sav_data['ms'].astype(int),
            )
        ]
    )
    if downsample_factor > 1:
        n_frames = images.shape[0]
        n_groups = n_frames // downsample_factor

        if n_groups > 0:
            last_downsample_frame = n_groups * downsample_factor
            times = times[:last_downsample_frame]
            images = images[:last_downsample_frame, :, :]
            # reshape to (n_groups, group_size, img_H, img_W) and average over group_size
            images = images.reshape(
                n_groups, downsample_factor, images.shape[1], images.shape[2]
                ).mean(axis=1).astype(np.uint16)
            # keep the first timestamp of each group
            times = times[::downsample_factor]
        else:
            raise ValueError(
                f"downsample_factor={downsample_factor} is too large for "
                f"the number of frames={n_frames} in the file."
            )
    return times, images


def _find_lamp_skymap(location_code, alt, redownload=True):
    """
    Find the path to the skymap file.
    """
    local_dir = lamp_local_base_dir / 'skymaps' / f'lamp_{location_code.lower()}'

    # Check if the skymaps are already downloaded.
    local_skymap_paths = list(pathlib.Path(local_dir).rglob(f'{location_code.lower()}*.sav'))

    if (len(local_skymap_paths) == 0) or redownload:
        # Download the skymaps.
        d = download.Downloader(lamp_skymap_base_url)
        ds = d.ls(f'{location_code.lower()}_*.sav')
        local_skymap_paths = []
        for d in ds:
            local_skymap_paths.append(d.download(local_dir))

    for local_skymap_path in local_skymap_paths:
        if local_skymap_path.name == f'{location_code.lower()}_{alt:03}.sav':
            return local_skymap_path

    raise FileNotFoundError(
        f'Unable to find the "{location_code.lower()}_{alt:03}.sav" LAMP skymap.'
    )


def psa_project_lamp_skymap(location_code, alt, redownload):
    """
    Load the skymap file and apply the transformations.
    """
    skymap_path = _find_lamp_skymap(location_code, alt, redownload)
    _skymap_dict = scipy.io.readsav(str(skymap_path), python_dict=True)
    skymap = copy.deepcopy(_skymap_dict)

    for key, val in skymap.items():
        invalid_idx = np.where(val == -999)
        if invalid_idx[0].shape[0]:
            skymap[key][invalid_idx] = np.nan

    # Mask all elevations < 0
    invalid_idx = np.where(skymap['ele'] < 0)
    for key in skymap.keys():
        skymap[key][invalid_idx] = np.nan
    skymap['path'] = skymap_path

    # Transform the glo (geographic longitude) array from (0 -> 360) to (-180 -> 180).
    valid_val_idx = np.where(~np.isnan(skymap['glo']))
    skymap['glo'][valid_val_idx] = np.mod(skymap['glo'][valid_val_idx] + 180, 360) - 180
    # Transform the azimuth (azm) array from (-180 -> 180) to (0 -> 360).
    valid_val_idx = np.where(~np.isnan(skymap['azm']))
    skymap['azm'][valid_val_idx] = np.mod(skymap['azm'][valid_val_idx], 360)
    return skymap