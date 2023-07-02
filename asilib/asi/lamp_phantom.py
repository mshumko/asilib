from datetime import datetime, timedelta
import pathlib
import re

import numpy as np
import pandas as pd
import h5py

import asilib
import asilib.utils as utils
import asilib.io.download as download


image_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-pwing/pub/raw/lamp/geoff/'
skymap_base_url = 'https://ergsc.isee.nagoya-u.ac.jp/psa-pwing/pub/raw/lamp/geoff/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'lamp_phantom'


def lamp_phantom(
    location_code, time=None, time_range=None, redownload=False, missing_ok=True, alt=100
):
    """
    Create an Imager instance of LAMP's Phantom ASI.

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

    Example
    -------

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A THEMIS ASI instance with the time stamps, images, skymaps, and metadata.
    """
    if location_code.lower() == 'vee':
        meta = {
            'array': 'lamp_phantom',
            'location': 'VEE',
            'lat': 67.0135,
            'lon': -146.406,
            'alt': 0.2,
            'cadence': 0.01,
            'resolution': (640, 640),
        }
    else:
        raise NotImplementedError

    _skymap = load_skymap(location_code, alt, redownload)
    skymap = {
        'lat': _skymap['Lat'],
        'lon': _skymap['Lon'],
        'alt': alt,
        'el': _skymap['El'],
        'az': _skymap['Az'],
        'path': _skymap['path'],
    }

    file_paths = _get_files(location_code, time, time_range, redownload, missing_ok)

    start_times = len(file_paths) * [None]
    end_times = len(file_paths) * [None]
    for i, file_path in enumerate(file_paths):
        date_match = re.search(r'\d{4}', file_path.name)
        start_times[i] = datetime.strptime(f'20220305_{date_match.group()}', '%Y%m%d_%H%M')
        end_times[i] = start_times[i] + timedelta(minutes=1)

    file_info = {
        'path': file_paths,
        'start_time': start_times,
        'end_time': end_times,
        'loader': lamp_reader,
    }
    if time_range is not None:
        file_info['time_range'] = time_range
    else:
        file_info['time'] = time
    return asilib.Imager(file_info, meta, skymap)


def _get_files(location_code, time, time_range, redownload, missing_ok):
    """
    Get the LAMP Phantom image file paths.
    """
    if (time is None) and (time_range is None):
        raise ValueError('time or time_range must be specified.')
    elif (time is not None) and (time_range is not None):
        raise ValueError('both time and time_range can not be simultaneously specified.')

    local_dir = local_base_dir / 'images' / location_code.lower()

    if time is not None:
        time = utils.validate_time(time)
        filename = f'narrow_{time.strftime("%H%M")}.mat'
        file_paths = list(pathlib.Path(local_dir).rglob(filename))

        if (len(file_paths) == 0) or redownload:
            d = download.Downloader(image_base_url + f'{filename}')
            file_paths = d.download(local_dir, redownload=redownload, stream=True)
        return file_paths

    # Find multiple image files.
    if time_range is not None:
        time_range = utils.validate_time_range(time_range)
        file_times = utils.get_filename_times(time_range, dt='minutes')
        file_paths = []

        for file_time in file_times:
            filename = f'narrow_{file_time.strftime("%H%M")}.mat'
            _local_file_path = list(pathlib.Path(local_dir).rglob(filename))

            if (len(_local_file_path) == 0) or redownload:
                d = download.Downloader(image_base_url + f'{filename}')
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
        return file_paths


def lamp_reader(path):
    """
    Loads in the Phantom images chunk_size at a time to avoid a memory
    overflow. The .mat format is version 7.3 which are technically
    in the hdf5 format.
    """
    chunk_size = 600
    # The filename only contains the start hour and minute, so we calculate
    # t0 assuming the 2022-03-05 launch date.
    date_match = re.search(r'\d{4}', path.name)
    t0 = datetime.strptime(f'20220305_{date_match.group()}', '%Y%m%d_%H%M')

    with h5py.File(path, 'r') as f:
        image_keys = [key for key in f.keys() if 'images' in key]
        assert len(image_keys) == 1, f'{len(image_keys)} image keys found in the file {path}.'
        image_key = image_keys[0]
        dt = 60 / f[image_key].shape[0]

        for pos in range(0, f[image_key].shape[0], chunk_size):
            times = np.array([t0 + timedelta(seconds=dt * i) for i in range(pos, pos + chunk_size)])
            # ::-1 switches column to row major
            images = np.transpose(f[image_key][pos : pos + chunk_size], axes=(0, 2, 1))
            # images = f[image_key][pos:pos+chunk_size]
            yield times, images


def find_skymap(location_code, alt, redownload=True):
    """
    Find the path to the skymap file.
    """
    local_dir = local_base_dir / 'skymaps' / location_code.lower()

    # Check if the AzEl skymap is already downloaded.
    local_azel_paths = list(
        pathlib.Path(local_dir).rglob('Average_fps_im_-17700_to_-17600_AzEl.txt')
    )

    if (len(local_azel_paths) == 0) or redownload:
        # Download the AzEl skymaps.
        d = download.Downloader(skymap_base_url + 'Average_fps_im_-17700_to_-17600_AzEl.txt')
        local_azel_paths = d.download(local_dir, redownload=redownload)

    if len(local_azel_paths) != 1:
        raise FileNotFoundError(
            f'Unable to find the "Average_fps_im_-17700_to_-17600_AzEl.txt" LAMP skymap.'
        )

    # Check if the Lat/Lon skymap is already downloaded.
    local_latlon_paths = list(
        pathlib.Path(local_dir).rglob(f'Average_fps_im_-17700_to_-17600_LatLong{alt:03}km.txt')
    )

    if (len(local_latlon_paths) == 0) or redownload:
        # Download the LatLon skymaps.
        d = download.Downloader(
            skymap_base_url + f'Average_fps_im_-17700_to_-17600_LatLong{alt:03}km.txt'
        )
        local_latlon_paths = d.download(local_dir, redownload=redownload)

    if len(local_latlon_paths) != 1:
        raise FileNotFoundError(
            f'Unable to find the "Average_fps_im_-17700_to_-17600_AzEl.txt" LAMP skymap.'
        )
    return local_azel_paths[0], local_latlon_paths[0]


def load_skymap(location_code, alt, redownload):
    """
    Load the skymap file and apply the transformations.
    """
    azel_path, latlon_path = find_skymap(location_code, alt, redownload)
    skymap = _load_skymap(azel_path, latlon_path)
    # skymap = _tranform_longitude_to_180(skymap)
    # skymap = _transform_azimuth_to_180(skymap)
    # skymap = _flip_skymap(skymap)
    return skymap


def _load_skymap(azel_path, latlon_path):
    """
    A helper function to load a THEMIS skymap and transform it.
    """
    # Load the skymap file and convert it to a dictionary.
    azel_txt_data = pd.read_csv(azel_path, skipinitialspace=True, skiprows=29, delimiter=' ')
    latlon_txt_data = pd.read_csv(latlon_path, skipinitialspace=True, skiprows=29, delimiter=' ')
    skymap_dict = {
        'Az': azel_txt_data['Az'].to_numpy().reshape(640, 640),
        'El': azel_txt_data['El'].to_numpy().reshape(640, 640),
        'Lat': latlon_txt_data['Lat'].to_numpy().reshape(640, 640),
        'Lon': latlon_txt_data['Long'].to_numpy().reshape(640, 640),
        'path': [azel_path, latlon_path],
    }
    return skymap_dict


def _flip_skymap(skymap):
    """
    IDL is a column-major language while Python is row-major. This function
    tranposes the 2- and 3-D arrays to make them compatable with the images
    that are saved in row-major.
    """
    for key in skymap:
        if hasattr(skymap[key], 'shape'):
            shape = skymap[key].shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                skymap[key] = skymap[key][:, :]  # For Az/El maps.
    return skymap


def _tranform_longitude_to_180(skymap):
    """
    Transform the glo array from (0 -> 360) to (-180 -> 180).
    """
    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(skymap['glo']))
    skymap['glo'][valid_val_idx] = np.mod(skymap['glo'][valid_val_idx] + 180, 360) - 180
    return skymap


def _transform_azimuth_to_180(skymap):
    """
    Transform the azm array from (-180 -> 180) to (0 -> 360).
    """
    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(skymap['azm']))
    skymap['azm'][valid_val_idx] = np.mod(skymap['azm'][valid_val_idx], 360)
    return skymap


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec

    import asilib.map
    from asilib.asi.lamp_phantom import lamp_phantom

    asi = lamp_phantom(
        'vee',
        time=datetime(2022, 3, 5, 11, 32),
        redownload=False,
    )

    fig = plt.figure(figsize=(8, 4))
    gs = matplotlib.gridspec.GridSpec(1, 2, fig)
    ax = fig.add_subplot(gs[0, 0])
    bx = asilib.map.create_map(
        lon_bounds=(asi.meta['lon'] - 8, asi.meta['lon'] + 8),
        lat_bounds=(asi.meta['lat'] - 4, asi.meta['lat'] + 4),
        fig_ax=(fig, gs[0, 1]),
    )
    ax.axis('off')
    asi.plot_fisheye(ax=ax)
    asi.plot_map(ax=bx)
    plt.show()
