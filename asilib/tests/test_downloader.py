"""
Tests the asilib's Downloader class.
"""
import pytest

import asilib
import asilib.io.download as download

def test_bad_url():
    """ 
    Checks that Downloader() raises an error for a non-existant URL.
    """
    with pytest.raises(ConnectionError):
        download.Downloader('https://data.phys.ucalgary.ca/sort_by_project/FAKE/')
    return


def test_find_file():
    """
    Looks for a file named 20140505_0505_gill_themis19_full.pgm.gz on the U Calgary server.
    """
    d = download.Downloader('https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/')
    d.find_url(subdirectories=['2014', '05', '05', 'gill*', 'ut05'], filename='20140505_0505_gill*.pgm.gz')
    assert d.url[0].split('/')[-1] == '20140505_0505_gill_themis19_full.pgm.gz'
    return


def test_download_file():
    """
    Looks for and downloads a file named 20140505_0505_gill_themis19_full.pgm.gz on 
    the U Calgary server.
    """
    d = download.Downloader('https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/')
    d.find_url(subdirectories=['2014', '05', '05', 'gill*', 'ut05'], filename='20140505_0505_gill*.pgm.gz')
    save_dir = asilib.config['ASI_DATA_DIR'] / 'themis' / 'images' / 'gill'
    save_path = save_dir / '20140505_0505_gill_themis19_full.pgm.gz'
    d.download(save_path, overwrite=True)
    assert save_path.exists()
    return


def test_download_file_no_subdirectories():
    """
    Test Download.download() for the case where no subdirectories were provided (the filename)
    is in the base_url.
    """
    d = download.Downloader('https://data.phys.ucalgary.ca/sort_by_project//THEMIS/asi/stream0/2011/07/07/pina_themis18/ut05/')
    d.find_url(filename='20110707_0500_pina_*.pgm.gz')
    save_dir = asilib.config['ASI_DATA_DIR'] / 'themis' / 'images' / 'pina'
    save_path = save_dir / '20110707_0500_pina_themis18_full.pgm.gz'
    d.download(save_path, overwrite=True)
    assert save_path.exists()
    return


def test_find_multiple_directories():
    """
    Test Download.download() for the case where no subdirectories were provided (the filename)
    is in the base_url.
    """
    d = download.Downloader('https://data.phys.ucalgary.ca/sort_by_project//THEMIS/asi/stream0')
    d.find_url(
        subdirectories=['2011', '07', '07', 'pina_*', 'ut05'],
        filename='20110707_050*_pina_*.pgm.gz'
        )
    assert d.url[0].split('/')[-1] == '20110707_0500_pina_themis18_full.pgm.gz'
    assert d.url[-1].split('/')[-1] == '20110707_0509_pina_themis18_full.pgm.gz'
    save_dir = asilib.config['ASI_DATA_DIR'] / 'themis' / 'images' / 'pina'
    save_path = save_dir / '20110707_0500_pina_themis18_full.pgm.gz'
    d.download(save_path, overwrite=True)
    assert d.save_path[0].name == '20110707_0500_pina_themis18_full.pgm.gz'
    assert d.save_path[-1].name == '20110707_0509_pina_themis18_full.pgm.gz'
    return
