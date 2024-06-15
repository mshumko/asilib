import pathlib
import shutil
import time

import asilib
from asilib.acknowledge import acknowledge

CONFIG_PATH = asilib.config['ASILIB_DIR'] / 'config.ini'

def test_always_acknowledge():
    try:
        shutil.copy(CONFIG_PATH, CONFIG_PATH.with_suffix('.backup'))
        assert acknowledge('test', dt=None) == True
        assert acknowledge('test', dt=None) == True
    except:
        raise
    finally:
        shutil.copy(CONFIG_PATH.with_suffix('.backup'), CONFIG_PATH)
        CONFIG_PATH.with_suffix('.backup').unlink(missing_ok=True)
    return

def test_delay_acknowledge():
    """
    The second call to acknowledge should raise False.
    """
    try:
        shutil.copy(CONFIG_PATH, CONFIG_PATH.with_suffix('.backup'))
        assert acknowledge('test', dt=2) == True
        assert acknowledge('test', dt=2) == False
    except:
        raise
    finally:
        shutil.copy(CONFIG_PATH.with_suffix('.backup'), CONFIG_PATH)
        CONFIG_PATH.with_suffix('.backup').unlink(missing_ok=True)
    return

def test_long_delay_acknowledge():
    """
    The second call to acknowledge should raise True.
    """
    try:
        shutil.copy(CONFIG_PATH, CONFIG_PATH.with_suffix('.backup'))
        assert acknowledge('test', dt=1) == True
        time.sleep(2)
        assert acknowledge('test', dt=1) == True
    except:
        raise
    finally:
        shutil.copy(CONFIG_PATH.with_suffix('.backup'), CONFIG_PATH)
        CONFIG_PATH.with_suffix('.backup').unlink(missing_ok=True)
    return