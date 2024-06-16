import pathlib
import shutil
import time

import asilib
from asilib.acknowledge import acknowledge

CONFIG_PATH = asilib.config['ASILIB_DIR'] / 'config.ini'
CONFIG_EXISTS = CONFIG_PATH.exists()

def backup_config(test_function):
    def wrapper():
        try:
            if CONFIG_EXISTS:
                shutil.copy(CONFIG_PATH, CONFIG_PATH.with_suffix('.backup'))
            test_function()
        except:
            raise
        finally:
            if CONFIG_EXISTS:
                shutil.copy(CONFIG_PATH.with_suffix('.backup'), CONFIG_PATH)
                CONFIG_PATH.with_suffix('.backup').unlink(missing_ok=True)
            else:
                CONFIG_PATH.unlink(missing_ok=True)
    return wrapper

@backup_config
def test_always_acknowledge():
    assert acknowledge('test', dt=None) == True
    assert acknowledge('test', dt=None) == True
    return

@backup_config
def test_delay_acknowledge():
    """
    The second call to acknowledge should raise False.
    """
    assert acknowledge('test', dt=2) == True
    assert acknowledge('test', dt=2) == False
    return

@backup_config
def test_long_delay_acknowledge():
    """
    The second call to acknowledge should raise True.
    """
    assert acknowledge('test', dt=1) == True
    time.sleep(2)
    assert acknowledge('test', dt=1) == True
    return