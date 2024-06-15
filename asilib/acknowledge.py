"""
Print an acknowledgment statement once per ASI import. This leaves
"""
from datetime import datetime
import configparser
import dateutil.parser

import asilib

CONFIG_PATH = asilib.config['ASILIB_DIR'] / 'config.ini'

def acknowledge(asi:str, dt:float=None) -> bool:
    """
    Returns True if an acknowledgment has never been made, or has been dt since the last
    acknowledgment.

    Parameters
    ----------
    asi:str
        The ASI name, e.g., "THEMIS", "TREx-NIR".
    dt:float
        The maximum elapsed time in seconds between printed acknowledgments. If None will always
        print.
    
    Returns
    -------
    bool
        True if an acknowledgement should be made and False otherwise.
    """
    if dt is None:
        return True
    
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    if config.has_option('acknowledged_asis', asi):
        last_acknowledged = dateutil.parser.parse(config['acknowledged_asis'][asi])
        if (datetime.now() - last_acknowledged).total_seconds() > dt:
            config['acknowledged_asis'] = dict(config['acknowledged_asis']) | {asi: datetime.now().isoformat()}
            _write_config(config)
            return True
        else:
            return False
    else:
        if config.has_section('acknowledged_asis'):
            config['acknowledged_asis'] = dict(config['acknowledged_asis']) | {asi: datetime.now().isoformat()}
        else:
            config['acknowledged_asis'] = {asi: datetime.now().isoformat()}
        _write_config(config)
        return True
    
def _write_config(config:configparser.ConfigParser):
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)
    return