import sys
import os
KEYCONFIG_FILE_PATH = BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/visualizer/seckey.ini'


if sys.version_info[0] == 2:
    PYTHON_MAJOR_VER = 2
elif sys.version_info[0] == 3:
    PYTHON_MAJOR_VER = 3
if PYTHON_MAJOR_VER == 2:
    import ConfigParser as configparser
elif PYTHON_MAJOR_VER == 3:
    import configparser as configparser


def parser():
    config = configparser.ConfigParser()
    if len(config.read(KEYCONFIG_FILE_PATH)) == 0:
        raise FileNotFoundError('DOES NOT EXIST CONFIG FILE!')
    return config

if __name__ == "__main__":
    config = parser()
    print(config['PLOTLY']['plotly_username'])
    print(config['PLOTLY']['plotly_apikey'])