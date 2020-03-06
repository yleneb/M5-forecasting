import pandas as pd
from zipfile import ZipFile
import os
import pathlib

ROOT = pathlib.Path().absolute()
RAW_DATA_PATH = ROOT / 'data' / 'raw'

def fetch_data(raw_data_path=RAW_DATA_PATH):
    """Downloads the data from kaggle, unzips, and adds to data folder"""
    # download the data
    os.system('kaggle competitions download -c m5-forecasting-accuracy')
    
    # Create a ZipFile Object and load .zip in it
    with ZipFile('m5-forecasting-accuracy.zip', 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(raw_data_path)
        
    # delete the zip file
    os.remove('m5-forecasting-accuracy.zip')

# fetch_data()