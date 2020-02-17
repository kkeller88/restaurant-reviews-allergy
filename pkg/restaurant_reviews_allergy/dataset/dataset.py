import os
from pathlib import Path
from datetime import datetime

import pandas as pd


PKG_DIR = Path(__file__).parents[3]
DATASET_DIR = os.path.join(PKG_DIR, "data", "dataset")


class Dataset(object):
    def __init__(self, directory=None):
        if directory is None:
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            self.directory = os.path.join(DATASET_DIR, current_time)
        else:
            self.directory = os.path.join(DATASET_DIR, directory)
        Path(self.directory).mkdir(parents=True, exist_ok=True)

    def save_data(self, data, name):
        path = os.path.join(self.directory, name + '.pkl')
        data.to_pickle(path)
        print(f'Saving data to {path}!')

    def load_data(self, name):
        path = os.path.join(self.directory, name + '.pkl')
        data = pd.read_pickle(path)
        print(f'Reading data from {path}!')
        return data
