import os
from pathlib import Path
import json
import itertools

import pandas as pd

PKG_DIR = Path(__file__).parents[3]
DATA_DIR = os.path.join(PKG_DIR, "data", "yelp_dataset")
VALID_FILE_NAMES = ('business', 'checkin', 'photo', 'review', 'tip', 'user')

def _check_file_name(file_name):
    if file_name not in VALID_FILE_NAMES:
        valid_file_names = ", ".join(VALID_FILE_NAMES)
        msg = f"invalid file name {file_name} provided! " \
            f"Choose one of {valid_file_names}."
        raise ValueError(msg)

# TODO: Get all data if n_rows is None
def read_review_file(file_name, n_rows=100, data_dir=DATA_DIR):
    print(f'Reading file {file_name}')
    _check_file_name(file_name)
    review_data = os.path.join(data_dir, file_name + ".json")
    if n_rows > 0:
        records = []
        f = open(review_data)
        for line in itertools.islice(f, n_rows):
          line = line.strip()
          if not line: continue
          records.append(json.loads(line))
        f.close()
    else:
        f = open(review_data)
        records = [
            json.loads(line.strip())
            for line in f.readlines()
            ]
        f.close()
    data = pd.DataFrame(records)
    return data
