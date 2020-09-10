import re

import pandas as pd
import h3

def get_region(data, resolution=5):
    data['region_id'] = [
        h3.geo_to_h3(
            row['latitude'],
            row['longitude'],
            resolution
            )
        for ix, row in data.iterrows()
        ]
    return data

def standardize_city_name(x):
    x = x.strip().lower()
    x = re.sub('[ _]','-', x)
    return x

def get_region_name(data, standardize_city=True):
    # NOTE: copy?
    if standardize_city:
        data['city'] = [standardize_city_name(x) for x in data['city']]
    region_names = data \
        .groupby('region_id', as_index=False) \
        .agg(lambda x: '-'.join(pd.Series.mode(x))) \
        .rename(columns={'city': 'region_name'})
    data = data \
        .merge(
            region_names[['region_name', 'region_id']],
            on='region_id',
            how='left'
            )
    return data
