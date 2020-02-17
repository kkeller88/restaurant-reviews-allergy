import os
import re

import pandas as pd

from restaurant_reviews_allergy.utils.data import read_review_file
from restaurant_reviews_allergy.dataset.dataset import Dataset


REVIEW_COLS = ['review_id', 'user_id', 'business_id', 'text', 'date']
BUSINESS_COLS = ['business_id', 'name', 'city', 'state',
                'latitude', 'longitude', 'stars', 'attributes', 'categories']


def _select_open_restaurants(businesses):
    businesses['is_food'] = [
        1 if re.search('Restaurant|Pubs|Bakeries|Food', x)
        else 0
        for x in businesses['categories'].fillna('')
        ]
    businesses = businesses[
        (businesses['is_open']==1) &
        (businesses['is_food']==1)
        ]
    return businesses

def create_base_data(n_rows):
    reviews = read_review_file("review", n_rows)
    businesses = read_review_file("business", n_rows);
    open_restaurants = _select_open_restaurants(businesses)
    dataset = reviews[REVIEW_COLS] \
        .merge(
            open_restaurants[BUSINESS_COLS],
            on = 'business_id',
            how = 'inner'
            )
    return dataset

def main(n_rows):
    base_data = create_base_data(n_rows)
    dataset = Dataset()
    dataset.save_data(base_data, 'base_data')


if __name__ == '__main__':
    main(n_rows=200000)
