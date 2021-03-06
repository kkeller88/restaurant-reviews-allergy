import os
import re

import pandas as pd

from restaurant_reviews_allergy.utils.data import read_review_file


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

def create_base_data(n_rows, **kwargs):
    reviews = read_review_file("review", n_rows, **kwargs)
    businesses = read_review_file("business", n_rows, **kwargs);
    open_restaurants = _select_open_restaurants(businesses)
    dataset = reviews[REVIEW_COLS] \
        .merge(
            open_restaurants[BUSINESS_COLS],
            on = 'business_id',
            how = 'inner'
            )
    return dataset
