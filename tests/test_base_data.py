import os
from pathlib import Path
import pytest
import pandas as pd

from restaurant_reviews_allergy.utils.data import read_review_file
from restaurant_reviews_allergy.dataset.base_data import create_base_data, _select_open_restaurants

FIXTURE_PATH = os.path.join(Path(__file__).parents[0], 'fixtures')

@pytest.fixture
def toy_business_data():
    return pd.DataFrame({
        'id': [0,1,2,3,4],
        'categories': ['Dummy', 'Dummy', 'Dummy', 'Food', 'Food'],
        'is_open': [0,0,1,1,1]
        })

def test_select_open_restaurants(toy_business_data):
    open_restaurants = _select_open_restaurants(toy_business_data)
    assert set(open_restaurants['id']) == {3, 4}

def test_base_data():
    base_data = create_base_data(10, data_dir=FIXTURE_PATH)
    assert isinstance(base_data, pd.DataFrame)
    assert len(base_data) > 0
