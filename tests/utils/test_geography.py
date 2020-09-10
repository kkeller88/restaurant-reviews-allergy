import pytest
import pandas as pd

from restaurant_reviews_allergy.utils.geography import (get_region,
    standardize_city_name, get_region_name)

@pytest.fixture
def data():
    data = pd.DataFrame({
        'latitude': [42.3601, 42.3736, 42.4793, 42.4184, 42.5047, 42.5048],
        'longitude': [71.0589, 71.1097, 71.1523, 71.1062, 71.1956, 71.1955],
        'id': ['bos', 'cam', 'wou', 'med', 'brl', 'fak'],
        'city': ['boston', 'Boston', 'medFord', 'woburn', 'burlington', 'fake']
        })
    return data

def test_get_region(data):
    data = get_region(data, 5)
    bos = data[data['id']=='bos']
    cam = data[data['id']=='cam']
    assert bos['region_id'].iloc[0] == cam['region_id'].iloc[0]

def test_standarize_city_name():
    city1 = ' maTH-town'
    city2 = 'MATH town'
    assert standardize_city_name(city1) == standardize_city_name(city2)

def test_get_region_name(data):
    data = get_region(data, 5)
    data = get_region_name(data)
    cam = data[data['id']=='cam']
    assert cam['region_name'].iloc[0] == 'boston'

def test_get_region_name_tie(data):
    data = get_region(data, 5)
    data = get_region_name(data)
    brl = data[data['id']=='brl']
    assert brl['region_name'].iloc[0] == 'burlington-fake'
