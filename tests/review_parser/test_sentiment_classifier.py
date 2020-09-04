import os
from pathlib import Path

import pandas as pd
import pytest

from restaurant_reviews_allergy.review_parser.sentiment_classifier import SentimentClassifier
from restaurant_reviews_allergy.utils.configs import read_model_config

from restaurant_reviews_allergy.package_data.package_data import sentiment_training_data

FIXTURE_PATH = os.path.join(Path(__file__).parents[1], 'fixtures')

@pytest.fixture
def training_data():
    data_path = os.path.join(FIXTURE_PATH, 'training_data.csv')
    data = pd.read_csv(data_path)
    return data

@pytest.fixture
def config():
    config = {
        'params': {
            'optimizer': ['Adagrad'],
            'learning_rate': [0.005, 0.05],
            'hidden_units': [[256, 100]],
            'embedding_model_trainable': ['False']
            }
        }
    return config

@pytest.mark.slow
def test_grid_search(training_data, config):
    model = SentimentClassifier.train_with_grid_search(
        training_data=training_data,
        validation_data = None,
        config = config
        )
    assert isinstance(model, SentimentClassifier)
