import pandas as pd
import pytest

import restaurant_reviews_allergy.utils.analysis as analysis

@pytest.fixture
def data():
    data = pd.DataFrame([
            [0, '0', '1', 'positive'],
            [0, '0', '1', 'positive'],
            [0, '1', '0', 'negtive'],
            [0, '1', '0', 'negative'],
            [1, '1', '0', 'positive'],
            [1, '1', '0', 'positive'],
            [1, '1', '0', 'negative'],
            [1, '1', '0', 'neutral'],
            ],
        columns = ['business_id', 'al1', 'al2', 'predicted']
        )
    return data

def test_calculate_weighted_sentiment():
    scores = ['positive', 'positive', 'negative', 'neutral']
    weighted_score = analysis.calculate_weighted_sentiment(scores)
    assert weighted_score == 0.5

def test_calculate_weighted_sentiment_empty():
    scores = []
    weighted_score = analysis.calculate_weighted_sentiment(scores)
    assert weighted_score == 0

def test_aggregate_sentiment_by_business(data):
    data_ = analysis.aggregate_sentiment_by_business(data)
    business1 = data_[data_['business_id'] == 1]
    assert data_.shape[0] == 2
    assert business1['sentiment_score'].iloc[0] > 0

def test_aggregate_sentiment_by_business_allergy(data):
    data_ = analysis.aggregate_sentiment_by_business_allergy(data, ['al1', 'al2'])
    print(data_)
    business0_al1 = data_[(data_['business_id'] == 0) & (data_['allergen'] == 'al1')]
    assert data_.shape[0] == 3
    assert business0_al1['sentiment_score'].iloc[0] < 0
