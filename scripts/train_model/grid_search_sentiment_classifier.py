import os
import pathlib
import mlflow

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import tensorflow as tf
import tensorflow_hub as hub

from restaurant_reviews_allergy.package_data.package_data import sentiment_training_data
from restaurant_reviews_allergy.review_parser.sentiment_classifier import SentimentClassifier
from restaurant_reviews_allergy.utils.configs import read_model_config

def main(config_name='single_model.json'):
    config = read_model_config(config_name)
    training_data = sentiment_training_data[
        sentiment_training_data['sentiment'].notnull()
        ]
    sentiment_classifier = SentimentClassifier()
    model = sentiment_classifier.train_with_grid_search(
        training_data = training_data,
        validation_data = training_data,
        config = config
        )
    validation_scores = model.validation_scores
    print(validation_scores)

    #mlflow.set_experiment('restaurant-reviews-allergy')
    #mlflow.set_tag('step', 'grid_search_sentiment_classifier')
    #mlflow.log_param('config_name', config_name)

if __name__ == '__main__':
    main('small_test.json')
