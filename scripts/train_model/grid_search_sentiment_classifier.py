import os
import pathlib
import mlflow

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import fire
from imblearn.over_sampling import RandomOverSampler

from restaurant_reviews_allergy.package_data.package_data import sentiment_training_data
from restaurant_reviews_allergy.review_parser.sentiment_classifier import SentimentClassifier
from restaurant_reviews_allergy.review_parser.sentiment_classifier_pyfunc import _log_model
from restaurant_reviews_allergy.utils.configs import read_model_config
from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger, download_data

def resample(train):
    train_x = train.drop(['sentiment'], axis=1)
    train_y = train['sentiment']
    train_x_, train_y_ = RandomOverSampler(random_state=123) \
        .fit_resample(train_x, train_y)
    train_ = pd.concat([train_x_, train_y_], axis=1)
    msg = f'Resampling data from {train.shape[0]} to {train_.shape[0]}'
    print(msg)
    return train

def main(config_name='single_model.json'):
    config = read_model_config(config_name)
    training_data = sentiment_training_data[
        sentiment_training_data['sentiment'].notnull()
        ]
    train, validation = train_test_split(
        training_data,
        test_size=0.2,
        random_state=123
        )
    train_ = resample(train)
    model = SentimentClassifier.train_with_grid_search(
        training_data = train_,
        validation_data = validation,
        config = config
        )
    validation_scores = model.validation_scores

    mlflow.set_experiment('restaurant-reviews-allergy')
    _log_model(model)
    mlflow.set_tag('step', 'grid_search_sentiment_classifier')
    mlflow.log_param('config_name', config_name)
    mlflow.log_metric('best_refit', validation_scores.get('best_refit'))
    logger = MlflowArtifactLogger()
    logger.add_artifact(config, 'model_config.json', 'json')
    logger.add_artifact(validation_scores, 'validation_scores.json', 'json')
    logger.log_artifacts('')

if __name__ == '__main__':
    fire.Fire(main)
