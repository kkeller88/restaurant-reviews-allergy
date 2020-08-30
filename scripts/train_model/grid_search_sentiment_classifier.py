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
from restaurant_reviews_allergy.review_parser.sentiment_classifier_pyfunc import _log_model
from restaurant_reviews_allergy.utils.configs import read_model_config
from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger, download_data

def main(config_name='single_model.json'):
    config = read_model_config(config_name)
    training_data = sentiment_training_data[
        sentiment_training_data['sentiment'].notnull()
        ]
    model = SentimentClassifier.train_with_grid_search(
        training_data=training_data,
        validation_data = None,
        config = config
        )
    validation_scores = model.validation_scores
    #artifact_path = model.save_model_artifacts()
    #print(artifact_path)
    #print(artifact_path)

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
    main('small_test.json')
