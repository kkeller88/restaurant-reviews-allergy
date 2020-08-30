import os
import pickle
import tempfile

import tensorflow as tf
from mlflow.pyfunc import PythonModel, log_model
from restaurant_reviews_allergy.review_parser.sentiment_classifier import SentimentClassifier
import restaurant_reviews_allergy as rra

# TODO: Update to load model
def _load_pyfunc(path):
    sentiment_classifier = SentimentClassifier.load_model_artifacts(path)
    return SentimentPythonModel(sentiment_classifier)

class SentimentPythonModel(PythonModel):
    def __init__(self, sentiment_classifier):
        self.sentiment_classifier = sentiment_classifier

    def predict(self, data):
        """
        Takes a dataframe with a column called "sentences".
        Returns dataframe with a column called "predicted".
        """
        data['predicted'] = self.sentiment_classifier.predict_estimator(data)
        return data

# TODO: Add conda env
def _log_model(sentiment_classifier):
    artifact_path = sentiment_classifier.save_model_artifacts()
    log_model(
        artifact_path = 'model',
        loader_module=__name__,
        data_path=artifact_path
    )
