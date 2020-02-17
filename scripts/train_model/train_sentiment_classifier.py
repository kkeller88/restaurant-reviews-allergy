import os
import pathlib

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub

from restaurant_reviews_allergy.package_data.package_data import sentiment_training_data
from restaurant_reviews_allergy.review_parser.sentiment_classifier import SentimentClassifier

def main():
    training_data = sentiment_training_data[
        sentiment_training_data['sentiment'].notnull()
        ]

    sc = SentimentClassifier()
    sc.train_estimator(training_data)
    predicted = sc.predict_estimator(training_data)
    print(predicted)

if __name__ == '__main__':
    main()
