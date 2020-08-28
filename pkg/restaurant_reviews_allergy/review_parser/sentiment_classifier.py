import os
import pathlib

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub


# TODO: Add in additional methods for cross validation, loading and saving
#   models, etc. Just laying down basic framework.
class SentimentClassifier(object):
    def __init__(self, estimator=None, encoder=None):
        self.outcome_name = "sentiment"
        self.estimator = estimator
        self.encoder = encoder

    def encode_outcome(self, outcome):
        try:
            outcome = self.encoder.transform(outcome)
            print('Using existing label encoder!')
        except:
            encoder = LabelEncoder()
            outcome = encoder.fit_transform(outcome)
            self.encoder = encoder
            print('New label encoder created!')
        return outcome

    def get_input_function(self, df, num_epochs=None, shuffle=False):
        df[self.outcome_name] = self.encode_outcome(df[self.outcome_name])
        input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
            df,
            df[self.outcome_name],
            num_epochs=1,
            shuffle=True
            )
        return input_fn

    def create_estimator(self):
        module_path = os.path.join(
            pathlib.Path(__file__).parents[3],
            'data',
            'universal_sentence_encoder'
            )

        embedded_text_feature_column = hub.text_embedding_column(
            key="sentences",
            module_spec=module_path
            )

        estimator = tf.compat.v1.estimator.DNNClassifier(
            hidden_units=[500, 100],
            feature_columns=[embedded_text_feature_column],
            n_classes=3,
            optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003)
            )

        return estimator

    def train_estimator(self, data):
        train_input_fn = self.get_input_function(
            data,
            num_epochs=None,
            shuffle=True
            )
        estimator = self.create_estimator()
        estimator.train(input_fn=train_input_fn, steps=50)
        self.estimator = estimator

    def predict_estimator(self, data):
        predict_input_fn = self.get_input_function(
            data
            )
        predicted = self.estimator.predict(input_fn=predict_input_fn)
        return list(predicted)
