import os
import pathlib
import itertools

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
        if self.encoder is not None:
            outcome = self.encoder.transform(outcome)
            print('Using existing label encoder!')
        else:
            encoder = LabelEncoder()
            outcome = encoder.fit_transform(outcome)
            self.encoder = encoder
            print('New label encoder created!')
        return outcome

    def get_input_function(self, df, num_epochs=None, shuffle=False):
        df = df.copy()
        df[self.outcome_name] = self.encode_outcome(df[self.outcome_name])
        input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
            df,
            df[self.outcome_name],
            num_epochs=1,
            shuffle=True
            )
        return input_fn

    # TODO: Optimizer, is model trainable
    def create_estimator(self, learning_rate=0.003, hidden_units=[500, 100], **kwargs):
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
            hidden_units=hidden_units,
            feature_columns=[embedded_text_feature_column],
            n_classes=3,
            optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate)
            )

        return estimator

    def train_estimator(self, data, **kwargs):
        train_input_fn = self.get_input_function(
            data,
            num_epochs=None,
            shuffle=True
            )
        estimator = self.create_estimator(**kwargs)
        estimator.train(input_fn=train_input_fn, steps=50)
        self.estimator = estimator

    def predict_estimator(self, data):
        predict_input_fn = self.get_input_function(
            data
            )
        predicted = self.estimator.predict(input_fn=predict_input_fn)
        classes = [int(x.get('classes')[0]) for x in predicted]
        return list(classes)

    # TODO: Better eval
    def evaluate_estimator(self, data):
        predicted = self.predict_estimator(data)
        actual = self.encode_outcome(data[self.outcome_name])
        refit = sum(actual==predicted)/len(actual)
        return refit, {}

    @classmethod
    def train_with_grid_search(cls, training_data, config, validation_data=None):
        if validation_data is None:
            validation_data = training_data
        params = config.get('params')
        param_grid = (dict(zip(params, x)) for x in itertools.product(*params.values()))

        validation_scores = []
        best_refit = 0
        for param_set in param_grid:
            clf = cls()
            clf.train_estimator(data=training_data, **param_set)
            refit, scores = clf.evaluate_estimator(validation_data)
            validation_scores.append({
                'refit': refit,
                'scores': scores,
                'params':param_set}
                )
            if refit >= best_refit:
                best_model = clf
        best_model.validation_scores = validation_scores
        return best_model
