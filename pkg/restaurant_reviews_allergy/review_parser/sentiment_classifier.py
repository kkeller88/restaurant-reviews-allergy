import os
import pathlib
import itertools
import tempfile
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import keras
import keras.backend as K
from keras import layers
from tensorflow.python.keras.models import Model

from restaurant_reviews_allergy.package_data.package_data import sentiment_training_data


# TODO: Original model was 1.0; using compat but might want to look at updates
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

    def create_estimator(self, learning_rate=0.003, hidden_units=[500, 100],
                            optimizer='Adagrad', **kwargs):
        module_path = os.path.join(
            pathlib.Path(__file__).parents[3],
            'data',
            'universal_sentence_encoder'
            )
        input_text = tf.keras.layers.Input(shape=[], dtype=tf.string)
        embedding = hub.KerasLayer(module_path,trainable=True)(input_text)
        dense = layers.Dense(256, activation='relu')(embedding)
        pred = layers.Dense(1, activation='softmax')(dense)
        model = tf.keras.models.Model(input_text, pred)
        model.compile(loss='categorical_crossentropy',
        	optimizer='adam', metrics=['accuracy'])

        return model

    def train_estimator(self, data, **kwargs):
        data = data.copy()
        # TODO: function for this prep
        # TODO: dummies?
        data[self.outcome_name] = self.encode_outcome(data[self.outcome_name])
        train_text = data['sentences'].tolist()
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = np.asarray(data[self.outcome_name], dtype = np.int8)
        model = self.create_estimator(**kwargs)
        history = model.fit(train_text,
                train_label,
                validation_data=(train_text, train_label),
                epochs=10,
                batch_size=32)
        self.estimator = model

    def predict_estimator(self, data):
        predict_text = data['sentences'].tolist()
        predict_text = np.array(predict_text, dtype=object)[:, np.newaxis]
        predicted = self.estimator.predict(predict_text, batch_size=32)
        classes = [x[0] for x in predicted]
        return classes

    # TODO: Better eval
    def evaluate_estimator(self, data):
        predicted = self.predict_estimator(data)
        actual = self.encode_outcome(data[self.outcome_name])
        refit = sum(actual==predicted)/len(actual)
        return refit, {}

    @classmethod
    def train_with_grid_search(cls, config, training_data, validation_data):
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
        validation_scores_ = {
            'scores_and_params': validation_scores,
            'best_refit': refit
            }
        best_model.validation_scores = validation_scores_
        return best_model

    def save_model_artifacts(self, path_base=None):
        if path_base is None:
            path_base = tempfile.mkdtemp()
        model_path = os.path.join(path_base, 'model')
        encoder_path = os.path.join(path_base, 'label_encoder.pkl')
        self.estimator.save(model_path)
        with open(encoder_path, 'wb') as file:
            pickle.dump(self.encoder, file)
        return path_base

    @classmethod
    def load_model_artifacts(cls, path_base):
        model_path = os.path.join(path_base, 'model')
        encoder_path = os.path.join(path_base, 'label_encoder.pkl')
        model = tf.python.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as file:
            encoder = pickle.load(file)
        return cls(estimator=model, encoder=encoder)
