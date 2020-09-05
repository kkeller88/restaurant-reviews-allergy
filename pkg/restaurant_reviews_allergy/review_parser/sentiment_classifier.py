import os
import pathlib
import itertools
import tempfile
import pickle

import pandas as pd
import numpy as np
from sklearn import metrics as m
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import keras
import keras.backend as K
from keras import layers
from tensorflow.python.keras.models import Model

from restaurant_reviews_allergy.package_data.package_data import sentiment_training_data


class SentimentClassifier(object):
    def __init__(self, estimator=None, encoder=None):
        self.outcome_name = "sentiment"
        self.estimator = estimator
        self.encoder = encoder

    def encode_outcome(self, outcome):
        outcome = outcome.to_numpy().reshape(-1, 1)
        if self.encoder is not None:
            outcome = self.encoder.transform(outcome)
            print('Using existing one hot encoder!')
        else:
            encoder = OneHotEncoder(sparse=False)
            outcome = encoder.fit_transform(outcome)
            self.encoder = encoder
            print('New one hot encoder created!')
        return outcome

    def decode_outcome(self, outcome):
        outcome = self.encoder.inverse_transform(outcome).ravel()
        return outcome

    def create_estimator(self, hidden_units=[256], optimizer='Adagrad',
                            embedding_model_trainable=False, **kwargs):
        def get_dense_layers(embedding, hidden_units):
            previous_layer = embedding
            for n_units in hidden_units:
                print(f'Adding dense layer with {n_units} units!')
                current_layer = layers.Dense(n_units, activation='relu')(previous_layer)
                previous_layer = current_layer
            return previous_layer

        module_path = os.path.join(
            pathlib.Path(__file__).parents[3],
            'data',
            'universal_sentence_encoder'
            )
        n_categories = len(self.encoder.categories_[0])
        input_text = tf.keras.layers.Input(shape=[], dtype=tf.string)
        embedding = hub.KerasLayer(module_path,trainable=embedding_model_trainable)(input_text)
        dense = get_dense_layers(embedding, hidden_units)
        pred = layers.Dense(n_categories, activation='softmax')(dense)
        model = tf.keras.models.Model(input_text, pred)
        model.compile(loss='categorical_crossentropy',
        	optimizer=optimizer, metrics=['accuracy'])

        return model

    def train_estimator(self, data, **kwargs):
        data = data.copy()
        encoded_outcome = self.encode_outcome(data[self.outcome_name])
        outcome_categories = list(self.encoder.categories_[0])
        data[outcome_categories] = encoded_outcome
        train_text = data['sentences'].tolist()
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = np.asarray(data[outcome_categories], dtype = np.int8)
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
        predicted = self.decode_outcome(predicted)
        return predicted

    # TODO: Log loss
    def evaluate_estimator(self, data):
        predicted = self.predict_estimator(data)
        actual = data[self.outcome_name]
        f1_macro = m.f1_score(actual, predicted, average='macro')
        f1_weighted = m.f1_score(actual, predicted, average='weighted')
        accuracy = m.accuracy_score(actual, predicted)
        refit = f1_macro
        score = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy
            }
        return refit, score

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
        print(f'Logging data from {path_base}')
        return path_base

    @classmethod
    def load_model_artifacts(cls, path_base):
        if 'model' not in os.listdir(path_base):
            path_base = os.join(path_base, os.listdir(path_base)[0])
        model_path = os.path.join(path_base, 'model')
        encoder_path = os.path.join(path_base, 'label_encoder.pkl')
        model = tf.python.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as file:
            encoder = pickle.load(file)
        return cls(estimator=model, encoder=encoder)
