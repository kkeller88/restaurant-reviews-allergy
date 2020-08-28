import os
import re
import mlflow

import pandas as pd

from restaurant_reviews_allergy.dataset.base_data import create_base_data, _select_open_restaurants
from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger



def main(n_rows):
    base_data = create_base_data(n_rows)

    mlflow.set_experiment('restaurant-reviews-allergy')
    mlflow.log_param('n_rows', n_rows)
    mlflow.set_tag('step', 'create_base_data')
    logger = MlflowArtifactLogger()
    logger.add_artifact(base_data, 'base_data.pkl')
    logger.log_artifacts('')

if __name__ == '__main__':
    main(n_rows=100)
