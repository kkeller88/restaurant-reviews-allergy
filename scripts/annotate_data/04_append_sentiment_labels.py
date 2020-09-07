import pandas as pd

import mlflow
from mlflow.pyfunc import load_model
import fire

from restaurant_reviews_allergy.utils.mlflow_ import MlflowArtifactLogger, download_chunked_data


def main(run_id, model_run_id):
    mlflow.set_experiment('restaurant-reviews-allergy')
    model_path = f'runs:/{model_run_id}/model'
    sentiment_model = load_model(model_path)
    chunks = download_chunked_data(run_id, 'chunks', format='pkl')
    scored_chunks = [
        sentiment_model.predict(chunk)
        for chunk in chunks
        ]
    df = pd.concat(scored_chunks)

    mlflow.log_param('run_id', run_id)
    mlflow.log_param('model_run_id', model_run_id)
    mlflow.set_tag('step', 'append_sentiment_labels')
    logger = MlflowArtifactLogger()
    logger.add_artifact(df, 'data_with_sentiment.pkl')
    logger.log_artifacts('')


if __name__ == '__main__':
    fire.Fire(main)
