import pandas as pd
import mlflow
import fire

from restaurant_reviews_allergy.utils.mlflow_ import MlflowArtifactLogger, download_data
from restaurant_reviews_allergy.review_parser.sentence_splitter import SentenceSplitter


# NOTE: Pulling out only sentences as IDs so we don't duplicate
#   large feilds like category and attribute
def _extract_sentence_data(base_data):
    sentences = base_data[['sentences', 'business_id', 'review_id']]
    sentences = sentences.explode('sentences')
    sentences['sentence_id'] = sentences \
        .groupby(['review_id']) \
        .cumcount()
    return sentences

def main(run_id):
    mlflow.set_experiment('restaurant-reviews-allergy')
    base_data = download_data(run_id, 'base_data.pkl')

    sentence_splitter = SentenceSplitter()
    base_data['sentences'] = [
        sentence_splitter.split_sentences(x)
        for x in base_data['text']
        ]
    sentences = _extract_sentence_data(base_data)

    mlflow.log_param('run_id', run_id)
    mlflow.set_tag('step', 'split_sentences')
    logger = MlflowArtifactLogger()
    logger.add_artifact(sentences, 'sentences.pkl')
    logger.log_artifacts('')

if __name__ == '__main__':
    fire.Fire(main)
