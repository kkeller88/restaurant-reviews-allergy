import uuid
from multiprocessing import Pool

import numpy as np
import mlflow
import fire

from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger, download_data
from restaurant_reviews_allergy.review_parser.allergen_search import simple_allergen_search

INCREMENT = 100000
N_PROCESSES = 4
DEFAULT_ALLERGENS = ['allergy', 'celiac', 'intolerant', 'intolerance', 'dairy', 'egg', 'gluten', 'soy', 'peanut', ' nut', 'shellfish', 'wheat', 'seafood']
ALLERGEN_COLS = ['is_' + allergen.replace(' ','_') for allergen in DEFAULT_ALLERGENS]

def chunk_dataframe(df):
    n_splits = max(1, int(INCREMENT/df.shape[0]))
    chunks = np.array_split(df, 3)
    return chunks

def apply_allergen_labels_and_log(chunk, allergens=DEFAULT_ALLERGENS):
    for allergen in allergens:
        chunk['is_' + allergen.replace(' ','_')] = [
            1 if simple_allergen_search(x.lower(), allergen)
            else 0
            for x in chunk['sentences']
            ]
    chunk = chunk[chunk[ALLERGEN_COLS].sum(axis=1) > 0]

    chunk_name = f'chunk_{uuid.uuid4().hex}.pkl'
    print(f'Logging file {chunk_name}!')
    logger = MlflowArtifactLogger()
    logger.add_artifact(chunk, chunk_name)
    logger.log_artifacts('chunks')

# TODO: Currently only allpwing DEFAULT_ALLERGENS; might want to update this in future
def main(run_id, allergens=DEFAULT_ALLERGENS):
    mlflow.set_experiment('restaurant-reviews-allergy')
    mlflow.log_param('run_id', run_id)
    mlflow.log_param('INCREMENT', INCREMENT)
    mlflow.log_param('N_PROCESSES', N_PROCESSES)
    mlflow.log_param('allergens', str(allergens))
    mlflow.set_tag('step', 'append_allergy_labels')

    sentences = download_data(run_id, 'sentences.pkl')
    chunks = chunk_dataframe(sentences)
    pool = Pool(processes=N_PROCESSES)
    pool.map(apply_allergen_labels_and_log, chunks)


if __name__ == '__main__':
    fire.Fire(main)
