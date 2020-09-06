import mlflow
from mlflow.pyfunc import load_model
import fire

from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger, download_data
from restaurant_reviews_allergy.review_parser.allergen_search import simple_allergen_search

DEFAULT_ALLERGENS = ['allergy', 'celiac', 'intolerant', 'intolerance', 'dairy', 'egg', 'gluten', 'soy', 'peanut', ' nut', 'shellfish', 'wheat', 'seafood']

def main(run_id, model_run_id, allergens=DEFAULT_ALLERGENS):
    mlflow.set_experiment('restaurant-reviews-allergy')
    model_path = f'runs:/{model_run_id}/model'
    sentiment_model = load_model(model_path)
    sentences = download_data(run_id, 'sentences.pkl')
    sentences['sentences'] = sentences['sentences'].str.lower()
    for allergen in allergens:
        sentences['is_' + allergen] = [
            1 if simple_allergen_search(x, allergen)
            else 0
            for x in sentences['sentences']
            ]
    sentences = sentiment_model.predict(sentences)
    mlflow.log_param('run_id', run_id)
    mlflow.log_param('model_run_id', model_run_id)
    mlflow.set_tag('step', 'annotate_data')
    logger = MlflowArtifactLogger()
    logger.add_artifact(sentences, 'annotated_data.pkl')
    logger.log_artifacts('')



if __name__ == '__main__':
    fire.Fire(main)
