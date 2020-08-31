import mlflow
from mlflow.pyfunc import load_model
import fire

from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger, download_data
from restaurant_reviews_allergy.review_parser.allergen_search import simple_allergen_search


def main(run_id, model_run_id, allergens=['gluten', 'soy', 'nut', 'a', 'the']):
    mlflow.set_experiment('restaurant-reviews-allergy')
    model_path = f'runs:/{model_run_id}/model'
    sentiment_model = load_model(model_path)
    sentences = download_data(run_id, 'sentences.pkl')
    for allergen in allergens:
        sentences['is_' + allergen] = [
            1 if simple_allergen_search(x, allergen)
            else 0
            for x in sentences['sentences']
            ]
    sentences = sentiment_model.predict(sentences)
    print(sentences)



if __name__ == '__main__':
    fire.Fire(main)
