import mlflow
from mlflow.pyfunc import load_model

from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger, download_data
from restaurant_reviews_allergy.review_parser.allergen_search import simple_allergen_search


def main(run_id, model_run_id, allergens):
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
    main(
        'ce4ebbe5c6b944cb80784756266cad31',
        '101c6a4489f64e019aefed02eb31ff43',
        ['gluten', 'soy', 'nut', 'a', 'the']
        )
