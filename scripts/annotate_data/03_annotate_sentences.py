import mlflow

from restaurant_reviews_allergy.utils.mlflow import MlflowArtifactLogger, download_data
from restaurant_reviews_allergy.review_parser.allergen_search import simple_allergen_search


def main(run_id, allergens):
    mlflow.set_experiment('restaurant-reviews-allergy')
    sentences = download_data(run_id, 'sentences.pkl')

    for allergen in allergens:
        sentences['is_' + allergen] = [
            1 if simple_allergen_search(x, allergen)
            else 0
            for x in sentences['sentences']
            ]

    print(sentences.groupby(['is_gluten', 'is_soy']).count())



if __name__ == '__main__':
    main('ce4ebbe5c6b944cb80784756266cad31', ['gluten', 'soy', 'nut'])
