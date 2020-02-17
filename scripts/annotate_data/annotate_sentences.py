from restaurant_reviews_allergy.dataset.dataset import Dataset
from restaurant_reviews_allergy.review_parser.allergen_search import simple_allergen_search


def main(dataset_name, allergens):
    dataset = Dataset(dataset_name)
    sentences = dataset.load_data('sentences')

    for allergen in allergens:
        sentences['is_' + allergen] = [
            1 if simple_allergen_search(x, allergen)
            else 0
            for x in sentences['sentences']
            ]

    print(sentences.groupby(['is_gluten', 'is_soy']).count())



if __name__ == '__main__':
    main('20200215112441', ['gluten', 'soy', 'nut'])
