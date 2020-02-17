import os
import re
import pandas as pd
from restaurant_reviews_allergy.dataset.dataset import Dataset, PKG_DIR

ALLERGY_EXPRESSIONS = 'allergy|allergies|allergic|celiac|gluten|tree nut'

def main(dataset_name='20200215134336'):
    dataset = Dataset(dataset_name)
    sentences = dataset.load_data('sentences')

    is_allergen = [
        bool(re.search(ALLERGY_EXPRESSIONS, x))
        for x in sentences['sentences']
        ]
    sentences_with_allergens = sentences[is_allergen]
    print('{n} sentences mentioning allegies!'.format(
        n=sentences_with_allergens.shape[0]))

    path = os.path.join(PKG_DIR, 'data', 'sentiment_training_data_.csv')
    sentences_with_allergens.to_csv(path, index=False)

if __name__ == '__main__':
    main()
