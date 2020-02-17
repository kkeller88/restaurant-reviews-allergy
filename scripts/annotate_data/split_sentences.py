import pandas as pd

from restaurant_reviews_allergy.dataset.dataset import Dataset
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

def main(dataset_name):
    dataset = Dataset(dataset_name)
    base_data = dataset.load_data('base_data')

    sentence_splitter = SentenceSplitter()
    base_data['sentences'] = [
        sentence_splitter.split_sentences(x)
        for x in base_data['text']
        ]

    sentences = _extract_sentence_data(base_data)
    dataset.save_data(sentences, 'sentences')

if __name__ == '__main__':
    main(dataset_name='20200215134336')
