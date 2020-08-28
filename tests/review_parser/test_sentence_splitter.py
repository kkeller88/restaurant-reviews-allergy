from restaurant_reviews_allergy.review_parser.sentence_splitter import SentenceSplitter

def test_clean_whitespace():
    text = '    WOAH     '
    clean_text = SentenceSplitter._clean_whitespace(text)
    assert clean_text == 'woah'

def test_split_sentences_list_of_strings():
    text = 'This is one. This is two. This is three'
    sentences = SentenceSplitter().split_sentences(text)
    assert isinstance(sentences, list)
    assert isinstance(sentences[0], str)
