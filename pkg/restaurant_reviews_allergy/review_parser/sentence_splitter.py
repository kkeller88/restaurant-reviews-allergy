import re

from spacy.lang.en import English
from spacy.pipeline import Sentencizer, SentenceSegmenter

# TODO: remove spaces, normalize HTML
class SentenceSplitter(object):
    def __init__(self):
        self.nlp = self._create_spacy_pipeline()

    @staticmethod
    def _create_spacy_pipeline():
        nlp = English()
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
        return nlp

    @staticmethod
    def _clean_whitespace(text):
        text = re.sub('[\n\t]|[ ]+', ' ', text) \
            .strip() \
            .lower()
        return text

    def split_sentences(self, text):
        sentences = [
            self._clean_whitespace(x.text)
            for x in self.nlp(text).sents
            ]
        return sentences
