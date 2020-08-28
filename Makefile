PKG ?= restaurant-reviews-allergy

conda-build:
	PYENV_VERSION=miniconda3-latest conda env create -f ./conda-dev.yaml

dev-env:
	conda-build
	python -m spacy download en_core_web_sm # if necessary, look at other options

develop:
	pip install -e .

build:
	python3 setup.py sdist 

download-use:
	mkdir data/universal_sentence_encoder
	curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC data/universal_sentence_encoder
