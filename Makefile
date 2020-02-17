# TODO: Need to fix up conda related commands, forgot about issues here
PKG ?= restaurant-reviews-allergy

conda-build: # activate conda
	conda env create -f conda.yaml
	conda activate $(PKG)

conda-rebuild: # activate conda
	conda deactivate
	conda env remove --name $(PKG)
	#conda-build

dev-env:
	conda-build
	python -m spacy download en_core_web_sm # if necessary, look at other options

develop:
	pip install -e .

build:
	python3 setup.py sdist bdist_wheel

download-use:
	mkdir data/universal_sentence_encoder
	curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC data/universal_sentence_encoder
