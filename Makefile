PKG ?= restaurant-reviews-allergy
EXPERIMENT_NAME ?= restaurant-reviews-allergy
GIT_REPO ?=

conda-build:
	PYENV_VERSION=miniconda3-latest conda env create -f ./conda-dev.yaml

download-use:
	mkdir data/universal_sentence_encoder
	curl -L "https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed" | tar -zxvC data/universal_sentence_encoder

dev-env:
	conda-build
	python -m spacy download en_core_web_sm # if necessary, look at other options
	download-use

develop:
	pip install -e .

build:
	python3 setup.py sdist


N_ROWS ?= 100

create-base-data-local:
		mlflow run . -e create_base_data \
		--experiment-name $(EXPERIMENT_NAME) \
		-P n_rows=$(N_ROWS) \
		--no-conda
