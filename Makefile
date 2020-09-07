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
SPLIT_SENTENCES_RUN_ID ?= bcde4de8d6be492195a9c44a20ef552e
ANNOTATE_DATA_RUN_ID ?= 4236fc9bc8cc420fbe3aeb286f75fe84
ANNOTATE_DATA_MODEL_ID ?= fd79683410634413950976f602c463a5
MODEL_CONFIG_NAME ?= grid_search.json
APPEND_SENTIMENT_RUN_ID ?= 3b65894f3bb34e70af8d08b054b9d83e


create-base-data-local:
		mlflow run . -e create_base_data \
		--experiment-name $(EXPERIMENT_NAME) \
		-P n_rows=$(N_ROWS) \
		--no-conda

split-sentences-local:
		mlflow run . -e split_sentences \
		--experiment-name $(EXPERIMENT_NAME) \
		-P run_id=$(SPLIT_SENTENCES_RUN_ID) \
		--no-conda

append-allergy-labels:
	mlflow run . -e append_allergy_labels \
	--experiment-name $(EXPERIMENT_NAME) \
	-P run_id=$(ANNOTATE_DATA_RUN_ID) \
	--no-conda

append-sentiment-labels:
	mlflow run . -e append_sentiment_labels \
	--experiment-name $(EXPERIMENT_NAME) \
	-P run_id=$(APPEND_SENTIMENT_RUN_ID) \
	-P model_run_id=$(ANNOTATE_DATA_MODEL_ID) \
	--no-conda

train-sentiment-local:
		mlflow run . -e train_sentiment \
		--experiment-name $(EXPERIMENT_NAME) \
		-P config_name=$(MODEL_CONFIG_NAME) \
		--no-conda
