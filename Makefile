PY ?= python
CONFIG ?= config/baseline.yaml

DATA_ZIP_DIR ?= data/raw/zips
DATA_OUT_DIR ?= data/processed/dataset
LABELS_CSV ?= data/raw/labels.csv
PREPARE_ARGS ?=

.PHONY: prepare train eval

prepare:
	$(PY) scripts/prepare_data.py \
		--zip_dir $(DATA_ZIP_DIR) \
		--out_dir $(DATA_OUT_DIR) \
		--labels_csv $(LABELS_CSV) \
		--make_splits \
		$(PREPARE_ARGS)

train:
	$(PY) scripts/train_cfg.py --config $(CONFIG)

eval:
	$(PY) scripts/eval_cfg.py --config $(CONFIG) --split test
