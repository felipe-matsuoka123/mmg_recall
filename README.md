# MammoRecall

End-to-end pipeline to prepare mammogram PNG data, train a classifier, and evaluate results.
The workflow is designed to run locally or on a rented VM with a few commands.

## Requirements

- Python 3.9+
- GPU recommended for training
- Conda environment (recommended) or manual pip install

### Conda environment (recommended)

Create and activate the environment from `environment.yml`:
```bash
conda env create -f environment.yml
conda activate mammorecall
```

If you change `environment.yml`, update the env:
```bash
conda env update -f environment.yml --prune
```

### Pip install (alternative)

Python packages (install via your preferred method):
- torch, torchvision
- timm
- pandas, numpy, pillow
- pyyaml
- scikit-learn (optional, for AUC)
- wandb (optional, for experiment tracking)

Example install:
```bash
pip install torch torchvision timm pandas numpy pillow pyyaml scikit-learn wandb
```

## Project layout

- `scripts/prepare_data.py`: unzip PNGs + build `labels.csv` + optional train/val/test splits
- `scripts/train_cfg.py`: config-driven training (with W&B logging)
- `scripts/eval_cfg.py`: config-driven evaluation + prediction CSV
- `config/baseline.yaml`: main config file
- `Makefile`: convenience commands (`prepare`, `train`, `eval`)

## Quick start (3 commands)

1) Prepare data (unzips + labels + splits):
```bash
make prepare DATA_ZIP_DIR=/path/to/zips \
  DATA_OUT_DIR=data/processed/myset \
  LABELS_CSV=/path/to/labels.csv
```

2) Train:
```bash
make train CONFIG=config/baseline.yaml
```

3) Evaluate:
```bash
make eval CONFIG=config/baseline.yaml
```

## Data preparation details

### Inputs

- One or more `.zip` files containing PNGs
- A labels CSV with at least:
  - image filename column (default: `filename`)
  - label column (default: `label`)
  - optional patient/group column for leakage-safe split (default: `patient_id`)

### Output

`prepare_data.py` writes:
- extracted PNGs under `<out_dir>/images/`
- `<out_dir>/labels.csv` with a `png_path` column
- optional `train.csv`, `val.csv`, `test.csv`

### Common examples

Use a labels CSV with a different filename column:
```bash
python scripts/prepare_data.py \
  --zip_dir /path/to/zips \
  --out_dir data/processed/myset \
  --labels_csv /path/to/labels.csv \
  --img_col image_name \
  --make_splits
```

If your labels CSV already has a `png_path` column and you just want splits:
```bash
python scripts/prepare_data.py \
  --out_dir data/processed/myset \
  --labels_csv /path/to/labels.csv \
  --make_splits
```

Verify extracted files exist:
```bash
python scripts/prepare_data.py \
  --zip_dir /path/to/zips \
  --out_dir data/processed/myset \
  --labels_csv /path/to/labels.csv \
  --verify
```

## Configuration (config/baseline.yaml)

Key fields to update:

- `data.data_root`: folder containing `images/`
- `data.labels_csv`: labels CSV with `png_path`
- `data.train_csv`, `data.val_csv`, `data.test_csv`: optional split CSVs
- `data.img_col`, `data.label_col`, `data.id_col`: CSV column names
- `model.backbone`, `model.num_classes`
- `train.epochs`, `train.batch_size`, `train.lr`
- `project.out_dir`: where checkpoints and preds are saved

Example snippet:
```yaml
data:
  data_root: data/processed/myset
  labels_csv: data/processed/myset/labels.csv
  train_csv: data/processed/myset/train.csv
  val_csv: data/processed/myset/val.csv
  test_csv: data/processed/myset/test.csv
  img_col: png_path
  label_col: label
  id_col: patient_id
```

## Training

Run training using the config file:
```bash
python scripts/train_cfg.py --config config/baseline.yaml
```

Override config values from the CLI:
```bash
python scripts/train_cfg.py --config config/baseline.yaml \
  --override train.batch_size=16 model.backbone=resnet18
```

Outputs:
- `runs/<experiment>/best.pt`
- `runs/<experiment>/last.pt`

## Evaluation

Evaluate a split and produce predictions CSV:
```bash
python scripts/eval_cfg.py --config config/baseline.yaml --split test
```

Outputs:
- `runs/<experiment>/preds_test.csv`

You can specify a checkpoint:
```bash
python scripts/eval_cfg.py --config config/baseline.yaml --split val \
  --ckpt runs/exp01/best.pt
```

## W&B integration

Enable logging in the config:
```yaml
wandb:
  enable: true
  project: mammorecall
  entity: your_wandb_username
  run_name: baseline_resnet50
  tags: ["baseline"]
```

Make sure `wandb` is installed and `WANDB_API_KEY` is set on the VM.

## Notes for VM runs

- Upload or mount your data to the VM, then point `data.data_root` and `data.labels_csv` to those paths.
- Store outputs in a writable path (e.g. `project.out_dir: /mnt/runs/exp01`).
- If your data stays zipped, you can pass `data.zip_path` in the config to read directly from zips.

## Troubleshooting

- `ImportError: Please install timm`: run `pip install timm`
- `Missing column 'png_path'`: re-run `prepare_data.py` or set `data.img_col`
- `AUC is None`: scikit-learn not installed or only one class in the split
