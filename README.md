# MammoRecall

Train recall prediction models for mammography images using processed RSNA, SPR,
and VinDr datasets. The repo is organized around reproducible experiment configs:
the config selects data artifacts, model, optimizer, split policy, W&B logging,
and run output paths.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate mammorecall
```

List available experiments:

```bash
python scripts/experiment.py list
```

Run a local smoke experiment:

```bash
python scripts/experiment.py train smoke_all_grayscale_convnext_tiny \
  --max-samples 64 \
  --run-dir runs/smoke_debug \
  --device auto \
  --no-wandb
```

Run tests:

```bash
python -m pytest
```

## Repository Layout

```text
config/cloud_exp/   cloud-oriented experiment configs
config/local_exp/   small/local experiment configs
config/env/         machine-specific path templates
docs/               deeper notes
scripts/            data prep, training, inference, cloud helpers
src/mammorecall/    reusable training/data/model code
tests/              regression tests for the experiment workflow
```

Generated data, model checkpoints, W&B state, and submissions should stay out of
git.

## Data Artifacts

Training configs expect processed data under:

```text
processed_datasets/
  combined_mammo_recall_labels_birads_only.csv
  combined_mammo_recall_splits_spr_holdout.csv
  rsna/
    grayscale.zip
    rgb_multiwindow.zip
  spr/
    grayscale.zip
    rgb_multiwindow.zip
  vindr/
    grayscale.zip
    rgb_multiwindow.zip
```

For cloud work, copy those same files to R2 using this remote layout:

```text
s2:mammo-recall-data/
  processed_datasets/
    combined_mammo_recall_labels_birads_only.csv
    combined_mammo_recall_splits_spr_holdout.csv
    rsna/grayscale.zip
    rsna/rgb_multiwindow.zip
    spr/grayscale.zip
    spr/rgb_multiwindow.zip
    vindr/grayscale.zip
    vindr/rgb_multiwindow.zip
```

`scripts/download_data.py` downloads the label CSV, fixed split CSV, and selected
dataset zips.

## Local Experiments

Put or symlink processed artifacts into `processed_datasets/`:

```bash
ln -s /path/to/processed_datasets processed_datasets
```

Check a config:

```bash
python scripts/experiment.py preflight all_grayscale_convnext_tiny_1024
```

Train:

```bash
python scripts/experiment.py train all_grayscale_convnext_tiny_1024 \
  --run-dir runs/all_grayscale_convnext_tiny_1024 \
  --wandb
```

Useful overrides:

```bash
--run-dir runs/name
--max-samples 128
--device cpu
--wandb / --no-wandb
--wandb-run-name name
```

All other experiment settings belong in the YAML config.

## Cloud VM Workflow

Default cloud paths:

```bash
DATA_ROOT=/mnt/data/processed_datasets
OUTPUT_ROOT=/mnt/outputs
TORCH_HOME=/mnt/outputs/torch-cache
REMOTE=s2:mammo-recall-data
```

On a fresh VM:

```bash
conda env create -f environment.yml
conda activate mammorecall

scripts/setup_vm.sh
export TORCH_HOME=/mnt/outputs/torch-cache
```

`setup_vm.sh` creates:

```text
/mnt/data/processed_datasets/
/mnt/outputs/
/mnt/outputs/torch-cache/
processed_datasets -> /mnt/data/processed_datasets
```

Configure rclone for Cloudflare R2 before downloading:

```bash
rclone config
rclone lsd s2:mammo-recall-data
```

Download grayscale artifacts:

```bash
REMOTE=s2:mammo-recall-data DATA_ROOT=/mnt/data/processed_datasets \
  scripts/download_data.sh grayscale
```

Download RGB multi-window artifacts:

```bash
REMOTE=s2:mammo-recall-data DATA_ROOT=/mnt/data/processed_datasets \
  scripts/download_data.sh rgb_multiwindow
```

Run a cloud smoke check:

```bash
scripts/run_smoke.sh grayscale
scripts/run_smoke.sh rgb_multiwindow
```

Run full cloud training:

```bash
scripts/run_train.sh grayscale
scripts/run_train.sh rgb_multiwindow
```

`run_train.sh` writes runs under `/mnt/outputs/<run_name>` and uploads successful
runs back to:

```text
s2:mammo-recall-data/runs/<run_name>
```

Disable upload:

```bash
UPLOAD_RESULTS=0 scripts/run_train.sh grayscale
```

Override a cloud config or run name:

```bash
CONFIG=config/cloud_exp/all_rgb_multiwindow_convnext_tiny_1024_weighed_loss.yaml \
RUN_NAME=all_rgb_multiwindow_convnext_tiny_1024_weighted_loss \
scripts/run_train.sh rgb_multiwindow
```

## Paths You May Need To Change

For training experiments:

```text
config/cloud_exp/*.yaml
config/local_exp/*.yaml
```

These point to processed zips, label CSVs, split CSVs, run names, model settings,
and W&B settings. Prefer changing these configs instead of adding CLI flags.

For raw dataset preprocessing and label manifest creation:

```text
config/env/paths_data_preprocessing.yaml
```

Create it from the example:

```bash
cp config/env/paths_data_preprocessing.example.yaml config/env/paths_data_preprocessing.yaml
```

Edit it for the current machine or VM:

```yaml
vindr_csv: /path/to/VinDr_Mammo/breast-level_annotations.csv
vindr_dicom_dir: /path/to/VinDr_Mammo/images
rsna_csv: /path/to/rsna_breast/train.csv
rsna_dicom_dir: /path/to/rsna_breast/train_images
spr_csv: /path/to/SPR_Mammo_Recall_train.csv
spr_dicom_dir: /path/to/SPR_Mammo_Recall
output: processed_datasets/combined_mammo_recall_labels.csv
```

Then create labels:

```bash
python scripts/create_recall_label_manifest.py \
  --paths-config config/env/paths_data_preprocessing.yaml \
  --rsna-label-mode birads_only
```

Create a fixed split:

```bash
python scripts/create_split_manifest.py \
  processed_datasets/combined_mammo_recall_labels_birads_only.csv \
  processed_datasets/combined_mammo_recall_splits_spr_holdout.csv
```

## Run Outputs

Each run directory contains:

```text
config.json
label_map.json
metrics.jsonl
train_rows.csv
val_rows.csv
test_rows.csv
val_predictions.csv
best.pt
last.pt
```

W&B logs train/validation metrics, AUROC, best-checkpoint ROC and PR curves,
confusion matrix, positive-score histogram, and uploads the main run artifacts.

## SPR Test Submission

If full processed SPR folders already exist:

```bash
python scripts/create_processed_spr_test_subset.py \
  --test-index processed_spr/test_set_index.csv \
  --processed-root processed_spr \
  --output-root processed_spr/test_subset \
  --mode symlink \
  --require-all-accessions
```

Run inference on the subset:

```bash
python scripts/infer_processed_spr_submission.py \
  --sample-submission processed_spr/test_set_index.csv \
  --processed-root processed_spr/test_subset
```

If you want to delete the full processed folders, recreate the subset with
`--mode copy` first. Symlink subsets depend on the original folders.

If processed images do not exist yet, preprocess only test DICOM accessions:

```bash
python scripts/prepare_spr_test_processed.py \
  --test-index processed_spr/test_set_index.csv \
  --spr-dicom-dir /path/to/SPR_Mammo_Recall \
  --output-root processed_spr/test_processed \
  --workers 4
```
