# Training

`scripts/train_classifier.py` trains from a checked-in experiment config. The
config owns the dataset paths, metadata/label joins, split policy, model,
optimizer, and logging defaults. The CLI is intentionally limited to run-local
overrides such as `--run-dir`, `--max-samples`, `--device`, `--wandb`, and
`--wandb-run-name`.

For day-to-day work, use the experiment runner:

```bash
python scripts/experiment.py list
python scripts/experiment.py preflight spr_grayscale_simple_cnn
python scripts/experiment.py train spr_grayscale_simple_cnn
```

The runner accepts a unique config stem such as `spr_grayscale_simple_cnn`, a
grouped alias such as `cloud/all_grayscale_convnext_tiny_1024`, or a direct YAML
path. Use `--dry-run` to print the commands without executing them.

The preprocessing metadata leaves labels blank. For training, pass a labels CSV
with at least:

```csv
image_id,target
1.2.3.4,0
1.2.3.5,1
```

The current combined manifest uses `target` for binary recall labels and
`dataset` for source filtering. The loader joins labels to metadata with
`image_id` by default. Rows without a matching label are skipped by the training
script; pass `--require-all-labels` if you want the run to fail instead.

## Label manifest paths

Dataset locations are machine-specific, so keep them in a local ignored paths
file instead of editing `scripts/create_recall_label_manifest.py`:

```bash
cp config/env/paths_data_preprocessing.example.yaml config/env/paths_data_preprocessing.yaml
$EDITOR config/env/paths_data_preprocessing.yaml

python scripts/create_recall_label_manifest.py \
  --paths-config config/env/paths_data_preprocessing.yaml \
  --rsna-label-mode birads_only
```

`config/env/paths_data_preprocessing.yaml` is ignored by git. On a VM, edit only that file or pass
a different paths file:

```bash
python scripts/create_recall_label_manifest.py --paths-config /mnt/data/paths.yaml
```

Individual path flags, such as `--spr-dicom-dir`, still override the YAML value
for one-off runs.

## Fixed experiment split

Create a durable train/validation/test split once and reuse it for every
experiment:

```bash
python scripts/create_split_manifest.py \
  processed_datasets/combined_mammo_recall_labels_birads_only.csv \
  processed_datasets/combined_mammo_recall_splits_spr_holdout.csv
```

The default split is patient/study grouped, stratified by dataset and binary
target at the group level, and reserves only SPR groups for the `test` split.
This keeps the local holdout aligned with the Kaggle target dataset while still
using RSNA, SPR, and VinDr for training and validation. The generated manifest
adds:

```csv
split_group,experiment_split
spr:patient-1,train
spr:patient-2,val
spr:patient-3,test
```

Training configs use `split_csv` and `split_col: experiment_split`. When a split
CSV is present, `scripts/train_classifier.py` trains only on `train` rows and
validates only on `val` rows. The `test` rows are held out for a separate final
evaluation step. Each run writes `train_rows.csv`, `val_rows.csv`, and, when
present, `test_rows.csv` under the run directory. If `split_csv` is omitted, the
script falls back to the older seeded patient-grouped train/validation split.

## Dataset previews

Before creating the full dataset zips, write three preview PNGs for each output
dataset:

```bash
python scripts/create_mammo_datasets.py /path/to/dicoms data/processed --preview-only
```

Preview files are written under `data/processed/previews/`.
Add `--preview-diagnostics` to also write the full image window, binary mask,
crop overlay, and cropped window under `data/processed/previews/diagnostics/`.

For full preprocessing, use multiple workers to parallelize DICOM decode and
image processing:

```bash
python scripts/create_mammo_datasets.py /path/to/dicoms data/processed --workers 4
```

## Baseline runs

SPR grayscale smoke run with the simple CNN:

```bash
python scripts/experiment.py train spr_grayscale_simple_cnn
```

Enable W&B for the same run:

```bash
python scripts/experiment.py train spr_grayscale_simple_cnn \
  --wandb \
  --wandb-run-name spr-grayscale-smoke
```

For a small local smoke test, use one of the smoke configs or create a short
config under `config/local_exp/`. You can still cap the row count or redirect outputs
from the CLI:

```bash
python scripts/experiment.py train smoke_all_grayscale_convnext_tiny \
  --max-samples 64 \
  --run-dir runs/smoke_debug
```

## Weighted loss experiment

Set `loss_weighting: balanced` in a training config to use class-weighted cross
entropy. The training script computes weights from the training split only:

```text
class_weight = train_sample_count / (num_classes * train_class_count)
```

Leave `loss_weighting: none` for the unweighted baseline. The resolved
`loss_class_weights` are written to `config.json`, printed at startup, and logged
to W&B as part of the run config.

ConvNeXt Tiny grayscale run:

```bash
python scripts/experiment.py train spr_grayscale_convnext_tiny
```

All available grayscale sources with the simple CNN at 512:

```bash
python scripts/experiment.py train all_grayscale_simple_cnn
```

Recommended first cloud experiment, using all available grayscale sources at
1024 with pretrained ConvNeXt Tiny:

```bash
python scripts/experiment.py train all_grayscale_convnext_tiny_1024 --preflight --check-wandb
```

RGB multi-window experiment, using all available RGB sources at 1024 with
pretrained ConvNeXt Tiny:

```bash
python scripts/experiment.py train all_rgb_multiwindow_convnext_tiny_1024 --preflight --check-wandb
```

With the current combined label manifest, the all-source grayscale config finds
labeled rows for RSNA, SPR, and VinDr. RSNA uses `processed_path_stem` as the
join key because its processed metadata image IDs differ from the manifest IDs.

Cloud smoke test before the full run:

```bash
export TORCH_HOME=/mnt/outputs/torch-cache

python scripts/experiment.py train smoke_all_grayscale_convnext_tiny \
  --run-dir /mnt/outputs/cloud_smoke_all_grayscale_convnext_tiny
```

RGB cloud smoke test before the full RGB run:

```bash
export TORCH_HOME=/mnt/outputs/torch-cache

python scripts/experiment.py train smoke_all_rgb_multiwindow_convnext_tiny \
  --run-dir /mnt/outputs/cloud_smoke_all_rgb_multiwindow_convnext_tiny
```

## SPR test inference subset

If the full SPR processed folders already exist, create a lightweight subset for
only the accessions listed in `test_set_index.csv`:

```bash
python scripts/create_processed_spr_test_subset.py \
  --test-index processed_spr/test_set_index.csv \
  --processed-root processed_spr \
  --output-root processed_spr/test_subset \
  --mode symlink \
  --require-all-accessions
```

The output layout is compatible with `scripts/infer_processed_spr_submission.py`:

```text
processed_spr/test_subset/grayscale/<AccessionNumber>/*.png
processed_spr/test_subset/rgb_multiwindow/<AccessionNumber>/*.png
```

Then run inference without scanning the full processed SPR folder:

```bash
python scripts/infer_processed_spr_submission.py \
  --sample-submission processed_spr/test_set_index.csv \
  --processed-root processed_spr/test_subset
```

If the processed folders do not exist yet, preprocess only the test accessions
from DICOMs into a small folder:

```bash
python scripts/prepare_spr_test_processed.py \
  --test-index processed_spr/test_set_index.csv \
  --spr-dicom-dir /path/to/SPR_Mammo_Recall \
  --output-root processed_spr/test_processed \
  --workers 4
```

For cloud training, keep the config in git and use the VM helper scripts:

```bash
scripts/setup_vm.sh
export TORCH_HOME=/mnt/outputs/torch-cache

scripts/download_data.sh grayscale
scripts/run_smoke.sh grayscale
scripts/run_train.sh grayscale
```

RGB run:

```bash
scripts/download_data.sh rgb_multiwindow
scripts/run_smoke.sh rgb_multiwindow
scripts/run_train.sh rgb_multiwindow
```

`run_train.sh` uploads the finished run directory back to `${REMOTE}/runs/` after
successful training. Disable that with `UPLOAD_RESULTS=0` if you only want local
checkpoints.

The helpers use these defaults, which can be overridden with environment
variables:

```bash
REMOTE=s2:mammo-recall-data
DATA_ROOT=/mnt/data/processed_datasets
OUTPUT_ROOT=/mnt/outputs
UPLOAD_RESULTS=1
```

Equivalent manual grayscale commands:

```bash
python scripts/download_data.py \
  --remote s2:mammo-recall-data \
  --variant grayscale \
  --execute

ln -s /mnt/data/processed_datasets processed_datasets
export TORCH_HOME=/mnt/outputs/torch-cache
python scripts/preflight.py \
  --config config/cloud_exp/all_grayscale_convnext_tiny_1024.yaml \
  --check-wandb \
  --run-dir /mnt/outputs/all_grayscale_convnext_tiny_1024

python scripts/train_classifier.py \
  --config config/cloud_exp/all_grayscale_convnext_tiny_1024.yaml \
  --run-dir /mnt/outputs/all_grayscale_convnext_tiny_1024
```

Equivalent manual RGB commands:

```bash
python scripts/download_data.py \
  --remote s2:mammo-recall-data \
  --variant rgb_multiwindow \
  --execute

python scripts/preflight.py \
  --config config/cloud_exp/all_rgb_multiwindow_convnext_tiny_1024.yaml \
  --check-wandb \
  --run-dir /mnt/outputs/all_rgb_multiwindow_convnext_tiny_1024

python scripts/train_classifier.py \
  --config config/cloud_exp/all_rgb_multiwindow_convnext_tiny_1024.yaml \
  --run-dir /mnt/outputs/all_rgb_multiwindow_convnext_tiny_1024

scripts/upload_results.sh rgb_multiwindow
```

The script writes `config.json`, `label_map.json`, `metrics.jsonl`, `best.pt`,
`last.pt`, `train_rows.csv`, `val_rows.csv`, and, when present, `test_rows.csv`
under the configured `run_dir`. `metrics.jsonl` contains one JSON record per
epoch. `val_predictions.csv` is overwritten whenever a new `best.pt` checkpoint
is saved, so the predictions match the selected validation checkpoint.

W&B logs running training loss/accuracy every 50 batches by default, plus
epoch-level train/validation loss, accuracy, and binary AUROC when both classes
are present in an epoch partition. When a new best validation checkpoint is
saved, W&B also logs validation ROC, precision-recall, confusion-matrix, and
positive-score histogram charts under stable keys so runs can be compared in the
same W&B workspace panel. The same best checkpoint step logs precision, recall,
specificity, F1, balanced accuracy, average precision, and confusion-matrix
counts.

Change the live training logging frequency with `wandb_log_interval` in the
experiment config; set it to `0` to disable batch-level logging. When W&B is
enabled, the script also uploads the main run artifacts as a model artifact at
the end of the run.
