# Training

`scripts/train_classifier.py` trains from one preprocessed PNG dataset. Pass
`--data-zip` for a zip that contains `metadata.csv`, or `--data-root` for an
unzipped image directory. If metadata is not inside the image directory, pass
`--metadata-csv`.

The preprocessing metadata leaves labels blank. For training, pass a labels CSV
with at least:

```csv
image_id,target
1.2.3.4,0
1.2.3.5,1
```

The current combined manifest uses `target` for binary recall labels and
`dataset` for source filtering. The loader joins labels to metadata with
`image_id` by default and uses `patient_id` to create a patient-grouped
train/validation split. Rows without a matching label are skipped by the training
script; pass `--require-all-labels` if you want the run to fail instead.

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
python scripts/train_classifier.py --config config/baselines/spr_grayscale_simple_cnn.yaml
```

Enable W&B for the same run:

```bash
python scripts/train_classifier.py \
  --config config/baselines/spr_grayscale_simple_cnn.yaml \
  --wandb \
  --wandb-run-name spr-grayscale-smoke
```

Any config value can be overridden from the CLI. For a small local GPU smoke test:

```bash
python scripts/train_classifier.py \
  --config config/baselines/spr_grayscale_simple_cnn.yaml \
  --epochs 1 \
  --batch-size 2 \
  --image-size 256 \
  --max-samples 64 \
  --num-workers 0
```

ConvNeXt Tiny grayscale run:

```bash
python scripts/train_classifier.py --config config/baselines/spr_grayscale_convnext_tiny.yaml
```

All available grayscale sources with the simple CNN at 512:

```bash
python scripts/train_classifier.py --config config/baselines/all_grayscale_simple_cnn.yaml
```

Recommended first cloud experiment, using all available grayscale sources at
1024 with pretrained ConvNeXt Tiny:

```bash
python scripts/train_classifier.py --config config/cloud/all_grayscale_convnext_tiny_1024.yaml
```

With the current combined label manifest, the all-source grayscale config finds
labeled rows for RSNA, SPR, and VinDr. RSNA uses `processed_path_stem` as the
join key because its processed metadata image IDs differ from the manifest IDs.

Cloud smoke test before the full run:

```bash
export TORCH_HOME=/mnt/outputs/torch-cache

python scripts/train_classifier.py \
  --config config/cloud/smoke_all_grayscale_convnext_tiny.yaml \
  --run-dir /mnt/outputs/cloud_smoke_all_grayscale_convnext_tiny
```

For cloud training, keep the config in git and place or symlink the data
artifacts at the paths referenced by the config:

```bash
ln -s /mnt/data/processed_datasets processed_datasets
export TORCH_HOME=/mnt/outputs/torch-cache

python scripts/train_classifier.py \
  --config config/cloud/all_grayscale_convnext_tiny_1024.yaml \
  --run-dir /mnt/outputs/all_grayscale_convnext_tiny_1024
```

The script writes `config.json`, `label_map.json`, `best.pt`, and `last.pt` under
the configured `run_dir`. W&B logs loss, accuracy, and binary AUROC when both
classes are present in an epoch partition. When W&B is enabled, the script also
uploads `best.pt`, `last.pt`, `config.json`, and `label_map.json` as a model
artifact at the end of the run.
