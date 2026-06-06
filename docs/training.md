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
python scripts/train_classifier.py --config config/spr_grayscale_simple_cnn.yaml
```

Enable W&B for the same run:

```bash
python scripts/train_classifier.py \
  --config config/spr_grayscale_simple_cnn.yaml \
  --wandb \
  --wandb-run-name spr-grayscale-smoke
```

Any config value can be overridden from the CLI. For a small local GPU smoke test:

```bash
python scripts/train_classifier.py \
  --config config/spr_grayscale_simple_cnn.yaml \
  --epochs 1 \
  --batch-size 2 \
  --image-size 256 \
  --max-samples 64 \
  --num-workers 0
```

ConvNeXt Tiny grayscale run:

```bash
python scripts/train_classifier.py --config config/spr_grayscale_convnext_tiny.yaml
```

All available grayscale sources with the simple CNN at 512:

```bash
python scripts/train_classifier.py --config config/all_grayscale_simple_cnn.yaml
```

The current combined label manifest does not match the processed RSNA image IDs,
so RSNA contributes no labeled rows until that ID mapping is fixed. SPR and VinDr
rows with matching labels are included.

For cloud training, keep the config in git and override only the paths that vary
by machine:

```bash
python scripts/train_classifier.py \
  --config config/spr_grayscale_convnext_tiny.yaml \
  --data-root /mnt/data/processed_datasets/spr/grayscale \
  --data-zip "" \
  --metadata-csv /mnt/data/processed_datasets/spr/grayscale_metadata.csv \
  --labels-csv /mnt/data/combined_mammo_recall_labels_birads_only.csv \
  --run-dir /mnt/outputs/spr_grayscale_convnext_tiny
```

The script writes `config.json`, `label_map.json`, `best.pt`, and `last.pt` under
the configured `run_dir`. W&B logs loss, accuracy, and binary AUROC when both
classes are present in an epoch partition.
