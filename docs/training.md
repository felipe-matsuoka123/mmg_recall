# Local training

`scripts/train_classifier.py` trains from one preprocessed dataset zip. The zip must
contain `metadata.csv` and the PNG paths listed in `processed_path`.

The preprocessing metadata leaves `label` blank. For training, either fill that
column before zipping or pass a labels CSV with at least:

```csv
image_id,label
1.2.3.4,0
1.2.3.5,1
```

The loader joins the labels CSV to metadata with `image_id` by default and uses
`patient_id` to create a patient-grouped local train/validation split.

## Baseline runs

Grayscale:

```bash
python scripts/train_classifier.py --config config/local_baseline.yaml
```

RGB replicated:

```bash
python scripts/train_classifier.py --config config/local_rgb_replicated.yaml
```

RGB multi-window:

```bash
python scripts/train_classifier.py --config config/local_rgb_multiwindow.yaml
```

Any config value can be overridden from the CLI. For a small local GPU smoke test:

```bash
python scripts/train_classifier.py \
  --config config/local_rgb_multiwindow.yaml \
  --epochs 1 \
  --batch-size 2 \
  --image-size 256 \
  --num-workers 0
```

To log a run to W&B:

```bash
python scripts/train_classifier.py \
  --config config/local_baseline.yaml \
  --wandb \
  --wandb-project mammorecall \
  --wandb-run-name grayscale-smoke
```

The script writes `config.json`, `label_map.json`, `best.pt`, and `last.pt` under
the configured `run_dir`.
