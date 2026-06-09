# Cloud Data Manifest

Use R2 as object storage for portable artifacts, then copy the needed files to the
Vast.ai instance local disk before training.

## Upload to R2 for grayscale training

```text
processed_datasets/rsna/grayscale.zip
processed_datasets/spr/grayscale.zip
processed_datasets/vindr/grayscale.zip
processed_datasets/combined_mammo_recall_labels_birads_only.csv
```

Optional, for debugging only:

```text
processed_datasets/spr/preprocessing_failures.csv
preview/previews/
```

Do not upload RGB zips for the first grayscale experiments unless you plan to
train RGB:

```text
processed_datasets/*/rgb_multiwindow.zip
```

## Keep in Git

```text
scripts/
src/
config/
docs/
notebooks/
tests/
environment.yml
.gitignore
```

## Keep local / do not commit

```text
processed_datasets/
runs/
wandb/
preview/
outputs/
processed_datasets/combined_mammo_recall_labels_birads_only.csv
```

The labels CSV is needed for training, but it contains dataset-derived local
metadata such as patient IDs and original local paths, so keep it as a data
artifact rather than source code.
