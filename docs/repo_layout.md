# Repository Layout

Keep source code and reproducible configuration in git. Keep datasets, generated
images, checkpoints, and tracker state outside git.

## Source-controlled

```text
config/
  baselines/  # reference and comparison runs
  cloud/      # cloud smoke tests and primary cloud experiments
  local/      # local toy/example configs
docs/
notebooks/
scripts/
src/
tests/
environment.yml
```

## Local artifacts

```text
processed_datasets/  # preprocessed zips and training label manifests
runs/                # local checkpoints and run metadata
wandb/               # local W&B cache/state
preview/             # preprocessing preview images
outputs/             # scratch outputs
```

For cloud training, copy or mount `processed_datasets/` into the repo root so
the checked-in configs work without path edits.
