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
  download_data.py  # rclone wrapper for grayscale/RGB artifacts
  preflight.py      # config, data, disk, CUDA, and W&B readiness checks
  setup_vm.sh       # create VM data/output folders and repo data symlink
  download_data.sh  # shell wrapper around download_data.py
  run_smoke.sh      # preflight and run the selected smoke config
  run_train.sh      # preflight and run the selected full config
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
