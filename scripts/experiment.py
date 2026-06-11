#!/usr/bin/env python3
"""Run common experiment workflows from named config files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "config"
CONFIG_DIRS = ("cloud_exp", "local_exp")


def config_paths() -> list[Path]:
    paths = []
    for directory in CONFIG_DIRS:
        paths.extend((CONFIG_ROOT / directory).glob("*.yaml"))
    return sorted(paths)


def config_aliases(path: Path) -> set[str]:
    relative = path.relative_to(CONFIG_ROOT).with_suffix("")
    return {path.stem, relative.as_posix(), f"config/{relative.as_posix()}"}


def resolve_config(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    matches = [path for path in config_paths() if name in config_aliases(path)]
    if not matches:
        raise SystemExit(f"Unknown experiment {name!r}. Run `python scripts/experiment.py list`.")
    if len(matches) > 1:
        options = ", ".join(path.relative_to(REPO_ROOT).as_posix() for path in matches)
        raise SystemExit(f"Ambiguous experiment {name!r}. Matches: {options}")
    return matches[0]


def run_command(command: list[str], *, dry_run: bool = False) -> int:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return 0
    print(printable)
    return subprocess.run(command, check=False).returncode


def add_run_overrides(command: list[str], args: argparse.Namespace) -> list[str]:
    if args.run_dir:
        command.extend(["--run-dir", str(args.run_dir)])
    if getattr(args, "max_samples", None) is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if getattr(args, "device", None):
        command.extend(["--device", args.device])
    if getattr(args, "wandb", None) is not None:
        command.append("--wandb" if args.wandb else "--no-wandb")
    if getattr(args, "wandb_run_name", None):
        command.extend(["--wandb-run-name", args.wandb_run_name])
    return command


def preflight_command(config: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "preflight.py"),
        "--config",
        str(config),
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    if args.run_dir:
        command.extend(["--run-dir", str(args.run_dir)])
    if args.check_wandb:
        command.append("--check-wandb")
    return command


def train_command(config: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_classifier.py"),
        "--config",
        str(config),
    ]
    return add_run_overrides(command, args)


def command_list(_args: argparse.Namespace) -> int:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in config_paths():
        grouped[path.parent.name].append(path)
    for group in CONFIG_DIRS:
        paths = grouped.get(group, [])
        if not paths:
            continue
        print(f"{group}:")
        for path in paths:
            print(f"  {path.stem:48s} {path.relative_to(REPO_ROOT)}")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    config = resolve_config(args.experiment)
    return run_command(preflight_command(config, args), dry_run=args.dry_run)


def command_train(args: argparse.Namespace) -> int:
    config = resolve_config(args.experiment)
    if args.preflight:
        status = run_command(preflight_command(config, args), dry_run=args.dry_run)
        if status != 0:
            return status
    return run_command(train_command(config, args), dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available experiment configs.")
    list_parser.set_defaults(func=command_list)

    preflight_parser = subparsers.add_parser("preflight", help="Check data/env readiness.")
    preflight_parser.add_argument("experiment", help="Config path, config/<group>/<name>, or unique stem.")
    preflight_parser.add_argument("--run-dir", type=Path, default=None)
    preflight_parser.add_argument("--check-wandb", action="store_true")
    preflight_parser.add_argument("--min-free-gb", type=float, default=20.0)
    preflight_parser.add_argument("--dry-run", action="store_true")
    preflight_parser.set_defaults(func=command_preflight)

    train_parser = subparsers.add_parser("train", help="Run an experiment config.")
    train_parser.add_argument("experiment", help="Config path, config/<group>/<name>, or unique stem.")
    train_parser.add_argument("--run-dir", type=Path, default=None)
    train_parser.add_argument("--max-samples", type=int, default=None)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    train_parser.add_argument("--wandb-run-name", default=None)
    train_parser.add_argument("--preflight", action="store_true")
    train_parser.add_argument("--check-wandb", action="store_true")
    train_parser.add_argument("--min-free-gb", type=float, default=20.0)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.set_defaults(func=command_train)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
