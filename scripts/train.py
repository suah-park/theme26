"""Command-line entry point for SAINT training.

The model/integration owner will connect this skeleton after the config schema
and trainer are implemented. Keep this path stable because Slurm calls it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a theme26 SAINT experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to a versioned experiment config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(f"model owner: connect trainer for config {args.config}")


if __name__ == "__main__":
    main()
