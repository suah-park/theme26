"""Command-line entry point for frozen SAINT inference."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen theme26 SAINT inference")
    parser.add_argument("--config", type=Path, required=True, help="Path to a versioned inference config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        f"model owner: connect inference for config {args.config} and checkpoint {args.checkpoint}"
    )


if __name__ == "__main__":
    main()
