"""Compatibility entry point for the E1 data QC CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theme26.data.cli import main


if __name__ == "__main__":
    main(["e1-validation-crop", *sys.argv[1:]])
