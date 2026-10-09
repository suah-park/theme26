"""Evaluate saved binary pairs or the output of scripts/infer.py; no training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from theme26.evaluation.physics_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
