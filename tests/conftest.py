"""Make the repo root importable (hebb/ is not an installed package)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
