#!/usr/bin/env python
"""Run the reproducible dataset pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fleet.data.generate import run  # noqa: E402

if __name__ == "__main__":
    run()
