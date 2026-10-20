"""Cross-platform launcher for the TrustRAG Streamlit app.

Usage: python run_app.py [extra streamlit arguments]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def main() -> int:
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT_DIR / "app" / "main.py"), *sys.argv[1:]]
    # Run from the repository root so .streamlit/config.toml and .env are picked up.
    return subprocess.call(command, cwd=ROOT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
