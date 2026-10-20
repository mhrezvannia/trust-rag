from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/main.py` puts only the script's own folder (app/) on sys.path,
# so the `app` package itself would not be importable. Add the repository root.
_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from app.config import get_settings  # noqa: E402
from app.ui import render_app  # noqa: E402
from app.utils import configure_logging  # noqa: E402


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    render_app()


if __name__ == "__main__":
    main()
