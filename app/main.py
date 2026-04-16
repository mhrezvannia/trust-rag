from __future__ import annotations

from app.config import get_settings
from app.ui import render_app
from app.utils import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    render_app()


if __name__ == "__main__":
    main()
