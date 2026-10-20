#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root so .streamlit/config.toml and .env are picked up.
cd "$(dirname "$0")"
streamlit run app/main.py "$@"
