# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-10-20

First versioned release. Features and UI are unchanged from the initial MVP; this release focuses on correctness, tests and reproducibility.

### Added
- Offline test suite (62 tests): chunking, vector search, re-ranking, prompt and citation construction, confidence and abstention, provider error handling, PDF ingestion, the evaluation loop and a headless Streamlit smoke test. A deterministic hashing embedder replaces the sentence-transformers model, and LLM calls are stubbed.
- GitHub Actions CI: ruff and pytest on Python 3.10, 3.11 and 3.12 with CPU-only PyTorch.
- `pyproject.toml` with project metadata (version 1.0.0) and ruff/pytest configuration.
- `requirements-dev.txt` (pytest, ruff).
- `run_app.py`, a cross-platform launcher.
- `CITATION.cff` and this changelog.

### Changed
- `requirements.txt` now uses version ranges (tested lower bound, below the next major version) instead of exact pins. Removed unused `tiktoken` and `scikit-learn`, and moved `pytest` to the development requirements.
- `run_app.sh` now runs from the repository root and forwards extra arguments to Streamlit.
- `sentence-transformers` is imported lazily, so modules that use the embedding service load without PyTorch until a model is needed.
- README rewritten: method, exact confidence and abstention rules, provider configuration, evaluation and limitations.
- `.gitignore` extended to cover virtual environments, tool caches, build outputs, `.env.*` files and FAISS index files.

### Fixed
- `streamlit run app/main.py` failed with `ModuleNotFoundError: No module named 'app'` because Streamlit puts only the script's folder on `sys.path`.
- Paragraph chunks lost their configured overlap: the overlap words kept after a chunk was flushed were immediately discarded.
- The number pattern used for numeric grounding never matched a trailing `%`, so percentages were compared as bare numbers.
- Malformed model output raised `ValueError`, and OpenAI SDK errors and socket timeouts were not caught, so the UI crashed instead of showing a provider error. These failures are now reported as `LLMProviderError`.
- The evaluation run now fails fast with a clear error, before the embedding model is loaded, when the PDFs contain no extractable text.
- The evaluation notebook can now import the `app` package and find `data/uploads/` when Jupyter is started from `notebooks/`.

### Security
- `.env.example` no longer contains a pre-filled API key value.
- Uploaded file names are reduced to their base name before being written to `data/uploads/`.

[Unreleased]: https://github.com/mhrezvannia/trust-rag/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mhrezvannia/trust-rag/releases/tag/v1.0.0
