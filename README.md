# TrustRAG

TrustRAG is a production-style MVP for trustworthy retrieval-augmented question answering over academic and technical PDF documents. It emphasizes grounded answers, explicit citations, conservative abstention, and lightweight confidence estimation instead of unchecked fluency.

The project currently supports four answer providers:

- `gemini` for Google Gemini API
- `ollama` for local inference
- `openai` for OpenAI API-backed answers
- `mock` for low-resource demos and retrieval-only testing

The repository is configured to support easy cloud-based testing on constrained hardware, while keeping the provider fully swappable through `.env`.

## Why This Project Matters

Trustworthy AI systems should not trade correctness for fluency. TrustRAG focuses on:

- Grounded generation from retrieved evidence
- Uncertainty-aware QA with explicit confidence signals
- Conservative abstention to reduce hallucinations
- Practical document QA for technical and academic workflows

This makes it a strong portfolio project for trustworthy AI, NLP, and LLM systems engineering.

## How to Position This Project

TrustRAG should be presented as a **research-engineering project**, not as a finished research contribution. Its current strength is that it demonstrates the ability to design, build, debug, and evaluate a trustworthy document QA pipeline end to end.

For PhD applications, that is valuable because it shows:

- systems thinking for LLM/NLP pipelines
- explicit concern for grounded generation and hallucination reduction
- practical handling of uncertainty and abstention
- willingness to investigate failure modes rather than only showing successful demos
- the ability to build clean, modular research infrastructure

What it does **not** yet claim:

- calibrated trust estimates
- benchmark-grade evaluation
- publication-level experimental rigor
- a novel algorithmic contribution on its own

This framing is intentional and keeps the project credible as a research-engineering artifact for a CV, GitHub portfolio, or statement of purpose.

## Features

- Multi-PDF upload and parsing with PyMuPDF
- Deterministic paragraph-aware chunking with overlap
- Sentence-transformers embeddings with local caching
- FAISS indexing and top-k retrieval
- Lightweight reranking for improved evidence selection
- Evidence-grounded answering with citations
- Lightweight confidence scoring and abstention logic
- Gemini, Ollama, and OpenAI provider support
- Mock provider for low-resource testing and demo mode
- Streamlit UI with evidence inspection and a developer panel
- Lightweight evaluation module and notebook
- Unit tests for chunking, retrieval, and confidence logic

## Architecture Overview

The application follows a direct, framework-light RAG pipeline:

1. `ingestion.py` extracts text page by page from uploaded PDFs.
2. `chunking.py` builds deterministic paragraph-aware chunks.
3. `embeddings.py` generates normalized embeddings and caches them locally.
4. `vectorstore.py` stores embeddings in FAISS and returns scored retrieval results.
5. `retrieval.py` handles top-k semantic search and lightweight reranking.
6. `prompts.py`, `llm.py`, and `answering.py` generate grounded answers with citations.
7. `confidence.py` estimates trust and triggers abstention when support is weak.
8. `evaluation.py` supports lightweight inspection of retrieval quality, abstention behavior, and groundedness signals.
9. `ui.py` exposes the system in Streamlit.

## Folder Structure

```text
trust-rag/
├── app/
├── data/
├── notebooks/
├── tests/
├── .env.example
├── README.md
├── requirements.txt
└── run_app.sh
```

## Installation

1. Create a Python 3.11 virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Create an environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## Provider Configuration

### Gemini API

Use this when you want cloud-hosted answers from Google Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

This is the easiest setup for testing on a laptop that cannot reliably run a local model. Be aware that free-tier Gemini usage can hit temporary quota or rate limits.

### Local Ollama

Use this when you want local inference without API costs:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_NUM_CTX=1024
OLLAMA_NUM_PREDICT=256
```

If Ollama returns a memory-related `500` error such as `memory layout cannot be allocated`, reduce `OLLAMA_NUM_CTX`, close other memory-heavy applications, or switch to a smaller local model.

### OpenAI API

Use this when you want OpenAI-backed answers:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

### Mock Mode

Use this to verify ingestion, chunking, retrieval, citations, and UI behavior on a low-resource machine:

```env
LLM_PROVIDER=mock
```

Mock mode does not call a real LLM. It is intended for debugging, retrieval validation, and product demos when local or online generation is unavailable.

## Environment Variables

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_NUM_CTX=1024
OLLAMA_NUM_PREDICT=256
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOG_LEVEL=INFO
```

## How to Run

From the project root:

```bash
streamlit run app/main.py
```

Or use:

```bash
sh run_app.sh
```

On Windows PowerShell:

```powershell
python -m streamlit run app/main.py
```

Note: the repository includes `.streamlit/config.toml` with `fileWatcherType = "none"` to avoid a known Streamlit watcher conflict with `torch`-based dependencies such as `sentence-transformers`.

## Quick Gemini Test

1. Put your Gemini key in `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

2. Run the app:

```powershell
python -m streamlit run app/main.py
```

3. Upload one or more PDFs and ask a question.

If Gemini fails, the app should show the provider error in the UI instead of crashing. If you hit quota limits, switch temporarily to `mock` mode and continue testing the retrieval and trust pipeline.

## Evaluation

The project includes:

- `app/evaluation.py` for lightweight QA logging
- `notebooks/evaluation.ipynb` for manual review
- basic inspection fields for abstention behavior, confidence, citations, and top retrieval score

This workflow is intentionally manual-first. Trustworthy document QA still benefits from inspecting whether answers remain grounded in retrieved evidence.

## Why It Is Credible for PhD Applications

TrustRAG is valuable in an application package because it shows more than “I can call an API.” It shows that you can:

- build a non-trivial NLP/LLM system from scratch
- reason about evidence quality and uncertainty
- debug failure modes in retrieval, grounding, and trust calibration
- translate a broad research concern such as hallucination reduction into a working system artifact

The strongest way to present it is:

- as a research-engineering project on trustworthy QA
- as supporting evidence of research interests in Trustworthy AI, NLP, or LLM systems
- as a platform that can be extended into a more rigorous evaluation study

## Current Limitations

- Confidence is heuristic and not statistically calibrated
- PDF extraction quality depends on source formatting
- The baseline embedding model is compact rather than domain-specialized
- Retrieval is dense-first with lightweight reranking, not a fully benchmarked reranker stack
- The evaluation workflow is manual-first rather than benchmark-driven
- Large local models may not load on memory-constrained laptops
- Cloud providers may impose temporary quota and rate limits

## What Would Make It More Research-Grade

To elevate this project from a strong engineering artifact to a stronger research artifact, the next steps would be:

- build a curated evaluation set of 20 to 50 document QA examples
- report retrieval and answer-grounding metrics systematically
- run ablations on retrieval, reranking, abstention, and confidence scoring
- analyze failure modes in a short technical report or blog post
- compare heuristic confidence with observed grounding quality

These additions would make the project a stronger signal of research maturity.

## Future Work

- Add stronger reranking and hybrid lexical+dense retrieval
- Persist and reload indices across sessions
- Add citation span highlighting and page previews
- Add automatic provider fallback on local model failure or cloud quota exhaustion
- Build benchmark-style evaluation for grounding and abstention
- Explore calibration of confidence against human-judged support quality

## Testing

Run:

```bash
python -m pytest
```
