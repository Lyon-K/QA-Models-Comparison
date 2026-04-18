# QA Models Comparison

This repository compares the final QA model lineup for public-health prompts:

- `T5`
- `noRAG`
- `RAG`
- `graphRAG`
- `hybridRAG`

The demo currently uses:

- a lightweight backend wrapper in [backend/service.py](C:/Users/76161/Documents/New%20project/repo/backend/service.py)
- a Streamlit frontend in [front_end/app.py](C:/Users/76161/Documents/New%20project/repo/front_end/app.py)

## Final Model Lineup

1. `T5`
   File: [models/seq2seq/T5.py](C:/Users/76161/Documents/New%20project/repo/models/seq2seq/T5.py)
   Role: generative baseline model

2. `noRAG`
   File: [models/noRag/noRag.py](C:/Users/76161/Documents/New%20project/repo/models/noRag/noRag.py)
   Role: direct LLM baseline without retrieval

3. `RAG`
   File: [models/RAG/RAG.py](C:/Users/76161/Documents/New%20project/repo/models/RAG/RAG.py)
   Role: retrieval-augmented generation

4. `graphRAG`
   File: [models/graphRAG/graphRAG.py](C:/Users/76161/Documents/New%20project/repo/models/graphRAG/graphRAG.py)
   Role: graph-based retrieval model

5. `hybridRAG`
   Role: hybrid retrieval-augmented model
   Note: there is no visible local `hybridRAG` implementation file in the current repository snapshot, so the backend returns a clean fallback message for this model.

## Architecture

The current structure is:

- [front_end/app.py](C:/Users/76161/Documents/New%20project/repo/front_end/app.py)
  Streamlit UI that renders one card per model.
- [backend/service.py](C:/Users/76161/Documents/New%20project/repo/backend/service.py)
  Backend wrapper that calls all five model slots and isolates failures so the demo does not crash.
- [models/seq2seq/T5.py](C:/Users/76161/Documents/New%20project/repo/models/seq2seq/T5.py)
  T5 baseline with a demo-safe fallback to pretrained `t5-small`.
- [models/noRag/noRag.py](C:/Users/76161/Documents/New%20project/repo/models/noRag/noRag.py)
  No-retrieval LLM baseline.
- [models/RAG/RAG.py](C:/Users/76161/Documents/New%20project/repo/models/RAG/RAG.py)
  Vector-store RAG model.
- [models/graphRAG/graphRAG.py](C:/Users/76161/Documents/New%20project/repo/models/graphRAG/graphRAG.py)
  Neo4j-backed graph retrieval model.

## Quick Start

Install the demo dependencies:

```bash
pip install -r requirements.txt
```

Run the frontend:

```bash
streamlit run front_end/app.py
```

## Expected Demo Behavior

- `T5` should produce a real explanation in the minimal demo path.
- `noRAG`, `RAG`, and `graphRAG` produce real outputs only if their optional dependencies, credentials, and services are installed and configured.
- `hybridRAG` currently falls back because no local implementation is present in this repo snapshot.
- Any unavailable optional model shows a clean message in the UI instead of crashing the app.

## Backend Output Shape

The backend returns:

```python
{
  "T5": "...",
  "noRAG": "...",
  "RAG": "...",
  "graphRAG": "...",
  "hybridRAG": "...",
}
```

## Optional Dependencies For Non-Demo Models

If you want to try the non-demo models too, install:

```bash
pip install python-dotenv ollama neo4j langchain_huggingface sentence-transformers langchain_community datasets pandas faiss-cpu feedparser langchain-text-splitters langchain-chroma langchain-core pypdf ipython
```

If you plan to run `RAG`, `graphRAG`, or `noRAG`, you can also copy `.env.example` to `.env` and fill in the required values.

For cloud `RAG` generation with `ministral-3:8b-cloud`, create a repo-root `.env` file with:

```env
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_MODEL=ministral-3:8b-cloud
OLLAMA_HOST=https://ollama.com
```

`OLLAMA_HOST` is optional and defaults to `https://ollama.com`, but it is helpful to keep it explicit for the cloud path.

## Local Test Commands

Test the backend wrapper:

```bash
python -c "from backend.service import get_fact_check_result; print(get_fact_check_result('Vaccines cause infertility'))"
```

Run the health check:

```bash
python scripts/health_check.py
```

Test `RAG` directly:

```bash
python scripts/test_rag.py
```

Test `noRAG` directly:

```bash
python scripts/test_norag.py
```

Test `hybridRAG` directly:

```bash
python scripts/test_hybrid_rag.py
```

Run the Streamlit UI:

```bash
streamlit run front_end/app.py
```

## Notes

- `T5` replaces the old `seq2seq` label everywhere in the current demo.
- The three retrieval variants in the final lineup are `RAG`, `graphRAG`, and `hybridRAG`.
- `noRAG` remains a separate baseline model.
- The backend keeps failures isolated so one unavailable model does not break the whole response.
