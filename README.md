# Multi-Agent Research Assistant

A small **agentic / multi-agent** pipeline for research papers: three specialist agents (paper summarizer, citation extractor, research gap finder) run in parallel over the same text, orchestrated from one entry point.

The code lives under the `project3/` directory in this repository; the product name is **Multi-Agent Research Assistant**.

## Features

- **Paper summarizer** — overview, contributions, method, results, author-stated limitations  
- **Citation extractor** — structured citation-like entries (JSON → table) with fallback display  
- **Research gap finder** — limitations, future work, conservative inferred gaps  

## Requirements

- Python 3.10+ recommended  
- A **Groq** API key and/or a **Google Gemini** API key  

### LLM configuration

Environment variables are read from **`project3/.env`** first, then from the process environment.

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | API key from [GroqCloud](https://console.groq.com/) |
| `GROQ_MODEL` | Optional. Default: `llama-3.3-70b-versatile` |
| `GEMINI_API_KEY` | Optional; from [Google AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | Optional. Default: `gemini-flash-latest` |
| `LLM_PROVIDER` | `groq` or `gemini`. If unset, **Groq is used when `GROQ_API_KEY` is set**, otherwise Gemini when `GEMINI_API_KEY` is set |

Example **`project3/.env`** (Groq only):

```bash
GROQ_API_KEY=gsk_your_key_here
# optional:
# GROQ_MODEL=llama-3.3-70b-versatile
# LLM_PROVIDER=groq
```

Example using **Gemini** instead:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

Never commit `.env` or API keys to git.

## Install

From the repository root:

```bash
cd project3
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (Streamlit)

```bash
cd project3
source .venv/bin/activate   # if you use a venv
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`). Upload a PDF or paste text, then click **Run all agents**.

## Run (library)

With `project3` as the current working directory so `src` imports resolve:

```bash
cd project3
python3 -c "from src.orchestrator import run_research_assistant; print(run_research_assistant('Your paper text here...').to_dict())"
```

## Project layout

```
project3/
  .env                 # your keys (create locally; not in git)
  README.md            # this file
  requirements.txt
  app/
    streamlit_app.py   # UI
  src/
    llm.py             # Groq / Gemini routing
    orchestrator.py    # parallel agent runs
    pdf_processor.py
    agents/
      summarizer.py
      citations.py
      gaps.py
```

## Troubleshooting

- **“GROQ_API_KEY is not set”** — Create `project3/.env` with `GROQ_API_KEY=...` or export the variable before running.  
- **Rate limits** — Wait and retry; consider `LLM_PROVIDER=gemini` with a valid `GEMINI_API_KEY` as a fallback.  
- **Import errors** — Run Streamlit or Python with **`cd project3`** so `src` is on the path as in this README.
