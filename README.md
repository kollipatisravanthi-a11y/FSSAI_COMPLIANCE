# FSSAI-Compliance Copilot

A Streamlit-based compliance auditor for food safety facilities. The app ingests facility SOPs and manuals, matches claims against a local FSSAI/HACCP regulations corpus, and generates audit reports with supporting clause references.

## Features

- Upload facility SOPs, manuals, or regulations documents
- Extract operational claims and classify gaps by hygiene, training, pest control, labeling, and compliance
- Retrieve relevant FSSAI clauses from a local vector store
- Display audit scores, gap summaries, and source-backed insights
- Download audit reports as PDF
- Optional LLM-backed analyst mode when `OPENAI_API_KEY` is provided

## Quickstart (Windows)

1) Clone the repo and enter the project folder

```powershell
git clone https://github.com/kollipatisravanthi-a11y/FSSAI_COMPLIANCE.git
cd FSSAI_COMPLIANCE
```

2) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install Python dependencies

```powershell
pip install -r requirements.txt
```

4) Build the local regulations index

- Add FSSAI/HACCP PDF or TXT files under `data/regulations/`
- Run:

```powershell
python .\scripts\build_vectorstore.py
```

This creates a persistent ChromaDB index in `chroma/`.

5) Run the Streamlit app

```powershell
streamlit run app.py
```

## Optional configuration

- Set `OPENAI_API_KEY` in your environment to enable LLM-backed analyst steps
- Use `python-dotenv` with a `.env` file if preferred

## Requirements

- Python 3.10 or newer
- `requirements.txt` lists the runtime dependencies used by the app

## Project structure

- `app.py` – Streamlit application entrypoint
- `fssai_copilot/` – backend modules for ingestion, search, orchestration, and reporting
- `scripts/build_vectorstore.py` – build the local ChromaDB regulations index
- `data/regulations/` – source regulation files and sample documents
- `chroma/` – persistent ChromaDB storage (generated after index build)
