# FSSAI-Compliance Copilot

Streamlit app that audits a facility SOP/manual against a local FSSAI/HACCP knowledge base using a **strict retrieval-first** workflow.

## Syllabus alignment (Unit III–V)

- **Unit III (FSSAI Licensing/Compliance):** Regulations corpus + clause retrieval via the local vector index (`scripts/build_vectorstore.py`, `fssai_copilot/vectorstore.py`).
- **Unit IV (GMP/GHP, Schedule 4):** Operational claims extraction + gap detection categories like Hygiene, Personnel/Training, Pest Control (`fssai_copilot/orchestrator.py`).
- **Unit V (Bioethics/Food Safety Ethics):** Report generation emphasizes transparency by linking gaps to retrieved sources and avoiding unsupported claims (`fssai_copilot/reporting.py`).

## Quickstart (Windows)

1) Create & activate a venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) (One-time) Build the local regulations index

- Put official FSSAI/Schedule 4/HACCP PDFs or TXT files under `data/regulations/`.
- Then run:

```powershell
python .\scripts\build_vectorstore.py
```

This creates a persistent ChromaDB index in `chroma/`.

4) Run the app

```powershell
streamlit run app.py
```

## Notes

- The app is designed to **ground outputs to retrieved clauses**. If there is no index (or retrieval returns nothing), it will avoid making strong claims.
- LLM mode is optional. If you set `OPENAI_API_KEY`, the auditor/analyst steps run in a stricter RAG mode (JSON-only outputs constrained to retrieved clauses).

## Project structure

- `app.py` – Streamlit UI
- `fssai_copilot/` – ingestion, vectorstore, orchestrator, PDF reporting
- `scripts/build_vectorstore.py` – index builder for regulations
