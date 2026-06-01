from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fssai_copilot.ingestion import extract_text
from fssai_copilot.orchestrator import run_audit
from fssai_copilot.reporting import render_pdf
from fssai_copilot.vectorstore import VectorStoreConfig, index_exists


st.set_page_config(
    page_title="FSSAI-Compliance Copilot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = VectorStoreConfig()
kb_ready = index_exists(cfg)

if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "extracted" not in st.session_state:
    st.session_state.extracted = None
if "uploaded_sig" not in st.session_state:
    st.session_state.uploaded_sig = None
if "report" not in st.session_state:
    st.session_state.report = None


def _severity_rank(sev: str) -> int:
    order = {"Critical": 0, "Major": 1, "Minor": 2}
    return order.get(sev, 99)


with st.sidebar:
    st.header("FSSAI-Compliance Copilot")
    st.caption("Grounded GAP analysis against your local regulations index.")

    st.subheader("Knowledge Base")
    st.write(f"Chroma path: `{cfg.persist_dir}`")
    st.write(f"Collection: `{cfg.collection}`")
    if kb_ready:
        st.success("Regulations index found")
    else:
        st.warning("Regulations index missing")
        st.markdown("Add regulations under `data/regulations/` and run `python scripts/build_vectorstore.py`.")

    st.subheader("Audit Settings")
    st.session_state.top_k = st.slider(
        "Clauses retrieved per claim",
        min_value=1,
        max_value=6,
        value=int(st.session_state.top_k),
        help="Higher values increase recall but may add noise.",
    )

    st.divider()
    st.caption("Tip: For real audits, replace the sample regulations with official documents.")


st.title("FSSAI-Compliance Copilot")
st.caption(
    "Upload a facility SOP/manual and generate a retrieval-grounded GAP analysis against your local FSSAI/HACCP knowledge base."
)


top_left, top_right = st.columns([1.15, 0.85], gap="large")

with top_left:
    with st.container(border=True):
        st.subheader("1) Upload Facility Document")
        uploaded = st.file_uploader(
            "Facility SOP/manual",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT.",
        )

        if uploaded is None:
            st.info("Upload a file to begin.")
        else:
            uploaded_sig = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.uploaded_sig != uploaded_sig:
                st.session_state.uploaded_sig = uploaded_sig
                st.session_state.report = None
                try:
                    st.session_state.extracted = extract_text(uploaded.name, uploaded.getvalue())
                except Exception as e:
                    st.session_state.extracted = None
                    st.error(str(e))

            extracted = st.session_state.extracted
            if extracted and extracted.text and extracted.text.strip():
                st.success(f"Extracted text from: {extracted.filename}")
            elif extracted is not None:
                st.error("No text could be extracted from the uploaded file.")

    with st.container(border=True):
        st.subheader("2) Review Extracted Text")
        extracted = st.session_state.extracted
        if not extracted or not extracted.text or not extracted.text.strip():
            st.info("Waiting for a valid upload.")
        else:
            st.caption("Preview is limited for performance.")
            with st.expander("Show extracted text preview", expanded=False):
                st.text_area("", extracted.text[:12000], height=280)


with top_right:
    with st.container(border=True):
        st.subheader("3) Run Audit")
        st.write(
            f"**KB status:** {'Ready' if kb_ready else 'Missing index'}  "+
            f"**Retrieved clauses/claim:** {int(st.session_state.top_k)}"
        )

        extracted = st.session_state.extracted
        can_run = bool(kb_ready and extracted and extracted.text and extracted.text.strip())

        if not kb_ready:
            st.warning("Build the regulations index before running audits.")
        if extracted is None:
            st.info("Upload a document to enable the audit.")

        run = st.button("Run Compliance Audit", type="primary", use_container_width=True, disabled=not can_run)
        if run and can_run:
            with st.spinner("Auditing document against retrieved clauses..."):
                st.session_state.report = run_audit(
                    extracted.filename,
                    extracted.text,
                    top_k=int(st.session_state.top_k),
                )

    with st.container(border=True):
        st.subheader("Summary")
        report = st.session_state.report
        if report is None:
            st.caption("Run an audit to see metrics and findings.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall compliance", f"{report.scorecard.overall_percent}%")
            m2.metric("Claims extracted", len(report.claims))
            m3.metric("Gaps detected", len(report.gaps))
            if report.notes:
                for n in report.notes:
                    st.info(n)


report = st.session_state.report
if report is None:
    st.stop()

st.subheader("Scorecard")
score_rows = [
    {"Category": k, "Score (%)": v}
    for k, v in report.scorecard.by_category.items()
    if v > 0
]
if score_rows:
    score_df = pd.DataFrame(score_rows).sort_values("Score (%)")
    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn(width="medium"),
            "Score (%)": st.column_config.NumberColumn(format="%.1f", width="small"),
        },
    )
else:
    st.write("No category scores available.")

st.subheader("GAP Findings")
if not report.gaps:
    st.success("No gaps detected from retrieved clauses (or insufficient evidence).")
else:
    gap_rows = [
        {
            "Severity": g.severity,
            "Category": g.category,
            "Claim": g.claim,
            "Clause": g.clause,
            "Source": g.source,
            "Recommendation": g.recommendation,
        }
        for g in report.gaps
    ]
    gap_df = pd.DataFrame(gap_rows)
    if not gap_df.empty and "Severity" in gap_df.columns:
        gap_df["_sev_rank"] = gap_df["Severity"].map(_severity_rank)
        gap_df = gap_df.sort_values(["_sev_rank", "Category"], ascending=[True, True]).drop(columns=["_sev_rank"])

    st.dataframe(
        gap_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Severity": st.column_config.TextColumn(width="small"),
            "Category": st.column_config.TextColumn(width="medium"),
            "Claim": st.column_config.TextColumn(width="large"),
            "Clause": st.column_config.TextColumn(width="large"),
            "Source": st.column_config.TextColumn(width="medium"),
            "Recommendation": st.column_config.TextColumn(width="large"),
        },
    )

st.subheader("Download")
with st.spinner("Generating PDF report..."):
    pdf_bytes = render_pdf(report)
st.download_button(
    "Download PDF Audit Report",
    data=pdf_bytes,
    file_name=f"FSSAI_Audit_{Path(report.facility_filename).stem}.pdf",
    mime="application/pdf",
    use_container_width=False,
)
