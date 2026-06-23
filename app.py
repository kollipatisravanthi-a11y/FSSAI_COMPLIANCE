from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from fssai_copilot.ingestion import extract_text
from fssai_copilot.orchestrator import run_audit
from fssai_copilot.reporting import render_pdf
from fssai_copilot.vectorstore import VectorStoreConfig, index_exists

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FSSAI Compliance Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --brand-deep:   #0B2B45;
    --brand-mid:    #1A5276;
    --brand-accent: #1ABC9C;
    --brand-warn:   #E67E22;
    --brand-danger: #C0392B;
    --brand-light:  #EAF4FB;
    --surface:      #F7FAFB;
    --card:         #FFFFFF;
    --border:       #D6E4EE;
    --text-pri:     #0D1B2A;
    --text-sec:     #4A6274;
    --text-muted:   #8AA4B8;
    --radius:       12px;
    --radius-lg:    18px;
    --shadow:       0 2px 16px rgba(11,43,69,0.08);
    --shadow-lg:    0 8px 32px rgba(11,43,69,0.14);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--surface) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-pri);
}
[data-testid="stHeader"] { background: var(--brand-deep) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--brand-deep) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #CDE0EC !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; font-family:'Space Grotesk',sans-serif; }
[data-testid="stSidebar"] hr  { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--brand-accent) !important; }

[data-testid="stMainBlockContainer"] { padding: 0 2rem 3rem; }

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, var(--brand-deep) 0%, var(--brand-mid) 60%, #117A65 100%);
    border-radius: var(--radius-lg);
    padding: 2.4rem 2.8rem 2rem;
    margin: 1.6rem 0 2rem;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='20'/%3E%3C/g%3E%3C/svg%3E");
}
.hero-badge {
    display:inline-block;
    background:rgba(26,188,156,0.18); border:1px solid rgba(26,188,156,0.4);
    color:#A8F0E3 !important; font-size:0.72rem; font-weight:600;
    letter-spacing:0.12em; text-transform:uppercase;
    padding:0.3rem 0.8rem; border-radius:999px; margin-bottom:0.9rem;
}
.hero-title {
    font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:700;
    color:#FFFFFF !important; line-height:1.15; margin:0 0 0.5rem;
}
.hero-sub { font-size:0.97rem; color:rgba(255,255,255,0.65) !important; max-width:640px; line-height:1.6; margin:0; }
.hero-shield { position:absolute; right:2.4rem; top:50%; transform:translateY(-50%); font-size:5rem; opacity:0.12; }

/* Step cards */
.step-label { display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem; }
.step-chip {
    background:var(--brand-deep); color:#fff !important;
    font-size:0.68rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
    padding:0.28rem 0.72rem; border-radius:999px;
}
.step-title {
    font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:700;
    color:var(--brand-deep) !important; margin:0;
}

/* Metric tiles — nowrap fix */
.metric-tile {
    background:var(--brand-light); border:1px solid var(--border);
    border-radius:var(--radius); padding:1.1rem 0.6rem 0.9rem;
    text-align:center; position:relative; overflow:hidden;
}
.metric-tile::after {
    content:''; position:absolute; bottom:0; left:0; right:0;
    height:3px; background:var(--brand-accent);
    border-radius:0 0 var(--radius) var(--radius);
}
.metric-value {
    font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700;
    color:var(--brand-deep) !important; line-height:1; margin-bottom:0.25rem;
}
/* KEY FIX: nowrap so "Compliance" stays one line */
.metric-label {
    font-size:0.68rem; font-weight:700; color:var(--text-muted) !important;
    text-transform:uppercase; letter-spacing:0.08em;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}

/* Status pills */
.status-pill {
    display:inline-flex; align-items:center; gap:0.4rem;
    padding:0.35rem 0.9rem; border-radius:999px;
    font-size:0.78rem; font-weight:600; margin-bottom:0.5rem;
}
.pill-ok   { background:#D5F5EC; color:#0E6655 !important; border:1px solid #A2DFC6; }
.pill-warn { background:#FEF9E7; color:#7D6608 !important; border:1px solid #F7DC6F; }
.pill-err  { background:#FDECEA; color:#922B21 !important; border:1px solid #F1948A; }
.pill-dot  { width:7px; height:7px; border-radius:50%; background:currentColor; }

/* Buttons */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,var(--brand-accent),#17A589) !important;
    color:#fff !important; border:none !important;
    border-radius:var(--radius) !important; font-weight:700 !important;
    font-size:0.95rem !important; padding:0.75rem 1.5rem !important;
    box-shadow:0 4px 14px rgba(26,188,156,0.35) !important;
    transition:all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled) { transform:translateY(-1px) !important; box-shadow:0 6px 20px rgba(26,188,156,0.45) !important; }
.stButton > button[kind="primary"]:disabled { background:#B2C4CE !important; box-shadow:none !important; }
.stDownloadButton > button { background:var(--brand-deep) !important; color:#fff !important; border:none !important; border-radius:var(--radius) !important; font-weight:600 !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] { background:#EAF4FB !important; border:2px dashed var(--brand-mid) !important; border-radius:var(--radius) !important; }

/* Dataframe header */
[data-testid="stDataFrame"] { border-radius:var(--radius) !important; overflow:hidden; }
[data-testid="stDataFrame"] th { background:var(--brand-deep) !important; color:#fff !important; font-size:0.77rem; text-transform:uppercase; letter-spacing:0.07em; }

/* Section headers */
.section-header { display:flex; align-items:center; gap:0.7rem; padding:1.6rem 0 0.6rem; border-top:1px solid var(--border); margin-top:1rem; }
.section-icon { width:32px; height:32px; background:var(--brand-light); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1rem; }
.section-title { font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700; color:var(--brand-deep) !important; margin:0; }

/* Expander */
[data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:var(--radius) !important; }

/* Sidebar KB card */
.kb-card { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.09); border-radius:10px; padding:0.9rem 1rem; margin:0.5rem 0 1rem; font-size:0.83rem; }
.kb-card code { background:rgba(255,255,255,0.08); border-radius:4px; padding:1px 5px; color:#A8F0E3 !important; }

/* API key box in sidebar */
.api-box { background:rgba(26,188,156,0.08); border:1px solid rgba(26,188,156,0.25); border-radius:10px; padding:0.85rem 1rem; margin:0.4rem 0 0.8rem; font-size:0.81rem; line-height:1.6; }
.api-box code { background:rgba(26,188,156,0.15); border-radius:4px; padding:1px 5px; color:#A8F0E3 !important; }

[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background:rgba(255,255,255,0.07) !important;
    border:1px solid rgba(255,255,255,0.15) !important;
    color:#fff !important; border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── State & config ────────────────────────────────────────────────────────────
cfg = VectorStoreConfig()
kb_ready = index_exists(cfg)

for key, default in [("top_k", 3), ("extracted", None), ("uploaded_sig", None), ("report", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

def _severity_rank(sev: str) -> int:
    return {"Critical": 0, "Major": 1, "Minor": 2}.get(sev, 99)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.8rem'>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.25rem;font-weight:700;color:#fff'>
            🛡️ FSSAI Copilot
        </div>
        <div style='font-size:0.78rem;color:#7DAECC;margin-top:0.3rem'>AI-powered compliance auditing</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── OpenAI API Key section ────────────────────────────────────────────────
    st.markdown("**🔑 OpenAI API Key**")
    api_key_input = st.text_input(
        "Enter your OpenAI key",
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.markdown('<span class="status-pill pill-ok" style="font-size:0.75rem">✓ Key set — RAG mode active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-warn" style="font-size:0.75rem">⚠ No key — heuristic mode</span>', unsafe_allow_html=True)

    st.markdown("""
    <div class="api-box">
        Get your key at <code>platform.openai.com</code><br>
        Without a key, audits run in <em>offline heuristic mode</em> — useful for demos but less accurate.<br>
        With a key, the auditor uses strict <strong>RAG mode</strong> grounded to retrieved clauses.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Knowledge Base ────────────────────────────────────────────────────────
    st.markdown("**📚 Knowledge Base**")
    pill_cls = "pill-ok" if kb_ready else "pill-warn"
    pill_label = "✓ Index ready" if kb_ready else "⚠ Index missing"
    st.markdown(f'<span class="status-pill {pill_cls}">{pill_label}</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kb-card">
        Path: <code>{cfg.persist_dir}</code><br>
        Collection: <code>{cfg.collection}</code>
    </div>
    """, unsafe_allow_html=True)
    if not kb_ready:
        st.markdown("""
        <div style='font-size:0.82rem;color:#F9A825;background:rgba(249,168,37,0.1);border:1px solid rgba(249,168,37,0.3);border-radius:8px;padding:0.7rem 0.9rem;'>
        Add PDFs under <code style='color:#FFD54F'>data/regulations/</code> then:<br><br>
        <code style='color:#A8F0E3'>python scripts/build_vectorstore.py</code>
        </div>
        """, unsafe_allow_html=True)
    st.divider()

    # ── Audit Settings ────────────────────────────────────────────────────────
    st.markdown("**⚙️ Audit Settings**")
    st.session_state.top_k = st.slider(
        "Clauses retrieved per claim", min_value=1, max_value=6,
        value=int(st.session_state.top_k),
        help="Higher values increase recall but may add noise.",
    )
    st.divider()
    st.markdown('<div style="font-size:0.78rem;color:#5B8FA8;line-height:1.6">💡 <em>For real audits, replace sample regulations with official FSSAI/HACCP documents.</em></div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-shield">🛡️</div>
    <div class="hero-badge">AI-Powered · Retrieval-Grounded</div>
    <div class="hero-title">FSSAI Compliance Copilot</div>
    <p class="hero-sub">Upload a facility SOP or manual and instantly generate a clause-grounded GAP analysis against your local FSSAI / HACCP knowledge base.</p>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.markdown('<div class="step-label"><span class="step-chip">Step 1</span><span class="step-title">Upload Facility Document</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop your SOP / manual here",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF · DOCX · TXT",
    )

    if uploaded is None:
        st.markdown('<div style="text-align:center;padding:1.2rem 0;color:#8AA4B8;font-size:0.9rem;">📂 No file selected — upload to begin</div>', unsafe_allow_html=True)
    else:
        uploaded_sig = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.uploaded_sig != uploaded_sig:
            st.session_state.uploaded_sig = uploaded_sig
            st.session_state.report = None
            try:
                st.session_state.extracted = extract_text(uploaded.name, uploaded.getvalue())
            except Exception as e:
                st.session_state.extracted = None
                st.error(f"Extraction failed: {e}")

        extracted = st.session_state.extracted
        if extracted and extracted.text and extracted.text.strip():
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.6rem;background:#D5F5EC;border:1px solid #A2DFC6;
                         border-radius:10px;padding:0.7rem 1rem;margin:0.5rem 0;'>
                <span style='font-size:1.2rem'>✅</span>
                <span style='color:#0E6655;font-weight:600;font-size:0.88rem'>Extracted — <em>{extracted.filename}</em></span>
            </div>""", unsafe_allow_html=True)
        elif extracted is not None:
            st.error("No readable text found in the uploaded file.")

    st.divider()
    st.markdown('<div class="step-label" style="padding-top:0.4rem"><span class="step-chip">Step 2</span><span class="step-title">Review Extracted Text</span></div>', unsafe_allow_html=True)

    extracted = st.session_state.extracted
    if not extracted or not extracted.text or not extracted.text.strip():
        st.markdown('<div style="color:#8AA4B8;font-size:0.9rem;padding:0.5rem 0">Waiting for a valid document upload…</div>', unsafe_allow_html=True)
    else:
        word_count = len(extracted.text.split())
        char_count = len(extracted.text)
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-bottom:0.8rem;'>
            <span style='background:#EAF4FB;border:1px solid #D6E4EE;border-radius:8px;padding:0.35rem 0.8rem;font-size:0.78rem;color:#1A5276;font-weight:600;'>📝 {word_count:,} words</span>
            <span style='background:#EAF4FB;border:1px solid #D6E4EE;border-radius:8px;padding:0.35rem 0.8rem;font-size:0.78rem;color:#1A5276;font-weight:600;'>🔤 {char_count:,} characters</span>
        </div>""", unsafe_allow_html=True)
        with st.expander("👁️ Preview extracted text", expanded=False):
            st.text_area("", extracted.text[:12000], height=280, label_visibility="collapsed")

with col_right:
    st.markdown('<div class="step-label"><span class="step-chip">Step 3</span><span class="step-title">Run Compliance Audit</span></div>', unsafe_allow_html=True)

    extracted = st.session_state.extracted
    can_run = bool(kb_ready and extracted and extracted.text and extracted.text.strip())

    kb_pill = '<span class="status-pill pill-ok"><span class="pill-dot"></span>KB ready</span>' if kb_ready \
              else '<span class="status-pill pill-warn"><span class="pill-dot"></span>KB missing</span>'
    doc_pill = '<span class="status-pill pill-ok"><span class="pill-dot"></span>Doc loaded</span>' if (extracted and extracted.text) \
               else '<span class="status-pill pill-err"><span class="pill-dot"></span>No doc</span>'
    api_pill = '<span class="status-pill pill-ok"><span class="pill-dot"></span>RAG mode</span>' if os.environ.get("OPENAI_API_KEY") \
               else '<span class="status-pill pill-warn"><span class="pill-dot"></span>Offline mode</span>'
    st.markdown(f"""
    <div style='display:flex;gap:0.6rem;margin-bottom:1rem;flex-wrap:wrap;'>
        {kb_pill}{doc_pill}{api_pill}
        <span class="status-pill" style="background:#EAF4FB;border:1px solid #D6E4EE;color:#1A5276 !important;">
            🔍 {int(st.session_state.top_k)} clauses/claim
        </span>
    </div>""", unsafe_allow_html=True)

    if not kb_ready:
        st.warning("Build the regulations index before running an audit.")
    if extracted is None:
        st.info("Upload a document to enable auditing.")

    run = st.button("🚀 Run Compliance Audit", type="primary", use_container_width=True, disabled=not can_run)

    if run and can_run:
        with st.spinner("Auditing against retrieved clauses…"):
            st.session_state.report = run_audit(
                extracted.filename, extracted.text, top_k=int(st.session_state.top_k),
            )

    st.divider()
    st.markdown('<div class="step-label"><span class="step-chip">Results</span><span class="step-title">Audit Summary</span></div>', unsafe_allow_html=True)

    report = st.session_state.report
    if report is None:
        st.markdown('<div style="text-align:center;padding:1.8rem 0;color:#8AA4B8;font-size:0.9rem;">📊 Run an audit to see metrics</div>', unsafe_allow_html=True)
    else:
        gaps = len(report.gaps)
        gap_color = "#C0392B" if gaps > 0 else "#0E6655"
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{report.scorecard.overall_percent}%</div>
                <div class="metric-label">Compliance</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{len(report.claims)}</div>
                <div class="metric-label">Claims</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-tile" style="--brand-accent:{gap_color}">
                <div class="metric-value" style="color:{gap_color} !important">{gaps}</div>
                <div class="metric-label">Gaps</div>
            </div>""", unsafe_allow_html=True)

        if report.notes:
            for n in report.notes:
                st.info(n)

# ── Below the fold ────────────────────────────────────────────────────────────
report = st.session_state.report
if report is None:
    st.stop()

# ── Charts row ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">📈</div>
    <span class="section-title">Visual Analysis</span>
</div>
""", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns([1, 1.6], gap="large")

# Donut — overall compliance
with chart_col1:
    score = report.scorecard.overall_percent
    gap_pct = round(100 - score, 1)
    donut = go.Figure(go.Pie(
        values=[score, gap_pct],
        labels=["Compliant", "Gap"],
        hole=0.68,
        marker_colors=["#1ABC9C", "#E74C3C"],
        textinfo="none",
        hovertemplate="%{label}: %{value}%<extra></extra>",
    ))
    donut.add_annotation(
        text=f"<b>{score}%</b>",
        x=0.5, y=0.52, font=dict(size=28, color="#0B2B45", family="Space Grotesk"),
        showarrow=False,
    )
    donut.add_annotation(
        text="Compliance",
        x=0.5, y=0.38, font=dict(size=12, color="#8AA4B8"),
        showarrow=False,
    )
    donut.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05, font=dict(size=12)),
        height=260,
    )
    st.markdown('<div style="font-weight:600;color:#0B2B45;font-size:0.9rem;margin-bottom:0.4rem;text-align:center;">Overall Compliance Score</div>', unsafe_allow_html=True)
    st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

# Horizontal bar — category scores
with chart_col2:
    score_rows = [
        {"Category": k, "Score": v}
        for k, v in report.scorecard.by_category.items()
        if v > 0
    ]
    if score_rows:
        bar_df = pd.DataFrame(score_rows).sort_values("Score")
        colors = ["#E74C3C" if s < 50 else "#E67E22" if s < 75 else "#1ABC9C" for s in bar_df["Score"]]
        bar_fig = go.Figure(go.Bar(
            x=bar_df["Score"],
            y=bar_df["Category"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in bar_df["Score"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        bar_fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#E8EEF3", ticksuffix="%", title=""),
            yaxis=dict(showgrid=False, title=""),
            height=260,
            font=dict(family="Inter", size=12, color="#4A6274"),
        )
        st.markdown('<div style="font-weight:600;color:#0B2B45;font-size:0.9rem;margin-bottom:0.4rem;text-align:center;">Score by Category</div>', unsafe_allow_html=True)
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No category data to chart.")

# ── Severity breakdown chart (if gaps exist) ──────────────────────────────────
if report.gaps:
    sevs = [g.severity for g in report.gaps]
    cats = [g.category for g in report.gaps]

    sev_counts = {"Critical": sevs.count("Critical"), "Major": sevs.count("Major"), "Minor": sevs.count("Minor")}
    sev_colors  = {"Critical": "#C0392B", "Major": "#E67E22", "Minor": "#3498DB"}

    cat_chart_col, sev_chart_col = st.columns(2, gap="large")

    with sev_chart_col:
        sev_fig = go.Figure(go.Bar(
            x=list(sev_counts.keys()),
            y=list(sev_counts.values()),
            marker_color=[sev_colors[s] for s in sev_counts],
            text=list(sev_counts.values()),
            textposition="outside",
            hovertemplate="%{x}: %{y} gaps<extra></extra>",
        ))
        sev_fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="#E8EEF3", title="Count"),
            height=240,
            font=dict(family="Inter", size=12, color="#4A6274"),
        )
        st.markdown('<div style="font-weight:600;color:#0B2B45;font-size:0.9rem;margin-bottom:0.4rem;text-align:center;">Gaps by Severity</div>', unsafe_allow_html=True)
        st.plotly_chart(sev_fig, use_container_width=True, config={"displayModeBar": False})

    with cat_chart_col:
        from collections import Counter
        cat_counts = Counter(cats)
        cat_df = pd.DataFrame({"Category": list(cat_counts.keys()), "Count": list(cat_counts.values())}).sort_values("Count")
        cat_fig = go.Figure(go.Bar(
            x=cat_df["Count"],
            y=cat_df["Category"],
            orientation="h",
            marker_color="#1A5276",
            text=cat_df["Count"],
            textposition="outside",
            hovertemplate="%{y}: %{x} gaps<extra></extra>",
        ))
        cat_fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#E8EEF3", title="Gaps"),
            yaxis=dict(showgrid=False, title=""),
            height=240,
            font=dict(family="Inter", size=12, color="#4A6274"),
        )
        st.markdown('<div style="font-weight:600;color:#0B2B45;font-size:0.9rem;margin-bottom:0.4rem;text-align:center;">Gaps by Category</div>', unsafe_allow_html=True)
        st.plotly_chart(cat_fig, use_container_width=True, config={"displayModeBar": False})

# ── Scorecard table ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">📊</div>
    <span class="section-title">Scorecard by Category</span>
</div>
""", unsafe_allow_html=True)

score_rows = [{"Category": k, "Score (%)": v} for k, v in report.scorecard.by_category.items() if v > 0]
if score_rows:
    score_df = pd.DataFrame(score_rows).sort_values("Score (%)")
    st.dataframe(
        score_df, use_container_width=True, hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%.1f%%", min_value=0, max_value=100, width="large"),
        },
    )
else:
    st.info("No category scores available.")

# ── GAP findings ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">⚠️</div>
    <span class="section-title">GAP Findings</span>
</div>
""", unsafe_allow_html=True)

if not report.gaps:
    st.markdown("""
    <div style='background:#D5F5EC;border:1px solid #A2DFC6;border-radius:12px;padding:1.2rem 1.5rem;display:flex;align-items:center;gap:0.8rem;'>
        <span style='font-size:1.5rem'>✅</span>
        <span style='color:#0E6655;font-weight:600;'>No gaps detected from retrieved clauses.</span>
    </div>""", unsafe_allow_html=True)
else:
    sevs2 = [g.severity for g in report.gaps]
    crit, major, minor = sevs2.count("Critical"), sevs2.count("Major"), sevs2.count("Minor")
    st.markdown(f"""
    <div style='display:flex;gap:0.7rem;margin-bottom:1rem;flex-wrap:wrap;'>
        <span class="status-pill pill-err" style="font-size:0.82rem">🔴 Critical: {crit}</span>
        <span class="status-pill pill-warn" style="font-size:0.82rem">🟡 Major: {major}</span>
        <span style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 0.9rem;border-radius:999px;font-size:0.82rem;font-weight:600;background:#EBF5FB;border:1px solid #AED6F1;color:#1A5276 !important;">🔵 Minor: {minor}</span>
    </div>""", unsafe_allow_html=True)

    gap_rows = [{"Severity": g.severity, "Category": g.category, "Claim": g.claim, "Clause": g.clause, "Source": g.source, "Recommendation": g.recommendation} for g in report.gaps]
    gap_df = pd.DataFrame(gap_rows)
    if not gap_df.empty:
        gap_df["_sev_rank"] = gap_df["Severity"].map(_severity_rank)
        gap_df = gap_df.sort_values(["_sev_rank", "Category"]).drop(columns=["_sev_rank"])
    st.dataframe(
        gap_df, use_container_width=True, hide_index=True,
        column_config={
            "Severity":       st.column_config.TextColumn("Severity", width="small"),
            "Category":       st.column_config.TextColumn("Category", width="medium"),
            "Claim":          st.column_config.TextColumn("Claim", width="large"),
            "Clause":         st.column_config.TextColumn("Clause", width="large"),
            "Source":         st.column_config.TextColumn("Source", width="medium"),
            "Recommendation": st.column_config.TextColumn("Recommendation", width="large"),
        },
    )

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">📥</div>
    <span class="section-title">Download Audit Report</span>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div style='background:var(--brand-light);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.5rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.5rem;'>
    <span style='font-size:1.4rem'>📄</span>
    <div>
        <div style='font-weight:600;color:#0B2B45;font-size:0.95rem;'>PDF Compliance Report</div>
        <div style='font-size:0.8rem;color:#4A6274;'>Complete audit with all findings, sources, and recommendations.</div>
    </div>
</div>""", unsafe_allow_html=True)

with st.spinner("Generating PDF…"):
    pdf_bytes = render_pdf(report)

st.download_button(
    "⬇️ Download PDF Audit Report",
    data=pdf_bytes,
    file_name=f"FSSAI_Audit_{Path(report.facility_filename).stem}.pdf",
    mime="application/pdf",
)