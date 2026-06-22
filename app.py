from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from fssai_copilot.database import AuditDatabase
from fssai_copilot.ingestion import extract_text
from fssai_copilot.orchestrator import run_audit
from fssai_copilot.reporting import render_pdf
from fssai_copilot.vectorstore import VectorStoreConfig, index_exists


st.set_page_config(
    page_title="FSSAI-Compliance Copilot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Color Theme (Classic UI inspired)
COLORS = {
    "primary": "#1f77b4",      # Deep blue
    "success": "#2ecc71",      # Green
    "warning": "#f39c12",      # Orange
    "danger": "#e74c3c",       # Red
    "info": "#3498db",         # Light blue
    "light_bg": "#ecf0f1",     # Very light gray
    "dark_text": "#2c3e50",    # Dark gray/blue
    "border": "#bdc3c7",       # Gray
}

# Custom CSS for consistent professional styling
st.markdown(f"""
<style>
    * {{ color: {COLORS['dark_text']}; }}
    
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #1565a0 100%);
        padding: 40px;
        border-radius: 12px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }}
    
    .main-header p {{
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.95;
    }}
    
    .metric-container {{
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid {COLORS['primary']};
    }}
    
    .metric-label {{
        font-size: 0.9em;
        color: #7f8c8d;
        font-weight: 500;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-value {{
        font-size: 2.2em;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 5px 0;
    }}
    
    .status-ready {{
        color: {COLORS['success']};
        font-weight: bold;
        font-size: 1.1em;
    }}
    
    .status-missing {{
        color: {COLORS['danger']};
        font-weight: bold;
        font-size: 1.1em;
    }}
    
    .gap-card {{
        border-radius: 8px;
        padding: 18px;
        margin: 12px 0;
        border-left: 5px solid;
        background-color: #f8f9fa;
    }}
    
    .gap-critical {{
        border-left-color: {COLORS['danger']};
        background-color: #fadbd8;
    }}
    
    .gap-major {{
        border-left-color: {COLORS['warning']};
        background-color: #fdebd0;
    }}
    
    .gap-minor {{
        border-left-color: {COLORS['success']};
        background-color: #d5f4e6;
    }}
    
    .gap-title {{
        font-weight: 700;
        font-size: 1.05em;
        margin-bottom: 10px;
    }}
    
    .gap-content {{
        font-size: 0.95em;
        line-height: 1.6;
    }}
    
    .gap-content strong {{
        color: {COLORS['dark_text']};
    }}
    
    .section-header {{
        color: {COLORS['primary']};
        font-weight: 700;
        font-size: 1.3em;
        margin: 20px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid {COLORS['light_bg']};
    }}
</style>
""", unsafe_allow_html=True)

# Initialize state
cfg = VectorStoreConfig()
kb_ready = index_exists(cfg)
db = AuditDatabase()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

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


def _create_metric_card(label: str, value: str, color: str) -> str:
    """Create styled metric card HTML."""
    return f"""
    <div class="metric-container" style="border-top-color: {color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """


def _configure_matplotlib():
    """Configure matplotlib for consistent, clean styling."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": COLORS["border"],
        "axes.linewidth": 0.8,
        "xtick.color": COLORS["dark_text"],
        "ytick.color": COLORS["dark_text"],
        "text.color": COLORS["dark_text"],
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "grid.alpha": 0.3,
        "grid.color": COLORS["border"],
    })


# Sidebar
with st.sidebar:
    st.markdown(f"### ⚙️ Configuration")
    
    st.markdown("#### 📚 Knowledge Base")
    if kb_ready:
        st.markdown(f'<p class="status-ready">✓ Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="status-missing">✗ Missing</p>', unsafe_allow_html=True)
    st.caption(f"Collection: `{cfg.collection}`")
    
    st.markdown("#### 🎛️ Audit Settings")
    st.session_state.top_k = st.slider(
        "Clauses per claim",
        min_value=1,
        max_value=6,
        value=int(st.session_state.top_k),
        help="Higher values increase recall.",
    )
    
    st.divider()
    st.markdown("""
    **💡 Quick Tips:**
    - Upload PDF, DOCX, or TXT files
    - Max file size: 50MB
    - Review all recommendations
    - Export reports as PDF
    """)


# Main Header
st.markdown("""
<div class="main-header">
    <h1>🔍 FSSAI Compliance Copilot</h1>
    <p>AI-Powered Compliance Gap Analysis • Regulatory Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Two-column layout for upload and audit
col_left, col_right = st.columns([1.15, 0.85], gap="large")

# LEFT: Document Upload and Preview
with col_left:
    st.markdown('<h3 class="section-header">📄 Step 1: Upload Facility Document</h3>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Select document",
        type=["pdf", "docx", "txt"],
        help="Supported: PDF, DOCX, TXT (max 50MB)",
    )

    if uploaded is None:
        st.info("👉 Upload a facility SOP, manual, or compliance document to begin analysis")
    else:
        if uploaded.size > MAX_UPLOAD_SIZE:
            st.error(f"❌ File too large: {uploaded.size / (1024*1024):.1f}MB (max 50MB)")
            st.session_state.extracted = None
            st.session_state.report = None
        else:
            uploaded_sig = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.uploaded_sig != uploaded_sig:
                st.session_state.uploaded_sig = uploaded_sig
                st.session_state.report = None
                try:
                    st.session_state.extracted = extract_text(uploaded.name, uploaded.getvalue())
                except Exception as e:
                    st.session_state.extracted = None
                    st.error(f"❌ Extraction error: {str(e)}")

            extracted = st.session_state.extracted
            if extracted and extracted.text and extracted.text.strip():
                st.success(f"✓ Extracted: **{extracted.filename}**")
                st.caption(f"📊 {len(extracted.text):,} characters")
            elif extracted is not None:
                st.error("❌ No readable text found in file")

    st.markdown('<h3 class="section-header">📖 Step 2: Text Preview</h3>', unsafe_allow_html=True)
    extracted = st.session_state.extracted
    if not extracted or not extracted.text or not extracted.text.strip():
        st.info("Waiting for document upload...")
    else:
        with st.expander("📋 View extracted content (first 12K chars)", expanded=False):
            st.text_area("", extracted.text[:12000], height=300, disabled=True, label_visibility="collapsed")

# RIGHT: Audit Controls and Summary
with col_right:
    st.markdown('<h3 class="section-header">⚡ Step 3: Run Audit</h3>', unsafe_allow_html=True)
    
    # Status row
    extracted = st.session_state.extracted
    can_run = bool(kb_ready and extracted and extracted.text and extracted.text.strip())
    
    kb_col, k_col = st.columns(2)
    with kb_col:
        st.metric("KB Status", "Ready" if kb_ready else "Missing")
    with k_col:
        st.metric("Top-K Clauses", int(st.session_state.top_k))
    
    if not kb_ready:
        st.warning("⚠️ Knowledge base not ready. Build regulations index first.")
    
    st.divider()
    
    # Run button
    run_btn = st.button(
        "🚀 Run Compliance Audit",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )
    
    if run_btn and can_run:
        with st.spinner("🔄 Analyzing document against regulations..."):
            st.session_state.report = run_audit(
                extracted.filename,
                extracted.text,
                top_k=int(st.session_state.top_k),
            )
        st.success("✓ Audit completed!")

    st.divider()
    st.markdown('<h3 class="section-header">📊 Quick Summary</h3>', unsafe_allow_html=True)
    
    report = st.session_state.report
    if report is None:
        st.info("Run audit to see results")
    else:
        overall = report.scorecard.overall_percent
        score_color = COLORS["success"] if overall >= 70 else COLORS["warning"] if overall >= 40 else COLORS["danger"]
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(_create_metric_card("Compliance", f"{overall}%", score_color), unsafe_allow_html=True)
        with m2:
            st.markdown(_create_metric_card("Claims", str(len(report.claims)), COLORS["info"]), unsafe_allow_html=True)
        with m3:
            st.markdown(_create_metric_card("Gaps", str(len(report.gaps)), COLORS["danger"]), unsafe_allow_html=True)
        
        if report.notes:
            st.warning("⚠️ Audit Notes:")
            for note in report.notes:
                st.write(f"• {note}")

    st.divider()
    st.markdown('<h3 class="section-header">📜 Recent Audits</h3>', unsafe_allow_html=True)
    history = db.list_reports(limit=5)
    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.caption("No audit history")


# Stop if no report
report = st.session_state.report
if report is None:
    st.stop()

# MAIN RESULTS SECTION
st.divider()
st.markdown('<h2 class="section-header">📈 Compliance Analysis Results</h2>', unsafe_allow_html=True)

# Graphs
_configure_matplotlib()

g_col1, g_col2 = st.columns(2, gap="large")

with g_col1:
    st.markdown("#### 📊 Category Scorecard")
    score_rows = [
        {"Category": k, "Score (%)": v}
        for k, v in report.scorecard.by_category.items()
        if v > 0
    ]
    if score_rows:
        score_df = pd.DataFrame(score_rows).sort_values("Score (%)", ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
        colors_list = [
            COLORS["success"] if x >= 70 else COLORS["warning"] if x >= 40 else COLORS["danger"]
            for x in score_df["Score (%)"]
        ]
        bars = ax.barh(score_df["Category"], score_df["Score (%)"], color=colors_list, edgecolor=COLORS["border"], linewidth=1.2)
        ax.set_xlabel("Compliance Score (%)", fontweight="600")
        ax.set_xlim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.0f}%', 
                   ha='left', va='center', fontweight="600", fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No category scores")

with g_col2:
    st.markdown("#### 🎯 Overall Compliance Status")
    overall = report.scorecard.overall_percent
    
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)
    sizes = [overall, 100 - overall]
    colors_pie = [
        COLORS["success"] if overall >= 70 else COLORS["warning"] if overall >= 40 else COLORS["danger"],
        COLORS["light_bg"]
    ]
    
    wedges, texts = ax.pie(sizes, labels=None, colors=colors_pie, startangle=90, 
                           wedgeprops={"edgecolor": COLORS["border"], "linewidth": 1.5})
    
    # Custom legend with better styling
    ax.text(0, -0.25, f"Compliant", ha='center', fontsize=11, fontweight="600", color=colors_pie[0])
    ax.text(0, -0.35, f"{overall}%", ha='center', fontsize=16, fontweight="700", color=colors_pie[0])
    
    ax.set_title("Overall Compliance", fontsize=13, fontweight="700", pad=20)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()

# Gaps by severity
st.markdown("#### 🚨 Gaps by Severity")
if not report.gaps:
    st.success("✓ No compliance gaps detected!")
else:
    severity_counts = {"Critical": 0, "Major": 0, "Minor": 0}
    for g in report.gaps:
        severity_counts[g.severity] = severity_counts.get(g.severity, 0) + 1
    
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        st.markdown(f"""
        <div class="metric-container" style="border-top-color: {COLORS['danger']};">
            <div class="metric-label">🔴 Critical</div>
            <div class="metric-value" style="color: {COLORS['danger']};">{severity_counts['Critical']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with s2:
        st.markdown(f"""
        <div class="metric-container" style="border-top-color: {COLORS['warning']};">
            <div class="metric-label">🟠 Major</div>
            <div class="metric-value" style="color: {COLORS['warning']};">{severity_counts['Major']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with s3:
        st.markdown(f"""
        <div class="metric-container" style="border-top-color: {COLORS['success']};">
            <div class="metric-label">🟢 Minor</div>
            <div class="metric-value" style="color: {COLORS['success']};">{severity_counts['Minor']}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Detailed findings table
st.markdown("#### 📋 Detailed Findings")
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
            "Score (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        },
    )

st.divider()

# Gap details
st.markdown("#### 📑 Gap Details")
if not report.gaps:
    st.success("✓ No gaps to address")
else:
    gap_rows = [
        {
            "Severity": g.severity,
            "Category": g.category,
            "Claim": g.claim[:80] + "..." if len(g.claim) > 80 else g.claim,
            "Recommendation": g.recommendation[:100] + "..." if len(g.recommendation) > 100 else g.recommendation,
        }
        for g in report.gaps
    ]
    gap_df = pd.DataFrame(gap_rows)
    if not gap_df.empty and "Severity" in gap_df.columns:
        gap_df["_rank"] = gap_df["Severity"].map(_severity_rank)
        gap_df = gap_df.sort_values(["_rank", "Category"]).drop(columns=["_rank"])
    
    st.dataframe(
        gap_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Severity": st.column_config.TextColumn(width="small"),
            "Category": st.column_config.TextColumn(width="medium"),
            "Claim": st.column_config.TextColumn(width="large"),
            "Recommendation": st.column_config.TextColumn(width="large"),
        },
    )

st.divider()

# Expandable detailed breakdown
st.markdown("#### 🔍 Detailed Gap Breakdown")
for i, gap in enumerate(report.gaps, 1):
    gap_class = f"gap-card gap-{gap.severity.lower()}"
    st.markdown(f"""
    <div class="{gap_class}">
        <div class="gap-title">Gap #{i}: {gap.category} [{gap.severity}]</div>
        <div class="gap-content">
            <p><strong>📍 Claim:</strong> {gap.claim}</p>
            <p><strong>📚 Regulation:</strong> {gap.clause}</p>
            <p><strong>📖 Source:</strong> {gap.source}</p>
            <p><strong>💡 Action:</strong> {gap.recommendation}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Download section
st.markdown("## 📥 Export Report")
with st.spinner("Generating PDF..."):
    pdf_bytes = render_pdf(report)

dl_col1, dl_col2 = st.columns([2, 1])
with dl_col1:
    st.download_button(
        "📄 Download Full PDF Audit Report",
        data=pdf_bytes,
        file_name=f"FSSAI_Audit_{Path(report.facility_filename).stem}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with dl_col2:
    st.caption(f"**Report:** {Path(report.facility_filename).stem}")

st.divider()

# Footer
st.markdown(f"""
<div style="background: {COLORS['light_bg']}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px;">
    <p><strong>✓ Audit Completed Successfully</strong></p>
    <p style="font-size: 0.95em; margin-top: 10px;">Share this report with your compliance team for action planning and remediation.</p>
</div>
""", unsafe_allow_html=True)
