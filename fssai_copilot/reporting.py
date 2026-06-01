from __future__ import annotations

import re

from fpdf import FPDF

from .models import AuditReport


def _latin1_safe(text: str) -> str:
    # Core PDF fonts in FPDF are effectively Latin-1; replace unsupported chars.
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _soft_wrap_long_tokens(text: str, max_token_len: int = 45) -> str:
    # FPDF multi_cell wraps on whitespace. If extraction produces very long tokens
    # (e.g., merged words, long IDs), multi_cell can fail with:
    # "Not enough horizontal space to render a single character".
    tokens = re.split(r"(\s+)", text)
    out: list[str] = []
    for tok in tokens:
        if tok.isspace() or len(tok) <= max_token_len:
            out.append(tok)
            continue
        chunks = [tok[i : i + max_token_len] for i in range(0, len(tok), max_token_len)]
        out.append(" ".join(chunks))
    return "".join(out)


def _pdf_multicell(pdf: FPDF, text: str, h: float = 5) -> None:
    # Ensure we always render from left margin so width isn't accidentally 0.
    pdf.set_x(pdf.l_margin)
    cleaned = text.replace("\r", "").replace("\t", "    ")
    cleaned = cleaned.replace("\u00a0", " ")  # NBSP
    cleaned = _soft_wrap_long_tokens(cleaned)
    cleaned = _latin1_safe(cleaned)
    # Use explicit cursor movement semantics (fpdf2).
    pdf.multi_cell(0, h, cleaned, new_x="LMARGIN", new_y="NEXT")


def render_pdf(report: AuditReport) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "FSSAI Compliance GAP Analysis Report", ln=1)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Facility file: {report.facility_filename}", ln=1)
    pdf.cell(0, 6, f"Generated: {report.created_at.isoformat()} UTC", ln=1)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Executive Summary", ln=1)
    pdf.set_font("Helvetica", "", 10)
    _pdf_multicell(pdf, f"Overall compliance score: {report.scorecard.overall_percent}%")

    if report.notes:
        pdf.ln(1)
        for n in report.notes:
            _pdf_multicell(pdf, f"Note: {n}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Scorecard", ln=1)
    pdf.set_font("Helvetica", "", 10)

    for cat, score in report.scorecard.by_category.items():
        if score <= 0:
            continue
        pdf.cell(0, 5, f"{cat}: {score:.1f}%", ln=1)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "GAP Findings", ln=1)

    if not report.gaps:
        pdf.set_font("Helvetica", "", 10)
        _pdf_multicell(pdf, "No gaps detected from retrieved clauses (or insufficient evidence).")
    else:
        for i, gap in enumerate(report.gaps, start=1):
            pdf.set_font("Helvetica", "B", 10)
            _pdf_multicell(pdf, f"{i}. [{gap.severity}] {gap.category}")
            pdf.set_font("Helvetica", "", 10)
            _pdf_multicell(pdf, f"Claim: {gap.claim}")
            _pdf_multicell(pdf, f"Clause: {gap.clause}")
            _pdf_multicell(pdf, f"Source: {gap.source}")
            _pdf_multicell(pdf, f"Recommendation: {gap.recommendation}")
            pdf.ln(1)

    # fpdf2 has returned different types across versions:
    # - older: `str` (latin-1)
    # - newer: `bytearray` (already bytes)
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode("latin-1", errors="replace")
