from __future__ import annotations

import os
import re
import json
from datetime import datetime

from .database import AuditDatabase
from .models import AuditReport, GapFinding, OperationalClaim, Scorecard
from .vectorstore import query_clauses


audit_db = AuditDatabase()


def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("OpenAI package is not installed. Install openai>=1.30 or run offline mode without OPENAI_API_KEY.")

    return OpenAI()


def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _json_loads_maybe(text: str):
    try:
        return json.loads(text)
    except Exception:
        # Try to recover from fenced blocks
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter for demo mode.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def auditor_agent_extract_claims(text: str) -> list[OperationalClaim]:
    """Extract operational claims from the facility document.

    - If `OPENAI_API_KEY` is set, you can extend this to an LLM call.
    - Default mode uses heuristics to keep the project runnable offline.
    """

    if _has_openai():
        client = _openai_client()
        sys = (
            "You extract operational compliance-relevant claims from facility documents. "
            "Return ONLY JSON. Do not add legal interpretations."
        )
        user = (
            "Extract up to 30 operational claims from the following facility text. "
            "Each claim must be a single, auditable statement about procedures/records/controls. "
            "Assign one category from: Hygiene, Processing, Storage/Temperature, Packaging/Labeling, "
            "Documentation/Records, Pest Control, Personnel/Training, Other. "
            "Return JSON with shape: {\"claims\":[{\"category\":...,\"text\":...,\"source_excerpt\":...}]}\n\n"
            f"FACILITY_TEXT:\n{text[:24000]}"
        )

        try:
            resp = client.chat.completions.create(
                model=_openai_model(),
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=1200,
            )
            output_text = resp.choices[0].message.get("content", "")
            data = _json_loads_maybe(output_text)
            raw_claims = data.get("claims", []) if isinstance(data, dict) else []
            claims: list[OperationalClaim] = []
            for rc in raw_claims:
                try:
                    claims.append(OperationalClaim(**rc))
                except Exception:
                    continue
            return claims[:40]
        except Exception:
            # Fall back to deterministic heuristic extraction.
            pass

    sentences = _split_sentences(text)

    keywords = {
        "Hygiene": ["clean", "sanitize", "disinfect", "wash", "mop", "deep clean"],
        "Storage/Temperature": ["temperature", "chiller", "freezer", "cold", "hot hold"],
        "Pest Control": ["pest", "rodent", "insect", "fumigation"],
        "Documentation/Records": ["record", "log", "register", "document", "trace"],
        "Personnel/Training": ["training", "hairnet", "gloves", "uniform", "handwash"],
        "Packaging/Labeling": ["label", "expiry", "batch", "pack"],
        "Processing": ["cook", "boil", "pasteur", "process", "prep"],
    }

    claims: list[OperationalClaim] = []
    for s in sentences:
        s_low = s.lower()
        chosen = None
        for cat, keys in keywords.items():
            if any(k in s_low for k in keys):
                chosen = cat
                break
        if not chosen:
            continue
        claims.append(OperationalClaim(category=chosen, text=s, source_excerpt=s[:280]))

    # If document is huge and we got too many, keep top-N to maintain responsiveness.
    return claims[:40]


def regulatory_agent_retrieve(claim: OperationalClaim, top_k: int = 3):
    return query_clauses(claim.text, top_k=top_k)


def analyst_agent_compare(claim: OperationalClaim, clauses) -> list[GapFinding]:
    """Detect contradictions between a claim and retrieved clauses.

    Demo heuristic:
    - Flags frequency conflicts (daily vs weekly vs monthly)
    - Flags missing temperature numbers when clauses mention temperature

    In LLM mode you can tighten this further, but we keep it deterministic here.
    """

    gaps: list[GapFinding] = []

    claim_text = claim.text.lower()

    def freq(s: str) -> str | None:
        if "daily" in s or "every day" in s:
            return "daily"
        if "weekly" in s or "every week" in s:
            return "weekly"
        if "monthly" in s or "every month" in s:
            return "monthly"
        return None

    claim_freq = freq(claim_text)

    for clause in clauses:
        clause_text = clause.text.lower()
        clause_freq = freq(clause_text)

        # Frequency mismatch: clause stricter than claim.
        if clause_freq and claim_freq:
            order = {"daily": 3, "weekly": 2, "monthly": 1}
            if order.get(clause_freq, 0) > order.get(claim_freq, 0):
                gaps.append(
                    GapFinding(
                        category=claim.category,
                        claim=claim.text,
                        clause=clause.text,
                        source=clause.source,
                        severity="Major",
                        recommendation=f"Update SOP frequency to {clause_freq} to match the requirement.",
                    )
                )
                continue

        # Temperature clause but claim lacks concrete limits.
        if "temperature" in clause_text and "temperature" in claim_text:
            has_number = bool(re.search(r"\b-?\d{1,3}\b", claim_text))
            if not has_number:
                gaps.append(
                    GapFinding(
                        category=claim.category,
                        claim=claim.text,
                        clause=clause.text,
                        source=clause.source,
                        severity="Minor",
                        recommendation="Add explicit temperature limits and monitoring frequency to the SOP.",
                    )
                )

        # Generic: if clause contains "shall"/"must" and claim sounds vague.
        if any(w in clause_text for w in ("shall", "must")) and any(
            w in claim_text for w in ("as needed", "when required", "sometimes")
        ):
            gaps.append(
                GapFinding(
                    category=claim.category,
                    claim=claim.text,
                    clause=clause.text,
                    source=clause.source,
                    severity="Minor",
                    recommendation="Replace vague wording with a measurable procedure (who/what/when/how).",
                )
            )

    return gaps


def analyst_agent_llm_batch(claims: list[OperationalClaim], retrieved: dict[int, list[dict]]) -> list[GapFinding]:
    """LLM-based gap detection grounded ONLY in provided retrieved clauses.

    `retrieved` maps claim index -> list of {text, source, chunk_id, score}.
    """

    if not _has_openai() or not claims:
        return []

    client = _openai_client()
    sys = (
        "You are a compliance analyst. You must ONLY use the provided retrieved clauses as evidence. "
        "If evidence is insufficient, do not create a gap. Return ONLY JSON."
    )
    user = (
        "For each operational claim, review the retrieved clauses. If there is a clear non-compliance or a "
        "measurable gap, output a GapFinding. If not, output none for that claim.\n\n"
        "Severity rules: Critical (immediate safety/legal risk), Major (material noncompliance), Minor (clarity/record gap).\n"
        "Return JSON: {\"gaps\":[{\"category\":...,\"claim\":...,\"clause\":...,\"source\":...,\"severity\":...,\"recommendation\":...}]}\n\n"
        f"CLAIMS_JSON:\n{json.dumps([c.model_dump() for c in claims], ensure_ascii=False)}\n\n"
        f"RETRIEVED_CLAUSES_JSON:\n{json.dumps(retrieved, ensure_ascii=False)}"
    )

    resp = client.chat.completions.create(
        model=_openai_model(),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=1200,
    )

    output_text = resp.choices[0].message.get("content", "")
    data = _json_loads_maybe(output_text)
    raw_gaps = data.get("gaps", []) if isinstance(data, dict) else []
    gaps: list[GapFinding] = []
    for g in raw_gaps:
        try:
            gaps.append(GapFinding(**g))
        except Exception:
            continue
    return gaps


def compute_scorecard(claims: list[OperationalClaim], gaps: list[GapFinding]) -> Scorecard:
    categories = [
        "Hygiene",
        "Processing",
        "Storage/Temperature",
        "Packaging/Labeling",
        "Documentation/Records",
        "Pest Control",
        "Personnel/Training",
        "Other",
    ]

    # Base: 100; subtract by severity and distribute to category buckets.
    penalty = {"Critical": 15.0, "Major": 8.0, "Minor": 3.0}

    by_category = {c: 100.0 for c in categories}
    for gap in gaps:
        by_category[gap.category] = max(0.0, by_category[gap.category] - penalty[gap.severity])

    # If we have no claims for a category, don't pretend it's 100; mark as N/A via 0.
    claim_cats = {c.category for c in claims}
    for c in categories:
        if c not in claim_cats:
            by_category[c] = 0.0

    present_scores = [v for c, v in by_category.items() if c in claim_cats]
    overall = float(sum(present_scores) / max(1, len(present_scores)))
    return Scorecard(overall_percent=round(overall, 1), by_category=by_category)


def run_audit(facility_filename: str, facility_text: str, top_k: int = 3) -> AuditReport:
    claims = auditor_agent_extract_claims(facility_text)

    gaps: list[GapFinding] = []
    if not claims:
        scorecard = Scorecard(overall_percent=0.0, by_category={"Other": 0.0})
        return AuditReport(
            facility_filename=facility_filename,
            created_at=datetime.utcnow(),
            claims=[],
            gaps=[],
            scorecard=scorecard,
            notes=["No auditable operational claims detected in the uploaded document."],
        )

    retrieved_for_llm: dict[int, list[dict]] = {}

    for idx, claim in enumerate(claims):
        clauses = regulatory_agent_retrieve(claim, top_k=top_k)
        if not clauses:
            continue
        retrieved_for_llm[idx] = [c.model_dump() for c in clauses]

        # Deterministic heuristic gaps (always available)
        gaps.extend(analyst_agent_compare(claim, clauses))

    # If LLM is available, prefer the grounded LLM batch gaps (more accurate).
    if _has_openai() and retrieved_for_llm:
        try:
            llm_gaps = analyst_agent_llm_batch(claims[:20], retrieved_for_llm)
            if llm_gaps:
                gaps = llm_gaps
        except Exception:
            pass

    scorecard = compute_scorecard(claims, gaps)
    notes: list[str] = []
    if not _has_openai():
        notes.append(
            "Running in offline demo mode (no `OPENAI_API_KEY`). Findings are heuristic and intended for scaffolding."
        )
    else:
        notes.append("LLM mode enabled: outputs are constrained to retrieved clauses.")

    report = AuditReport(
        facility_filename=facility_filename,
        created_at=datetime.utcnow(),
        claims=claims,
        gaps=gaps,
        scorecard=scorecard,
        notes=notes,
    )

    try:
        audit_db.save_report(report, top_k=top_k)
    except Exception:
        pass

    return report
