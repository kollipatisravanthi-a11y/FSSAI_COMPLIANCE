from __future__ import annotations

import sys
from pathlib import Path

# Allow running as: `python .\scripts\smoke_test_pdf.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fssai_copilot.orchestrator import run_audit
from fssai_copilot.reporting import render_pdf


def main() -> None:
    facility_text = (
        "Production equipment shall be cleaned daily after operations.\n"
        "Floors shall be cleaned at the end of each shift.\n"
        "Pest inspection shall be conducted monthly.\n"
    ) + ("X" * 400)  # long-token edge case

    report = run_audit("demo.txt", facility_text, top_k=3)
    pdf_bytes = render_pdf(report)
    print("PDF bytes:", len(pdf_bytes))


if __name__ == "__main__":
    main()
