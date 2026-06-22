from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import AuditReport


DEFAULT_DB_PATH = Path("audits.sqlite3")


class AuditDatabase:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                facility_filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                claim_count INTEGER NOT NULL,
                gap_count INTEGER NOT NULL,
                overall_percent REAL NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def save_report(self, report: AuditReport, top_k: int = 3) -> int:
        payload = json.dumps(report.model_dump(), ensure_ascii=False)
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_reports (
                facility_filename, created_at, top_k, claim_count, gap_count, overall_percent, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.facility_filename,
                report.created_at.isoformat(),
                top_k,
                len(report.claims),
                len(report.gaps),
                report.scorecard.overall_percent,
                payload,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        rows = cursor.execute(
            """
            SELECT id, facility_filename, created_at, top_k, claim_count, gap_count, overall_percent
            FROM audit_reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "id": row[0],
                "facility_filename": row[1],
                "created_at": row[2],
                "top_k": row[3],
                "claim_count": row[4],
                "gap_count": row[5],
                "overall_percent": row[6],
            }
            for row in rows
        ]
