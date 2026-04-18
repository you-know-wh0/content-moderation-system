from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ReviewStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS moderation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decision_source TEXT NOT NULL,
                    toxicity_score REAL NOT NULL,
                    ai_confidence REAL NOT NULL,
                    requires_human_review INTEGER NOT NULL,
                    matched_rule TEXT,
                    non_toxic_score REAL NOT NULL,
                    toxic_score REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    review_status TEXT NOT NULL,
                    reviewer_decision TEXT,
                    reviewer_name TEXT,
                    reviewer_notes TEXT,
                    reviewed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_moderation_events_review_status
                ON moderation_events (review_status, created_at DESC)
                """
            )

    def record_event(self, payload: dict[str, Any]) -> int:
        label_scores = payload["label_scores"]
        review_status = "PENDING" if payload["requires_human_review"] else "NOT_REQUIRED"

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO moderation_events (
                    text,
                    status,
                    decision_source,
                    toxicity_score,
                    ai_confidence,
                    requires_human_review,
                    matched_rule,
                    non_toxic_score,
                    toxic_score,
                    latency_ms,
                    review_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["text"],
                    payload["status"],
                    payload["decision_source"],
                    payload["toxicity_score"],
                    payload["ai_confidence"],
                    int(payload["requires_human_review"]),
                    payload["matched_rule"],
                    label_scores["non_toxic"],
                    label_scores["toxic"],
                    payload["latency_ms"],
                    review_status,
                ),
            )
            return int(cursor.lastrowid)

    def pending_reviews(self, limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT *
                FROM moderation_events
                WHERE review_status = 'PENDING'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def submit_review(
        self,
        review_id: int,
        *,
        decision: str,
        reviewer_name: str | None,
        reviewer_notes: str | None,
    ) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                UPDATE moderation_events
                SET review_status = 'RESOLVED',
                    reviewer_decision = ?,
                    reviewer_name = ?,
                    reviewer_notes = ?,
                    reviewed_at = CURRENT_TIMESTAMP
                WHERE id = ? AND review_status = 'PENDING'
                """,
                (decision, reviewer_name, reviewer_notes, review_id),
            )
            row = connection.execute(
                "SELECT * FROM moderation_events WHERE id = ?",
                (review_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def analytics(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_events,
                    AVG(latency_ms) AS average_latency_ms,
                    SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) AS approved_count,
                    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) AS rejected_count,
                    SUM(CASE WHEN status = 'HUMAN_REVIEW' THEN 1 ELSE 0 END) AS human_review_count,
                    SUM(CASE WHEN review_status = 'PENDING' THEN 1 ELSE 0 END) AS pending_reviews,
                    SUM(CASE WHEN review_status = 'RESOLVED' THEN 1 ELSE 0 END) AS reviewed_items,
                    SUM(CASE WHEN decision_source = 'blacklist' THEN 1 ELSE 0 END) AS blacklist_count,
                    SUM(CASE WHEN decision_source = 'whitelist' THEN 1 ELSE 0 END) AS whitelist_count,
                    SUM(CASE WHEN decision_source = 'model' THEN 1 ELSE 0 END) AS model_count
                FROM moderation_events
                """
            ).fetchone()

        total_events = int(totals["total_events"] or 0)
        human_review_count = int(totals["human_review_count"] or 0)
        return {
            "total_events": total_events,
            "average_latency_ms": round(float(totals["average_latency_ms"] or 0.0), 3),
            "human_review_rate": round(human_review_count / total_events, 4)
            if total_events
            else 0.0,
            "pending_reviews": int(totals["pending_reviews"] or 0),
            "reviewed_items": int(totals["reviewed_items"] or 0),
            "status_counts": {
                "APPROVED": int(totals["approved_count"] or 0),
                "REJECTED": int(totals["rejected_count"] or 0),
                "HUMAN_REVIEW": human_review_count,
            },
            "decision_source_counts": {
                "blacklist": int(totals["blacklist_count"] or 0),
                "whitelist": int(totals["whitelist_count"] or 0),
                "model": int(totals["model_count"] or 0),
            },
        }
