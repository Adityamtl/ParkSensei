"""
incident_db.py — SQLite incident log and after-action feedback store.
Adapted from ARES (alt2) database pattern. Lightweight, no ORM.
"""

import sqlite3, json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "incidents.db"


def _init():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            zone            TEXT NOT NULL,
            scenario        TEXT NOT NULL,
            risk_score      INTEGER NOT NULL,
            risk_level      TEXT NOT NULL,
            latitude        REAL,
            longitude       REAL,
            officers_req    INTEGER DEFAULT 0,
            vehicles_req    INTEGER DEFAULT 0,
            hospital        TEXT,
            notes           TEXT,
            status          TEXT DEFAULT 'ACTIVE'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incident_feedback (
            incident_id         TEXT PRIMARY KEY,
            submitted_at        TEXT NOT NULL,
            actual_officers     INTEGER NOT NULL,
            actual_barricades   INTEGER NOT NULL,
            road_closure_used   INTEGER NOT NULL DEFAULT 0,
            actual_duration_min INTEGER,
            resolution_notes    TEXT,
            FOREIGN KEY (incident_id) REFERENCES incidents (id)
        )
    """)
    conn.commit()
    conn.close()


_init()


def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def save_incident(inc: dict) -> str:
    """Save a new incident and return its ID."""
    inc_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    c = _conn()
    c.execute("""
        INSERT OR REPLACE INTO incidents
        (id, created_at, zone, scenario, risk_score, risk_level,
         latitude, longitude, officers_req, vehicles_req, hospital, notes, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        inc_id, datetime.now().isoformat(),
        inc.get("zone", ""), inc.get("scenario", ""),
        inc.get("risk_score", 0), inc.get("risk_level", ""),
        inc.get("latitude"), inc.get("longitude"),
        inc.get("officers_req", 0), inc.get("vehicles_req", 0),
        inc.get("hospital", ""), inc.get("notes", ""),
        "ACTIVE"
    ))
    c.commit()
    c.close()
    return inc_id


def get_all_incidents() -> list:
    c = _conn()
    rows = c.execute("SELECT * FROM incidents ORDER BY created_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_active_incidents() -> list:
    c = _conn()
    rows = c.execute(
        "SELECT * FROM incidents WHERE status = 'ACTIVE' ORDER BY created_at DESC"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def save_feedback(fb: dict):
    """Save after-action feedback and mark incident as RESOLVED."""
    c = _conn()
    try:
        with c:
            c.execute("""
                INSERT OR REPLACE INTO incident_feedback
                (incident_id, submitted_at, actual_officers, actual_barricades,
                 road_closure_used, actual_duration_min, resolution_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                fb["incident_id"], datetime.now().isoformat(),
                fb.get("actual_officers", 0), fb.get("actual_barricades", 0),
                fb.get("road_closure_used", 0), fb.get("actual_duration_min"),
                fb.get("resolution_notes", "")
            ))
            c.execute(
                "UPDATE incidents SET status = 'RESOLVED' WHERE id = ?",
                (fb["incident_id"],)
            )
    finally:
        c.close()


def get_feedback(incident_id: str) -> dict | None:
    c = _conn()
    row = c.execute(
        "SELECT * FROM incident_feedback WHERE incident_id = ?",
        (incident_id,)
    ).fetchone()
    c.close()
    return dict(row) if row else None


def get_incidents_with_feedback() -> list:
    """Return all resolved incidents with their feedback joined."""
    c = _conn()
    rows = c.execute("""
        SELECT i.*, f.actual_officers, f.actual_barricades,
               f.road_closure_used, f.actual_duration_min, f.resolution_notes
        FROM incidents i
        LEFT JOIN incident_feedback f ON i.id = f.incident_id
        WHERE i.status = 'RESOLVED'
        ORDER BY i.created_at DESC
    """).fetchall()
    c.close()
    return [dict(r) for r in rows]


def count_incidents() -> dict:
    c = _conn()
    total = c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    active = c.execute("SELECT COUNT(*) FROM incidents WHERE status='ACTIVE'").fetchone()[0]
    resolved = c.execute("SELECT COUNT(*) FROM incidents WHERE status='RESOLVED'").fetchone()[0]
    c.close()
    return {"total": total, "active": active, "resolved": resolved}
