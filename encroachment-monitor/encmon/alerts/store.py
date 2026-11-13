"""Alert store, audit log, RBAC, and escalation (paper Sec. Alerts/Deployment).

SQLite-backed persistence for the application layer: detections become
alerts, alerts follow the tiered protocol (NEW -> acknowledged / dismissed /
confirmed -> ESCALATED when unacknowledged past the SLA), every access and
acknowledgement lands in an audit log, and role-based access control
restricts actions per role.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

# Roles (least privilege first)
ROLES = ("field_officer", "senior_official", "admin")
ACTION_PERMISSIONS = {
    "view": {"field_officer", "senior_official", "admin"},
    "acknowledge": {"field_officer", "senior_official", "admin"},
    "confirm": {"field_officer", "senior_official", "admin"},
    "dismiss": {"field_officer", "senior_official", "admin"},
    "escalate": {"senior_official", "admin"},
    "view_audit": {"admin"},
    "manage_users": {"admin"},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    pw_hash TEXT NOT NULL,
    role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parcels (
    id TEXT PRIMARY KEY,
    name TEXT, protected INTEGER, ring_json TEXT, area_m2 REAL
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id TEXT UNIQUE,
    parcel_id TEXT,
    lat REAL, lon REAL,
    x_m REAL, y_m REAL,
    ndbi_t1 REAL, ndbi_t2 REAL, d_ndbi REAL,
    area_m2 REAL,
    severity TEXT,
    is_true_positive INTEGER,
    detected_at TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER UNIQUE REFERENCES detections(id),
    status TEXT NOT NULL DEFAULT 'NEW',
    severity TEXT,
    assigned_role TEXT DEFAULT 'field_officer',
    created_at TEXT,
    acknowledged_at TEXT,
    acknowledged_by TEXT,
    closed_at TEXT,
    closed_by TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    user TEXT,
    role TEXT,
    action TEXT,
    detail TEXT
);
"""


def _hash_pw(pw: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000)
    return salt.hex() + "$" + dk.hex()


def _verify_pw(pw: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split("$")
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 100_000)
    return hmac.compare_digest(dk.hex(), dk_hex)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


class AlertStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        if not self.list_users():
            self._seed_users()

    def _seed_users(self):
        demo = [("officer", "officer123", "field_officer"),
                ("official", "official123", "senior_official"),
                ("admin", "admin123", "admin")]
        for u, pw, r in demo:
            self.add_user(u, pw, r)

    # ------------------------------------------------------------------
    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- users / RBAC ----------------------------------------------------
    def add_user(self, username: str, password: str, role: str):
        assert role in ROLES
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO users(username, pw_hash, role) VALUES (?,?,?)",
                (username, _hash_pw(password), role),
            )

    def list_users(self):
        return self._conn.execute("SELECT username, role FROM users").fetchall()

    def authenticate(self, username: str, password: str):
        row = self._conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        if row and _verify_pw(password, row["pw_hash"]):
            return dict(username=row["username"], role=row["role"])
        return None

    def allowed(self, role: str, action: str) -> bool:
        return role in ACTION_PERMISSIONS.get(action, set())

    # -- audit -------------------------------------------------------------
    def audit(self, user: str, role: str, action: str, detail: str = ""):
        with self._tx() as c:
            c.execute(
                "INSERT INTO audit_log(ts, user, role, action, detail) VALUES (?,?,?,?,?)",
                (_now(), user, role, action, detail),
            )

    def audit_tail(self, limit: int = 200):
        return self._conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # -- parcels -----------------------------------------------------------
    def load_parcels(self, parcels: list[dict]):
        with self._tx() as c:
            for p in parcels:
                c.execute(
                    "INSERT OR REPLACE INTO parcels(id, name, protected, ring_json, area_m2)"
                    " VALUES (?,?,?,?,?)",
                    (p["id"], p["name"], int(p["protected"]),
                     json.dumps(p["ring"]), p["area_m2"]),
                )

    def get_parcels(self):
        rows = self._conn.execute("SELECT * FROM parcels").fetchall()
        return [
            dict(id=r["id"], name=r["name"], protected=bool(r["protected"]),
                 ring=json.loads(r["ring_json"]), area_m2=r["area_m2"])
            for r in rows
        ]

    # -- detections / alerts ----------------------------------------------
    def insert_detection(self, d: dict) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO detections(site_id, parcel_id, lat, lon, x_m, y_m,"
                " ndbi_t1, ndbi_t2, d_ndbi, area_m2, severity, is_true_positive, detected_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (d["site_id"], d.get("parcel_id"), d.get("lat"), d.get("lon"),
                 d.get("x_m"), d.get("y_m"), d.get("ndbi_t1"), d.get("ndbi_t2"),
                 d.get("d_ndbi"), d.get("area_m2"), d.get("severity"),
                 int(d.get("is_true_positive", 0)), d.get("detected_at", _now())),
            )
            row = c.execute("SELECT id FROM detections WHERE site_id=?", (d["site_id"],)).fetchone()
            return int(row["id"])

    def raise_alert(self, detection_id: int, severity: str):
        with self._tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO alerts(detection_id, status, severity, created_at)"
                " VALUES (?, 'NEW', ?, ?)",
                (detection_id, severity, _now()),
            )

    def alerts(self, status: str | None = None):
        q = (
            "SELECT a.*, d.site_id, d.parcel_id, d.lat, d.lon, d.d_ndbi, d.is_true_positive"
            " FROM alerts a JOIN detections d ON d.id = a.detection_id"
        )
        args = ()
        if status:
            q += " WHERE a.status=?"
            args = (status,)
        q += " ORDER BY a.id"
        return self._conn.execute(q, args).fetchall()

    def alert_by_id(self, alert_id: int):
        return self._conn.execute(
            "SELECT * FROM alerts WHERE id=?", (alert_id,)
        ).fetchone()

    # -- alert lifecycle ----------------------------------------------------
    def _assert_allowed(self, role: str, action: str):
        if not self.allowed(role, action):
            raise PermissionError(f"role '{role}' may not perform '{action}'")

    def acknowledge(self, alert_id: int, user: dict):
        self._assert_allowed(user["role"], "acknowledge")
        with self._tx() as c:
            c.execute(
                "UPDATE alerts SET status='ACKNOWLEDGED', acknowledged_at=?, acknowledged_by=?"
                " WHERE id=? AND status='NEW'",
                (_now(), user["username"], alert_id),
            )
        self.audit(user["username"], user["role"], "acknowledge", f"alert {alert_id}")

    def confirm(self, alert_id: int, user: dict):
        self._assert_allowed(user["role"], "confirm")
        with self._tx() as c:
            c.execute(
                "UPDATE alerts SET status='CONFIRMED', closed_at=?, closed_by=? WHERE id=?",
                (_now(), user["username"], alert_id),
            )
        self.audit(user["username"], user["role"], "confirm", f"alert {alert_id}")

    def dismiss(self, alert_id: int, user: dict):
        self._assert_allowed(user["role"], "dismiss")
        with self._tx() as c:
            c.execute(
                "UPDATE alerts SET status='DISMISSED', closed_at=?, closed_by=? WHERE id=?",
                (_now(), user["username"], alert_id),
            )
        self.audit(user["username"], user["role"], "dismiss", f"alert {alert_id}")

    def escalate(self, alert_id: int, user: dict | None = None, reason: str = ""):
        user = user or dict(username="system", role="admin")
        self._assert_allowed(user["role"], "escalate")
        with self._tx() as c:
            c.execute(
                "UPDATE alerts SET status='ESCALATED', acknowledged_by=COALESCE(acknowledged_by, ?)"
                " WHERE id=? AND status IN ('NEW','ACKNOWLEDGED')",
                (user["username"], alert_id),
            )
        self.audit(user["username"], user["role"], "escalate",
                   f"alert {alert_id} {reason}".strip())

    def run_escalation(self, sla_hours: float, now_ts: float | None = None):
        """Escalate alerts unacknowledged past the SLA; returns escalated ids.

        The SLA window uses each alert's created_at; 'now_ts' (epoch seconds)
        is injectable for tests.
        """
        now_ts = now_ts if now_ts is not None else time.time()
        escalated = []
        for row in self.alerts():
            if row["status"] not in ("NEW", "ACKNOWLEDGED"):
                continue
            created = time.mktime(time.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
            if now_ts - created >= sla_hours * 3600:
                self.escalate(row["id"], reason=f"SLA {sla_hours:.0f}h exceeded")
                escalated.append(row["id"])
        return escalated

    # -- reporting ----------------------------------------------------------
    def report_summary(self) -> dict:
        det = self._conn.execute(
            "SELECT COUNT(*) n, SUM(is_true_positive) tp FROM detections"
        ).fetchone()
        st = self._conn.execute(
            "SELECT status, COUNT(*) n FROM alerts GROUP BY status"
        ).fetchall()
        return dict(
            detections=det["n"],
            true_positives=int(det["tp"] or 0),
            by_status={r["status"]: r["n"] for r in st},
        )
