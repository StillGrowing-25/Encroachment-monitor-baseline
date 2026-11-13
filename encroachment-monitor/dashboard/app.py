"""Web dashboard -- the framework's application layer (paper Table I).

GIS map of detections on protected parcels, the tiered alert queue with
acknowledge / confirm / dismiss actions, automatic escalation of
unacknowledged alerts past the SLA, HTML/CSV reports, an audit log, and
role-based access control (field officer / senior official / admin).

Run:  python dashboard/app.py   ->  http://127.0.0.1:5000
Demo logins (seeded): officer/officer123, official/official123, admin/admin123
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys

# make the package importable when launched from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import (  # noqa: E402
    Flask,
    abort,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from encmon.alerts.store import ACTION_PERMISSIONS, AlertStore  # noqa: E402
from encmon.config import AOI_CENTER_LATLON, DASHBOARD_DB, ESCALATION_SLA_HOURS  # noqa: E402

app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))
app.secret_key = os.environ.get("DASHBOARD_SECRET", "encroachment-monitor-demo-key")

store = AlertStore(os.environ.get("DASHBOARD_DB", DASHBOARD_DB))

_PAGE_PREFIXES = ("/map", "/alerts", "/report", "/audit", "/")


@app.before_request
def _login_gate():
    """Pages redirect to login; API endpoints answer 401 via require()."""
    if request.path in ("/login", "/health") or request.path.startswith("/static"):
        return None
    if current_user() is None:
        for pref in _PAGE_PREFIXES:
            if request.path == pref or (pref != "/" and request.path.startswith(pref + "/")):
                return redirect(url_for("login"))
    return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def current_user() -> dict | None:
    return session.get("user")


def require(action: str):
    user = current_user()
    if not user:
        abort(401)
    if not store.allowed(user["role"], action):
        abort(403)
    return user


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user = store.authenticate(request.form.get("username", ""),
                                  request.form.get("password", ""))
        if user:
            session["user"] = user
            store.audit(user["username"], user["role"], "login")
            return redirect(url_for("map_view"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    user = current_user()
    if user:
        store.audit(user["username"], user["role"], "logout")
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------
def _parcel_features():
    feats = []
    for p in store.get_parcels():
        ring = [[x, y] for x, y in p["ring"]]
        feats.append({
            "type": "Feature",
            "properties": {"id": p["id"], "protected": p["protected"], "area_m2": p["area_m2"]},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return feats


@app.route("/api/parcels.geojson")
def parcels_geojson():
    require("view")
    return {"type": "FeatureCollection", "features": _parcel_features()}


@app.route("/api/detections.geojson")
def detections_geojson():
    require("view")
    rows = store._conn.execute(
        "SELECT site_id, parcel_id, lat, lon, d_ndbi, severity, is_true_positive"
        " FROM detections WHERE parcel_id IS NOT NULL"
    ).fetchall()
    feats = [{
        "type": "Feature",
        "properties": {
            "site_id": r["site_id"], "parcel_id": r["parcel_id"],
            "d_ndbi": r["d_ndbi"], "severity": r["severity"],
            "is_true_positive": bool(r["is_true_positive"]),
        },
        "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
    } for r in rows]
    return {"type": "FeatureCollection", "features": feats}


@app.route("/")
@app.route("/map")
def map_view():
    require("view")
    store.audit(current_user()["username"], current_user()["role"], "view_map")
    return render_template(
        "map.html", user=current_user(), center=AOI_CENTER_LATLON,
        n_open=store.alerts("NEW").__len__(),
    )


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------
@app.route("/alerts")
def alerts_view():
    user = require("view")
    # the periodic escalation job runs on every queue inspection
    escalated = store.run_escalation(ESCALATION_SLA_HOURS)
    status = request.args.get("status") or None
    rows = store.alerts(status)
    return render_template(
        "alerts.html", user=user, alerts=rows, status=status,
        escalated_ids=escalated, sla_hours=ESCALATION_SLA_HOURS,
        can=lambda a: store.allowed(user["role"], a),
    )


def _alert_action(action: str):
    user = require(action)
    alert_id = int(request.form.get("alert_id", request.view_args.get("alert_id", 0)))
    if not store.alert_by_id(alert_id):
        abort(404)
    getattr(store, action)(alert_id, user)
    if request.accept_mimetypes.best == "application/json" or request.is_json:
        return {"ok": True, "alert_id": alert_id, "status": action}
    return redirect(url_for("alerts_view"))


@app.route("/alerts/<int:alert_id>/ack", methods=["POST"])
def alert_ack(alert_id):
    return _alert_action("acknowledge")


@app.route("/alerts/<int:alert_id>/confirm", methods=["POST"])
def alert_confirm(alert_id):
    return _alert_action("confirm")


@app.route("/alerts/<int:alert_id>/dismiss", methods=["POST"])
def alert_dismiss(alert_id):
    return _alert_action("dismiss")


@app.route("/alerts/<int:alert_id>/escalate", methods=["POST"])
def alert_escalate(alert_id):
    return _alert_action("escalate")


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@app.route("/report")
def report_view():
    user = require("view")
    store.audit(user["username"], user["role"], "view_report")
    summary = store.report_summary()
    by_sev = store._conn.execute(
        "SELECT severity, COUNT(*) n FROM detections GROUP BY severity"
    ).fetchall()
    return render_template("report.html", user=user, summary=summary,
                           by_severity=by_sev, sla_hours=ESCALATION_SLA_HOURS)


@app.route("/report.csv")
def report_csv():
    require("view")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["alert_id", "site_id", "parcel_id", "lat", "lon", "severity",
                "status", "is_true_positive", "created_at"])
    for r in store.alerts():
        w.writerow([r["id"], r["site_id"], r["parcel_id"], r["lat"], r["lon"],
                    r["severity"], r["status"], r["is_true_positive"], r["created_at"]])
    out = make_response(buf.getvalue())
    out.headers["Content-Disposition"] = "attachment; filename=encroachment_report.csv"
    out.headers["Content-Type"] = "text/csv"
    return out


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
@app.route("/audit")
def audit_view():
    user = require("view_audit")
    rows = store.audit_tail(500)
    return render_template("audit.html", user=user, rows=rows)


@app.route("/health")
def health():
    return {"status": "ok", "alerts": store.report_summary()}


if __name__ == "__main__":
    print("Dashboard on http://127.0.0.1:5000  (logins: officer/officer123, "
          "official/official123, admin/admin123)")
    app.run(host="127.0.0.1", port=5000, debug=False)
