"""Dashboard integration tests via the Flask test client (isolated temp DB)."""
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def dash(tmp_path_factory):
    db = tmp_path_factory.mktemp("dash") / "alerts.db"
    os.environ["DASHBOARD_DB"] = str(db)
    # re-import (or first-import) the dashboard with the temp DB
    try:
        del sys.modules["dashboard_app_mod"]
    except KeyError:
        pass
    spec = importlib.util.spec_from_file_location(
        "dashboard_app_mod",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "dashboard", "app.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # minimal seed: parcel + detection + alert
    mod.store.load_parcels([dict(
        id="P-TEST", name="Test parcel", protected=True,
        ring=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], area_m2=10000.0,
    )])
    det = mod.store.insert_detection(dict(
        site_id="T-1", parcel_id="P-TEST", lat=11.0168, lon=76.9558,
        x_m=50.0, y_m=50.0, ndbi_t1=0.05, ndbi_t2=0.20, d_ndbi=0.15,
        area_m2=1200.0, severity="high", is_true_positive=True))
    mod.store.raise_alert(det, "high")
    return mod


def test_login_required(dash):
    client = dash.app.test_client()
    rv = client.get("/map")
    assert rv.status_code == 302            # page -> redirected to login
    rv = client.get("/api/detections.geojson")
    assert rv.status_code == 401            # API -> 401


def test_bad_login_rejected(dash):
    client = dash.app.test_client()
    rv = client.post("/login", data={"username": "officer", "password": "nope"})
    assert b"Invalid credentials" in rv.data


def test_officer_workflow(dash):
    client = dash.app.test_client()
    assert client.post("/login", data={"username": "officer", "password": "officer123"}
                       ).status_code == 302
    rv = client.get("/alerts")
    assert rv.status_code == 200 and b"T-1" in rv.data
    # RBAC: officer has no escalate button / forbidden direct escalate
    rv = client.post("/alerts/1/escalate")
    assert rv.status_code == 403
    rv = client.post("/alerts/1/ack")
    assert rv.status_code in (302, 200)
    rv = client.post("/alerts/1/confirm")
    assert rv.status_code in (302, 200)
    alert = dash.store.alerts()[0]
    assert alert["status"] == "CONFIRMED"


def test_official_may_escalate(dash):
    client = dash.app.test_client()
    client.post("/login", data={"username": "official", "password": "official123"})
    det = dash.store.insert_detection(dict(site_id="T-2", lat=11.0, lon=77.0,
                                           ndbi_t1=0.05, ndbi_t2=0.2, d_ndbi=0.1,
                                           severity="medium"))
    dash.store.raise_alert(det, "medium")
    rv = client.post("/alerts/2/escalate")
    assert rv.status_code in (302, 200)
    assert dash.store.alert_by_id(2)["status"] == "ESCALATED"


def test_admin_audit_and_geojson(dash):
    client = dash.app.test_client()
    client.post("/login", data={"username": "admin", "password": "admin123"})
    assert client.get("/audit").status_code == 200
    rv = client.get("/api/parcels.geojson")
    assert rv.status_code == 200 and b"P-TEST" in rv.data
    rv = client.get("/api/detections.geojson")
    assert b"T-1" in rv.data
    assert client.get("/report").status_code == 200
    rv = client.get("/report.csv")
    assert rv.status_code == 200 and b"T-1" in rv.data
    rv = client.get("/health")
    assert rv.status_code == 200
