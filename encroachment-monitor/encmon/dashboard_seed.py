"""Seed the dashboard's application layer from pipeline detections.

Loads the parcel cadastre, inserts the full-pipeline detections as
alerts on protected parcels, backdates a handful of alerts past the SLA so
the escalation workflow is demonstrable, and exports GeoJSON products for
the map view and downstream GIS (QGIS/GeoServer).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .alerts.store import AlertStore
from .config import DASHBOARD_DB, DATA_DIR, ESCALATION_SLA_HOURS
from .detection.gis import ProtectedLayer
from .spectra import ndbi


def _severity(d_ndbi: float) -> str:
    if d_ndbi > 0.15:
        return "high"
    if d_ndbi > 0.09:
        return "medium"
    return "low"


def _built_area_m2(t2: np.ndarray, px_m: float = 10.0) -> float:
    """Built footprint estimate: pixels whose own NDBI sits in built range."""
    n = int((ndbi(t2) > 0.15).sum())
    return n * px_m * px_m


def export_geojsons(sites: list[dict], parcels: list[dict]) -> tuple[str, str]:
    layer = ProtectedLayer(parcels)
    p_feats = []
    for p in parcels:
        p_feats.append({
            "type": "Feature",
            "properties": {"id": p["id"], "protected": p["protected"],
                           "area_m2": p["area_m2"]},
            "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in p["ring"]]]},
        })
    d_feats = []
    for s in sites:
        if not s.get("flag_full"):
            continue
        d_feats.append({
            "type": "Feature",
            "properties": {
                "site_id": s["site_id"], "parcel_id": s.get("parcel_id"),
                "d_ndbi": round(s["d_ndbi"], 4),
                "severity": _severity(s["d_ndbi"]),
                "is_true_positive": bool(s["encroachment"]),
            },
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
        })
    parcels_path = os.path.join(DATA_DIR, "parcels.geojson")
    detect_path = os.path.join(DATA_DIR, "detections.geojson")
    for path, fc in ((parcels_path, p_feats), (detect_path, d_feats)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": fc}, f)
    return parcels_path, detect_path


def seed_dashboard(sites: list[dict], parcels: list[dict],
                   db_path: str = DASHBOARD_DB) -> dict:
    store = None
    if os.path.exists(db_path):
        try:
            os.remove(db_path)                  # deterministic reseed
        except PermissionError:
            # file locked (e.g. a running dashboard) -- wipe rows instead
            store = AlertStore(db_path)
            store._conn.executescript(
                "DELETE FROM alerts; DELETE FROM detections;"
                " DELETE FROM parcels; DELETE FROM audit_log; DELETE FROM users;"
            )
            store._conn.commit()
    if store is None:
        store = AlertStore(db_path)
    store.load_parcels(parcels)

    layer = ProtectedLayer(parcels)
    n_det = n_alerts = 0
    for s in sites:
        if not s.get("flag_full"):
            continue
        prot, parcel_id = layer.contains_point(s["x"], s["y"])
        det_id = store.insert_detection({
            "site_id": s["site_id"],
            "parcel_id": parcel_id,
            "lat": s["lat"], "lon": s["lon"], "x_m": s["x"], "y_m": s["y"],
            "ndbi_t1": s["ndbi_t1"], "ndbi_t2": s["ndbi_t2"],
            "d_ndbi": s["d_ndbi"], "area_m2": _built_area_m2(s["t2"]),
            "severity": _severity(s["d_ndbi"]),
            "is_true_positive": bool(s["encroachment"]),
            "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        store.raise_alert(det_id, _severity(s["d_ndbi"]))
        n_det += 1
        n_alerts += 1

    # Backdate a few NEW alerts past the SLA so escalation is demonstrable
    back = store._conn.execute(
        "SELECT a.id FROM alerts a WHERE a.status='NEW' ORDER BY a.id LIMIT 6"
    ).fetchall()
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(time.time() - (ESCALATION_SLA_HOURS + 6) * 3600))
    for r in back:
        store._conn.execute("UPDATE alerts SET created_at=? WHERE id=?",
                            (old, r["id"]))
    store._conn.commit()
    store.audit("system", "admin", "seed",
                f"{n_det} detections, {len(back)} alerts backdated past SLA")

    geo_p, geo_d = export_geojsons(sites, parcels)
    summary = store.report_summary()
    print(f"  dashboard: {n_det} detections -> alerts.db, "
          f"{len(back)} backdated for escalation demo")
    print(f"  geojson: {geo_p}, {geo_d}")
    return dict(summary=summary, parcels=geo_p, detections=geo_d)
