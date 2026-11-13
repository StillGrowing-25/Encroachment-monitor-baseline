"""Fast unit tests for the framework's core pieces (no full training)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from encmon.config import CLASSES, NDBI_DELTA, NDBI_TAU
from encmon.detection.change import ChangeConfig, detect_raster, detect_sites
from encmon.detection.gis import ProtectedLayer
from encmon.datagen.parcels import generate_parcels, point_in_ring
from encmon.datagen.patches import generate_patches, _sample_composition
from encmon.datagen.sites import generate_sites
from encmon.evaluation.metrics import binary_counts, prf_iou, confusion_matrix, per_class_prf
from encmon.models.tree import DecisionTree
from encmon.models.forest import RandomForest
from encmon.models.mlp import MLP
from encmon.spectra import ndbi, ndvi, ndwi, patch_features, FEATURE_NAMES
from encmon.preprocess import pipeline as pre


# ---------------------------------------------------------------------------
# spectra / features
# ---------------------------------------------------------------------------
def test_indices_values_and_bounds():
    bands = np.zeros((8, 8, 5))
    bands[..., 1] = 0.10   # green
    bands[..., 3] = 0.40   # nir  -> NDWI strongly negative
    assert ndwi(bands).mean() < -0.5
    bands2 = np.zeros((8, 8, 5))
    bands2[..., 1] = 0.40
    bands2[..., 3] = 0.10
    assert ndwi(bands2).mean() > 0.5
    assert np.all(ndvi(bands2) <= 1.0 + 1e-6)
    assert np.all(ndbi(bands2) >= -1.0 - 1e-6)


def test_feature_vector_shape_and_names():
    rng = np.random.default_rng(0)
    bands = rng.random((10, 10, 5))
    f = patch_features(bands)
    assert f.shape == (18,)
    assert len(FEATURE_NAMES) == 18
    assert FEATURE_NAMES[14] == "ndbi_mean"
    assert FEATURE_NAMES[16] == "texture_mean"


# ---------------------------------------------------------------------------
# models (from scratch)
# ---------------------------------------------------------------------------
def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = ((X[:, 0] + X[:, 1] > 0.8) | (X[:, 2] - X[:, 3] > 1.2)).astype(int)
    return X, y


def test_decision_tree_learns_toy_rule():
    """An OR of two halfspaces; greedy axis-aligned trees reach ~0.85 here
    (verified against sklearn on the same data)."""
    X, y = _toy()
    tr = DecisionTree(max_depth=8, random_state=0).fit(X[:300], y[:300])
    acc = (tr.predict(X[300:]) == y[300:]).mean()
    assert acc > 0.78


def test_random_forest_beats_single_tree_noise():
    rng = np.random.default_rng(1)
    X, y = _toy(600, seed=1)
    X += rng.normal(0, 0.8, X.shape)          # label noise via features
    tr, te = slice(400), slice(400, 600)
    rf = RandomForest(n_trees=60, random_state=0).fit(X[tr], y[tr])
    dt = DecisionTree(max_depth=10, random_state=0).fit(X[tr], y[tr])
    rf_acc = (rf.predict(X[te]) == y[te]).mean()
    dt_acc = (dt.predict(X[te]) == y[te]).mean()
    assert rf_acc > 0.78
    assert rf_acc >= dt_acc - 0.05


def test_mlp_converges_on_separable_data():
    X, y = _toy(500, seed=2)
    m = MLP(hidden=(16,), epochs=40, lr=5e-3, batch=64, random_state=0)
    m.fit(X[:400], y[:400])
    assert (m.predict(X[400:]) == y[400:]).mean() > 0.9


def test_rf_landcover_directionality():
    """Sanity: on real generated patches, nonbuilt classes separate ~perfectly."""
    X, y, _ = generate_patches(1500, seed=3)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    rf = RandomForest(n_trees=40, random_state=0).fit(X[idx[:1000]], y[idx[:1000]])
    pred = rf.predict(X[idx[1000:1200]])
    acc = (pred == y[idx[1000:1200]]).mean()
    assert acc > 0.80
    # nonbuilt vs built boundary is nearly perfect (paper's premise)
    yb_true = np.isin(y[idx[1000:1200]], [3, 4])
    yb_pred = np.isin(pred, [3, 4])
    counts = binary_counts(yb_true, yb_pred)
    assert counts["tp"] == yb_true.sum()          # perfect built recall
    assert counts["fp"] == 0                      # no nonbuilt flagged built


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------
def test_change_rule_truth_table():
    cfg = ChangeConfig(tau=0.13, delta=0.045)
    t1 = np.array([0.08, 0.16, 0.08, 0.05])
    t2 = np.array([0.20, 0.20, 0.10, 0.20])
    expected = np.array([True, False, False, True])   # delta fails in [2]
    assert (detect_raster(t1, t2, cfg) == expected).all()


def test_site_generation_counts_and_change():
    sites, parcels, aoi = generate_sites(42)
    assert len(sites) == 6000
    assert sum(s["encroachment"] for s in sites) == 567
    det = detect_sites(sites)
    n_flash = sum(1 for s in det if s.get("flash"))
    assert n_flash == 84
    assert all(s["change"] for s in det if s.get("flash"))   # every flash fires
    assert not any(s["change"] and s["weak"] for s in det)    # weak site missed


def test_gis_overlay_matches_parcels():
    sites, parcels, _ = generate_sites(42)
    layer = ProtectedLayer(parcels)
    rng = np.random.default_rng(0)
    for _ in range(60):
        p = parcels[rng.integers(len(parcels))]
        x0, y0 = p["ring"][0]
        inside = point_in_ring(x0 + 1.0, y0 + 1.0, p["ring"])
        got, pid = layer.contains_point(x0 + 1.0, y0 + 1.0)
        assert got == (inside and p["protected"])
        if got:
            assert pid == p["id"]


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def test_radiometric_round_trip():
    rng = np.random.default_rng(0)
    sr = rng.random((16, 16, 5))
    assert np.abs(pre.to_surface_reflectance(pre.to_toa(sr)) - sr).max() < 1e-9


def test_registration_finds_shift():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 5))
    img = np.kron(img, np.ones((4, 4, 1)))[:64, :64]     # coarse structure
    from encmon.preprocess.pipeline import shift_image, estimate_shift

    moved = shift_image(img, 2.0, -1.0)
    dy, dx = estimate_shift(img, moved)
    assert abs(dy - 2.0) <= 0.6 and abs(dx + 1.0) <= 0.6


def test_cloud_mask_and_augmentation():
    rng = np.random.default_rng(1)
    label = rng.integers(0, 4, (32, 32))
    scl = pre.simulate_scl(label, cloud_frac=0.1, rng=rng)
    cm = pre.cloud_mask(scl)
    assert 0.05 < cm.mean() < 0.2
    scene = rng.random((10, 10, 5))
    masked, valid = pre.apply_cloud_mask(scene, scl[:10, :10])
    assert (masked[~valid] == 0).all()
    augs = [pre.augment_patch(scene, rng) for _ in range(6)]
    assert any(not np.array_equal(a, scene) for a in augs)


# ---------------------------------------------------------------------------
# metrics + alerts store
# ---------------------------------------------------------------------------
def test_metrics_exact():
    counts = dict(tp=566, fp=58, fn=1, tn=5375)
    m = prf_iou(counts)
    assert abs(m["precision"] - 0.9071) < 1e-3
    assert abs(m["recall"] - 0.9982) < 1e-3
    assert abs(m["f1"] - 0.9505) < 1e-3
    assert abs(m["iou"] - 0.9056) < 1e-3
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], 2)
    pc = per_class_prf(cm)
    assert pc[0]["recall"] == 0.5 and pc[1]["recall"] == 1.0


def test_alert_store_lifecycle_and_rbac(tmp_path):
    from encmon.alerts.store import AlertStore

    store = AlertStore(str(tmp_path / "alerts.db"))
    user = store.authenticate("officer", "officer123")
    assert user and user["role"] == "field_officer"
    assert store.authenticate("officer", "wrong") is None
    assert not store.allowed(user["role"], "view_audit")

    det_id = store.insert_detection(dict(
        site_id="T-1", parcel_id="P-0001", lat=11.0, lon=77.0,
        ndbi_t1=0.05, ndbi_t2=0.2, d_ndbi=0.15, area_m2=1200.0,
        severity="high", is_true_positive=True))
    store.raise_alert(det_id, "high")
    alert = store.alerts()[0]
    assert alert["status"] == "NEW"

    # a field officer may not escalate
    with pytest.raises(PermissionError):
        store.escalate(alert["id"], user)
    store.acknowledge(alert["id"], user)
    assert store.alerts()[0]["status"] == "ACKNOWLEDGED"
    official = dict(username="official", role="senior_official")
    store.escalate(alert["id"], official, reason="test")
    assert store.alerts()[0]["status"] == "ESCALATED"
    assert any(r["action"] == "escalate" for r in store.audit_tail(10))

    # SLA sweep escalates stale NEW alerts
    det2 = store.insert_detection(dict(site_id="T-2", lat=11.0, lon=77.0,
                                       ndbi_t1=0.05, ndbi_t2=0.2, d_ndbi=0.15,
                                       severity="medium"))
    store.raise_alert(det2, "medium")
    import time as _t
    old = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(_t.time() - 80 * 3600))
    store._conn.execute("UPDATE alerts SET created_at=? WHERE detection_id=?",
                        (old, det2))
    store._conn.commit()
    escalated = store.run_escalation(sla_hours=72.0)
    assert len(escalated) == 1


# ---------------------------------------------------------------------------
# scalability
# ---------------------------------------------------------------------------
def test_scalability_table():
    from encmon.evaluation.scalability import scalability_table

    rows = scalability_table(sec_per_km2=0.0275, workers=64)
    assert [r.scope for r in rows] == ["District", "State", "Nation"]
    assert rows[0].tiles == 459            # 3000 / 6.55 -> ceil
    assert rows[1].tiles == 19848
    assert rows[2].tiles == 501833
