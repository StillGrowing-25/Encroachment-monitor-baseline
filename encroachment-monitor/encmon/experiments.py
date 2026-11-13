"""Experiment stages that reproduce every measured result of the paper.

Each stage returns a results dict and (optionally) writes tables/figures.
Stages are independent so they can be run selectively; run_all.py drives
them in order and renders RESULTS.md.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .config import (
    CLASS_NAMES,
    CLASSES,
    CONF_LEVEL,
    DATA_DIR,
    DEEP_BASELINES,
    MLP_BATCH,
    MLP_EPOCHS,
    MLP_HIDDEN,
    MLP_L2,
    MLP_LR,
    N_PATCHES,
    N_RESAMPLES,
    PARALLEL_WORKERS,
    PATCH,
    RF_BINS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES,
    RF_TREES,
    SCENE_SIZE,
    SEED,
    TABLE_DIR,
    TRAIN_FRACTION,
    TREE_MAX_DEPTH,
)
from .datagen.patches import generate_patches
from .datagen.scene import generate_scene
from .datagen.sites import generate_sites
from .detection.pipeline import EncroachmentPipeline, ablation
from .evaluation.explain import permutation_importance
from .evaluation.metrics import confusion_matrix, mean_std_ci, per_class_prf
from .evaluation.scalability import scalability_table
from .evaluation.segmentation import run_scene_pipeline, segmentation_metrics
from .models.forest import RandomForest
from .models.mlp import MLP
from .models.tree import DecisionTree
from .spectra import FEATURE_NAMES
from .preprocess.pipeline import (
    augment_patch,
    co_register,
    simulate_scl,
    apply_cloud_mask,
    to_surface_reflectance,
    to_toa,
)


def _save_table(name: str, rows: list[dict], columns: list[str]) -> str:
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in columns) + "\n")
    return path


def _save_json(name: str, obj) -> str:
    path = os.path.join(TABLE_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    return path


def _make_rf(seed: int) -> RandomForest:
    return RandomForest(n_trees=RF_TREES, max_depth=RF_MAX_DEPTH,
                        min_samples=RF_MIN_SAMPLES, bins=RF_BINS, random_state=seed)


def _make_mlp(seed: int) -> MLP:
    return MLP(hidden=MLP_HIDDEN, epochs=MLP_EPOCHS, lr=MLP_LR, batch=MLP_BATCH,
               l2=MLP_L2, random_state=seed)


def _make_tree(seed: int) -> DecisionTree:
    return DecisionTree(max_depth=TREE_MAX_DEPTH, random_state=seed)


# ---------------------------------------------------------------------------
# Stage 0: preprocessing demonstration (Table III)
# ---------------------------------------------------------------------------
def stage_preprocessing(seed: int = SEED) -> dict:
    """Demonstrate every Table III step on a real generated scene."""
    rng = np.random.default_rng(seed)
    scene = generate_scene(seed)

    # radiometric: SR -> TOA -> back to SR (round-trip error quantified)
    toa = to_toa(scene["t1"])
    sr_back = to_surface_reflectance(toa)
    rad_err = float(np.abs(sr_back - scene["t1"]).max())

    # geometric: co-register t2 (which carries a sub-pixel shift) onto t1
    t2_reg, (dy, dx) = co_register(scene["t1"], scene["t2"])
    true_dy, true_dx = scene["shift"]
    reg_resid = float(np.hypot(dy - true_dy, dx - true_dx))

    # cloud masking from the scene-classification band
    scl = simulate_scl(scene["label1"], cloud_frac=0.07, rng=rng)
    masked, valid = apply_cloud_mask(scene["t1"], scl)

    # augmentation of one training patch
    X, y, _ = generate_patches(2, seed)
    from .datagen.patches import _render_pixels, _sample_composition

    lab = _sample_composition("illegal", rng)
    bands = _render_pixels(lab, rng)
    aug = [augment_patch(bands, rng) for _ in range(4)]
    aug_differ = any(not np.array_equal(a, bands) for a in aug)

    res = dict(
        radiometric_max_abs_error=rad_err,
        registration_estimated_shift=(float(dy), float(dx)),
        registration_true_shift=(float(true_dy), float(true_dx)),
        registration_residual_px=reg_resid,
        cloud_fraction=float((~valid).mean()),
        augmentation_active=bool(aug_differ),
    )
    _save_json("table3_preprocessing", res)
    print(f"  [preprocess] radiometric round-trip max err {rad_err:.2e}; "
          f"registration residual {reg_resid:.3f} px (<1 px RMSE); "
          f"cloud masked {res['cloud_fraction'] * 100:.1f}%; augmentation active {aug_differ}")
    return res


# ---------------------------------------------------------------------------
# Stage 1: land-cover classification (Table V, Fig. 2)
# ---------------------------------------------------------------------------
# Bump when generators/models change so cached artifacts are invalidated.
LC_CACHE_VERSION = "5"


def _landcover_cache_path(seed: int) -> str:
    return os.path.join(DATA_DIR, f"landcover_seed{seed}_v{LC_CACHE_VERSION}.pkl")


def stage_landcover(seed: int = SEED, verbose: bool = False) -> dict:
    import pickle

    cache_path = _landcover_cache_path(seed)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        X, y = cached["X"], cached["y"]
        rf_first = cached["rf"]
        cm_first = cached["cm"]
        accs = cached["accs"]
        print(f"  [land-cover] loaded cached patches + models ({cache_path})")
    else:
        print(f"  generating {N_PATCHES} patches ...")
        X, y, meta = generate_patches(N_PATCHES, seed)
        n_train = int(TRAIN_FRACTION * len(y))
        n_classes = len(CLASSES)
        models = {
            "Random Forest": _make_rf,
            "MLP": _make_mlp,
            "Decision Tree": _make_tree,
        }
        model_index = {name: i * 7 for i, name in enumerate(models)}
        accs = {name: [] for name in models}
        rf_first = None
        cm_first = None
        order_rng = np.random.default_rng(seed + 99)
        for r in range(N_RESAMPLES):
            idx = order_rng.permutation(len(y))
            tr, te = idx[:n_train], idx[n_train:]
            for name, factory in models.items():
                t0 = time.perf_counter()
                m = factory(seed + r * 10 + model_index[name])
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                accs[name].append(float((pred == y[te]).mean()))
                if name == "Random Forest":
                    print(f"    resample {r + 1}/{N_RESAMPLES}: RF acc "
                          f"{accs[name][-1]:.4f} ({time.perf_counter() - t0:.1f}s)")
                    if r == 0:
                        rf_first = m
                        cm_first = confusion_matrix(y[te], pred, n_classes)
        with open(cache_path, "wb") as f:
            pickle.dump(dict(X=X, y=y, rf=rf_first, cm=cm_first, accs=accs), f,
                        protocol=4)

    n_classes = len(CLASSES)
    pc_first = per_class_prf(cm_first)
    stats = {k: mean_std_ci(v, CONF_LEVEL) for k, v in accs.items()}
    per_class = {
        CLASSES[i]: dict(
            precision=pc_first[i]["precision"],
            recall=pc_first[i]["recall"],
            f1=pc_first[i]["f1"],
        )
        for i in range(n_classes)
    }
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    rows = []
    for cls in CLASSES:
        rows.append(dict(
            class_=CLASS_NAMES[cls],
            precision=round(per_class[cls]["precision"], 2),
            recall=round(per_class[cls]["recall"], 2),
            f1=round(per_class[cls]["f1"], 2),
        ))
    _save_table("table5_landcover", rows, ["class_", "precision", "recall", "f1"])
    summary_rows = [dict(model=k, accuracy_mean=round(v["mean"], 4),
                         accuracy_std=round(v["std"], 4),
                         ci_low=round(v["ci"][0], 4), ci_high=round(v["ci"][1], 4))
                    for k, v in stats.items()]
    _save_table("table5b_model_comparison", summary_rows,
                ["model", "accuracy_mean", "accuracy_std", "ci_low", "ci_high"])

    res = dict(
        accuracy_stats=stats,
        per_class=per_class,
        macro_f1=macro_f1,
        confusion_matrix=cm_first.tolist(),
        model_comparison=summary_rows,
    )
    _save_json("table5_landcover", res)
    rf_acc = stats["Random Forest"]
    print(f"  [land-cover] RF accuracy {rf_acc['mean'] * 100:.1f}% ± {rf_acc['std'] * 100:.1f}% "
          f"(macro-F1 {macro_f1:.3f}); "
          f"legal F1 {per_class['legal']['f1']:.2f}, illegal F1 {per_class['illegal']['f1']:.2f}")
    res["_rf_model"] = rf_first
    res["_X"] = X
    res["_y"] = y
    return res


# ---------------------------------------------------------------------------
# Stage 2: encroachment ablation on 6,000 bi-temporal sites (Table VI)
# ---------------------------------------------------------------------------
def stage_ablation(seed: int = SEED, rf_model=None) -> dict:
    print("  generating 6,000 bi-temporal sites ...")
    sites, parcels, aoi = generate_sites(seed)
    if rf_model is None:
        lc = stage_landcover(seed)
        rf_model = lc["_rf_model"]
    pipe = EncroachmentPipeline(rf_model, parcels)
    t0 = time.perf_counter()
    annotated = pipe.classify_built(sites)
    annotated = pipe.run(annotated)
    print(f"  sites classified + change + GIS overlay in {time.perf_counter() - t0:.1f}s")
    rows = ablation(annotated)

    table_rows = [dict(
        config=r["config"],
        precision=round(r["precision"], 2),
        recall=round(r["recall"], 2),
        f1=round(r["f1"], 2),
        iou=round(r["iou"], 2),
        TP=r["tp"], FP=r["fp"], FN=r["fn"], TN=r["tn"],
    ) for r in rows]
    _save_table("table6_ablation", table_rows,
                ["config", "precision", "recall", "f1", "iou", "TP", "FP", "FN", "TN"])
    full = rows[-1]
    print(f"  [ablation] IoU {rows[0]['iou']:.2f} -> {rows[1]['iou']:.2f} -> {full['iou']:.3f}; "
          f"full: P {full['precision']:.2f} R {full['recall']:.3f} F1 {full['f1']:.2f} "
          f"({full['tp']} TP / {full['fp']} FP / {full['fn']} FN / {full['tn']} TN)")

    # lat/lon for the dashboard
    lat, lon = aoi.to_latlon(np.array([s["x"] for s in annotated]),
                             np.array([s["y"] for s in annotated]))
    for s, la, lo in zip(annotated, lat, lon):
        s["lat"] = float(la)
        s["lon"] = float(lo)
    return dict(rows=rows, table_rows=table_rows, sites=annotated,
                parcels=parcels, aoi=aoi)


# ---------------------------------------------------------------------------
# Stage 3: pixel-level segmentation quality on the synthetic scene (Sec. VI-C)
# ---------------------------------------------------------------------------
def stage_scene(seed: int = SEED, rf_model=None) -> dict:
    print(f"  generating {SCENE_SIZE}x{SCENE_SIZE} evaluation scene ...")
    scene = generate_scene(seed)
    if rf_model is None:
        lc = stage_landcover(seed)
        rf_model = lc["_rf_model"]
    res = run_scene_pipeline(scene, rf_model)
    seg = segmentation_metrics(res, scene["gt_encroachment"])
    print(f"  [scene] pixel-level IoU {seg['iou']:.3f} F1 {seg['f1']:.3f} "
          f"P {seg['precision']:.2f} R {seg['recall']:.2f} "
          f"({seg['tp']} TP / {seg['fp']} FP / {seg['fn']} FN)")
    km2 = (SCENE_SIZE * 10.0 / 1000.0) ** 2
    sec_per_km2 = res["timing"]["total_s"] / km2
    print(f"  [scene] pipeline time {res['timing']['total_s']:.2f}s for {km2:.2f} km2 "
          f"-> {sec_per_km2 * 1000:.1f} ms/km2")
    _save_json("table6b_segmentation", {**{k: v for k, v in seg.items()},
                                        "sec_per_km2": sec_per_km2})
    return dict(scene=scene, res=res, seg=seg, sec_per_km2=sec_per_km2)


# ---------------------------------------------------------------------------
# Stage 4: explainability (Fig. 4)
# ---------------------------------------------------------------------------
def stage_explainability(lc: dict) -> dict:
    X, y = lc["_X"], lc["_y"]
    rf = lc["_rf_model"]
    n_train = int(TRAIN_FRACTION * len(y))
    rng = np.random.default_rng(SEED + 99)
    idx = rng.permutation(len(y))
    te = idx[n_train:]
    imp = permutation_importance(rf, X[te], y[te], list(FEATURE_NAMES),
                                 n_repeats=10, seed=SEED)
    top3 = [d["feature"] for d in imp[:3]]
    print(f"  [explain] top-3 features: {', '.join(top3)}")
    _save_table("fig4_importance",
                [dict(feature=d["feature"], importance=round(d["importance_mean"], 4),
                      std=round(d["importance_std"], 4)) for d in imp],
                ["feature", "importance", "std"])
    return dict(importance=imp, top3=top3)


# ---------------------------------------------------------------------------
# Stage 5: comparison with deep architectures + scalability (Tables VII-VIII)
# ---------------------------------------------------------------------------
def stage_scalability(seg_timing_sec_per_km2: float) -> dict:
    rows = scalability_table(seg_timing_sec_per_km2, PARALLEL_WORKERS)
    table_rows = [dict(scope=r.scope, area_km2=r.area_km2, tiles=r.tiles,
                       processing_hours=round(r.wall_hours, 2),
                       core_hours=round(r.core_hours, 1), storage=r.storage)
                  for r in rows]
    _save_table("table8_scalability", table_rows,
                ["scope", "area_km2", "tiles", "processing_hours", "core_hours", "storage"])
    deep_rows = [dict(method=m, iou=v, note=n) for m, v, n in DEEP_BASELINES]
    print("  [scalability] " + "; ".join(
        f"{r.scope}: {r.tiles:,} tiles {r.wall_hours:.2f} h {r.storage}" for r in rows))
    return dict(scale_rows=rows, table_rows=table_rows, deep_rows=deep_rows)
