"""Paper figures (Figs. 1-5), rendered with matplotlib (Agg backend)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..config import FIG_DIR, CLASS_NAMES

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_SPECTRAL = "#9ecae1"
C_CHANGE = "#4292c6"
C_FULL = "#08519c"
C_BAD = "#e6550d"


def _save(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig1_architecture() -> str:
    """Four-layer architecture of the land-monitoring system (Fig. 1 / Table I)."""
    layers = [
        ("Application", "web dashboard: GIS maps, alerts, escalation, reports"),
        ("Processing & AI", "preprocessing + RF land-cover + change detection + GIS overlay"),
        ("Data storage", "cloud object store + PostGIS (parcels, raster tiles)"),
        ("Data acquisition", "Bhuvan / Sentinel-2 / Landsat-8 multi-temporal imagery"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["#08519c", "#4292c6", "#9ecae1", "#deebf7"]
    title_colors = ["#ffffff", "#ffffff", "#0b2a4a", "#0b2a4a"]
    body_colors = ["#eaf3fb", "#f2f8fd", "#123c63", "#123c63"]
    for i, ((title, body), color) in enumerate(zip(layers, colors)):
        y = 0.80 - i * 0.22
        box = FancyBboxPatch(
            (0.16, y - 0.088), 0.68, 0.15,
            boxstyle="round,pad=0.012", fc=color, ec="#1f4e79", lw=1.2,
        )
        ax.add_patch(box)
        ax.text(0.5, y + 0.025, f"{title} layer", ha="center", va="center",
                fontsize=10, fontweight="bold", color=title_colors[i])
        ax.text(0.5, y - 0.034, body, ha="center", va="center", fontsize=8.2,
                color=body_colors[i])
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.094), (0.5, y - 0.136),
                                         arrowstyle="-|>", mutation_scale=16,
                                         color="#1f4e79", lw=1.4))
    # scheduled ingest / product out-flows
    ax.add_patch(FancyArrowPatch((0.08, 0.80), (0.15, 0.80), arrowstyle="-|>",
                                 mutation_scale=14, color="#666666", lw=1.2))
    ax.text(0.075, 0.79, "scheduled\nacquisition", ha="right", va="center", fontsize=7.5, color="#555555")
    ax.add_patch(FancyArrowPatch((0.85, 0.80), (0.92, 0.80), arrowstyle="-|>",
                                 mutation_scale=14, color="#666666", lw=1.2))
    ax.text(0.925, 0.79, "alerts to\nofficers", ha="left", va="center", fontsize=7.5, color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.01, 0.94)
    ax.axis("off")
    ax.set_title("Fig. 1  Four-layer architecture of the land-monitoring system")
    return _save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------------------
def fig2_landcover(per_class_f1: dict[str, float], cm, accuracy: float) -> str:
    """Land-cover accuracy and Random Forest confusion matrix (Fig. 2)."""
    cm = np.asarray(cm)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    names = [CLASS_NAMES[c] for c in CLASS_NAMES]
    vals = [per_class_f1[c] for c in CLASS_NAMES]
    colors = ["#74c476" if v > 0.9 else "#fdae6b" if v > 0.7 else C_BAD for v in vals]
    bars = ax1.barh(range(5), vals, color=colors, edgecolor="#333333", lw=0.5)
    ax1.set_yticks(range(5), names)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel("F1 score")
    ax1.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax1.set_title(f"Per-class F1 (5-resample mean accuracy {accuracy * 100:.1f}%)")

    im = ax2.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                     color="white" if cm[i, j] > cm.max() * 0.6 else "#1a1a1a")
    short = ["Open", "Veg", "Water", "Legal", "Illegal"]
    ax2.set_xticks(range(5), short, rotation=30)
    ax2.set_yticks(range(5), short)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    acc_cm = 100.0 * np.trace(cm) / max(1, cm.sum())
    ax2.set_title(f"Random forest confusion matrix (accuracy {acc_cm:.1f}%)")
    ax2.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    fig.suptitle("Fig. 2  Land-cover accuracy and Random Forest confusion matrix", y=1.02)
    return _save(fig, "fig2_landcover.png")


# ---------------------------------------------------------------------------
def fig3_ablation(rows: list[dict], scene_res: dict, seg: dict) -> str:
    """Encroachment ablation + detected-change raster (Fig. 3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw=dict(width_ratios=[1, 1.05]))
    configs = [r["config"] for r in rows]
    metrics = ("precision", "recall", "f1", "iou")
    width = 0.19
    x = np.arange(len(configs))
    for k, m in enumerate(metrics):
        vals = [r[m] for r in rows]
        ax1.bar(x + (k - 1.5) * width, vals, width, label=m.upper(),
                color=["#9ecae1", "#4292c6", "#08519c", "#e6550d"][k])
    ax1.set_xticks(x, ["Spectral\nonly", "+ Change\ndetection", "+ Change\n+ GIS"])
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("score")
    ax1.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax1.set_title("Encroachment detection ablation")
    for k, m in enumerate(metrics):
        for xi, r in zip(x, rows):
            ax1.text(xi + (k - 1.5) * width, r[m] + 0.02, f"{r[m]:.2f}",
                     ha="center", fontsize=6.2)

    # right: detected-change raster with TP/FP/FN and protected boundary
    flag = scene_res["flag_full"]
    prot = scene_res["protected"]
    gt = scene_res["gt"]
    rgb = np.full(flag.shape + (3,), 0.94)
    rgb[prot] = [0.90, 0.96, 0.90]                     # protected zone tint
    tp = flag & gt
    fp = flag & ~gt
    fn = ~flag & gt
    rgb[fn] = [1.0, 0.85, 0.3]                         # FN: amber
    rgb[fp] = [0.95, 0.35, 0.25]                       # FP: red
    rgb[tp] = [0.10, 0.35, 0.85]                       # TP: blue
    b = scene_res["boundary"]
    rgb[b] = [0.0, 0.45, 0.0]                          # protected boundary
    ax2.imshow(rgb, interpolation="nearest")
    ax2.set_title(
        f"Detected-change raster  (IoU {seg['iou']:.3f}, F1 {seg['f1']:.3f}, "
        f"P {seg['precision']:.2f}, R {seg['recall']:.2f})",
        fontsize=9,
    )
    ax2.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="s", ls="", color=c, label=l, markersize=8)
        for c, l in [(np.array([0.10, 0.35, 0.85]), "TP"),
                     (np.array([0.95, 0.35, 0.25]), "FP"),
                     (np.array([1.0, 0.85, 0.3]), "FN"),
                     (np.array([0.0, 0.45, 0.0]), "protected boundary")]
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Fig. 3  Encroachment ablation and a detected-change raster "
                 "(TP/FP/FN with protected-zone boundary)", y=1.01)
    return _save(fig, "fig3_ablation.png")


# ---------------------------------------------------------------------------
def fig4_importance(imp: list[dict]) -> str:
    """Spectral feature importance (Fig. 4)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    names = [d["feature"] for d in imp][::-1]
    vals = [d["importance_mean"] for d in imp][::-1]
    errs = [d["importance_std"] for d in imp][::-1]
    top3 = set(d["feature"] for d in imp[:3])
    colors = [C_FULL if n in top3 else C_SPECTRAL for n in names]
    ax.barh(range(len(names)), vals, xerr=errs, color=colors, edgecolor="#333", lw=0.4)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("mean drop in accuracy when permuted")
    ax.set_title("Fig. 4  Spectral feature importance (permutation, top 3 highlighted)")
    return _save(fig, "fig4_feature_importance.png")


# ---------------------------------------------------------------------------
def fig5_comparison(proposed_iou: float, deep: list[tuple[str, float, str]],
                    scale_rows) -> str:
    """IoU vs deep architectures + processing-time scalability (Fig. 5)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    labels = [d[0] for d in deep] + ["Proposed\n(change + GIS)"]
    vals = [d[1] for d in deep] + [proposed_iou]
    colors = [C_SPECTRAL] * len(deep) + [C_FULL]
    bars = ax1.bar(range(len(vals)), vals, color=colors, edgecolor="#333", lw=0.5)
    ax1.bar_label(bars, fmt="%.2f", fontsize=8)
    ax1.axhline(max(d[1] for d in deep), color="#888", ls="--", lw=0.8)
    ax1.set_xticks(range(len(vals)), labels, fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("IoU")
    ax1.set_title("Encroachment IoU vs deep architectures")

    scopes = [r.scope for r in scale_rows]
    hours = [max(r.wall_hours, 1e-3) for r in scale_rows]
    ax2.plot(scopes, hours, "o-", color=C_FULL, lw=1.8)
    for s, h in zip(scopes, hours):
        label = f"{h:.2f} h" if h >= 0.1 else f"{h * 60:.0f} min"
        ax2.annotate(label, (s, h), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylabel("wall-clock processing hours (log)")
    ax2.set_title(f"Processing-time scalability ({scale_rows[0].tiles:,}-{scale_rows[-1].tiles:,} tiles)")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Fig. 5  Encroachment IoU versus deep architectures, and "
                 "processing-time scalability from district to nation", y=1.02)
    return _save(fig, "fig5_comparison_scalability.png")
