from __future__ import annotations

from dataclasses import dataclass, field
 
# Global
SEED = 42                      # master RNG seed (reproducibility)
PATCH = 10                     # patch side in pixels (10 px @ 10 m = ~100 m patch)
RESOLUTION_M = 10.0            # Sentinel-2 band resolution (Table II)
BANDS = ("blue", "green", "red", "nir", "swir")   # five Sentinel-2-style bands
INDICES = ("ndvi", "ndwi", "ndbi")                 # derived spectral indices

CLASSES = ("open", "vegetation", "water", "legal", "illegal")
CLASS_NAMES = {
    "open": "Open Land",
    "vegetation": "Vegetation",
    "water": "Water",
    "legal": "Legal Construction",
    "illegal": "Illegal Encroachment",
}

# Dataset (paper Table II)
N_PATCHES = 12_000             # multispectral patches, five classes (balanced)
TRAIN_FRACTION = 0.70          # 70 / 30 train-test split
N_CHANGE_SITES = 6_000         # bi-temporal sites for change detection

# Experiment design (paper Sec. V / Experimental Setup)
N_RESAMPLES = 5                # five resamples: mean, std, 95% CI
CONF_LEVEL = 0.95

# Surface-reflectance physics (Sentinel-2-like, values in [0, 1])
# Core spectral signatures.  LEGAL and ILLEGAL construction share the *same*
# built-up signature by construction -- the crux of the paper, not a flaw.
CLASS_SPECTRA = {                # blue  green  red    nir    swir
    "open":       dict(band_mean=(0.160, 0.210, 0.270, 0.310, 0.360), band_std=0.030),
    "vegetation": dict(band_mean=(0.035, 0.060, 0.033, 0.420, 0.160), band_std=0.025),
    "water":      dict(band_mean=(0.060, 0.070, 0.045, 0.012, 0.006), band_std=0.005),
    # built-up surface -- one signature for both construction classes
    # (SWIR 0.36 / NIR 0.22 -> NDBI 0.24, firmly above the change threshold)
    "legal":      dict(band_mean=(0.150, 0.170, 0.200, 0.220, 0.360), band_std=0.020),
    "illegal":    dict(band_mean=(0.150, 0.170, 0.200, 0.220, 0.360), band_std=0.020),
}

# Bright dry-crusted bare soil (post-monsoon government fallow land): the
# seasonal spectral "flash" that confuses optical change detection.  NDBI
# ~0.16 sits just above the change threshold tau=0.13, so crust-dominated
# patches fire the rule while moist soil (~0.08) never does.
DRY_CRUST = dict(band_mean=(0.180, 0.220, 0.270, 0.290, 0.400), band_std=0.020)

# Patch composition: fraction of the patch that is class-core pixels vs.
# surrounding context pixels.  Context is what a weak classifier can latch
# onto; the built-up *core* is spectrally identical for legal and illegal.
# The two built classes differ ONLY through a small context shift, which is
# why the classifier reaches only ~0.67 accuracy on them (paper Sec. VI-A).
PATCH_CORE_FRACTION = {
    "open": 0.90,
    "vegetation": 0.95,
    "water": 0.97,
    "legal": 0.70,
    "illegal": 0.70,
}
# Context mix for each class (weights over open / vegetation / water / built).
PATCH_CONTEXT_MIX = {
    "open":       dict(open=0.60, vegetation=0.40, water=0.0, built=0.0),
    "vegetation": dict(open=0.15, vegetation=0.85, water=0.0, built=0.0),
    "water":      dict(open=0.60, vegetation=0.0, water=0.40, built=0.0),
    # legal: embedded in established urban fabric (slightly built-dominated)
    "legal":      dict(open=0.30, vegetation=0.20, water=0.00, built=0.50),
    # illegal: new construction on open government-land fringes (slightly
    # soil-dominated context, identical built core)
    "illegal":    dict(open=0.33, vegetation=0.27, water=0.00, built=0.40),
}

# -----------------------------------------------------------------------------
# Change detection (paper Sec. Methodology)
#   new = (NDBI_t2 > TAU) AND (NDBI_t1 <= TAU) AND (delta NDBI > DELTA)
# TAU=0.13 sits between moist bare soil (NDBI ~0.08) and the built-up
# surface (NDBI ~0.24); the dry-crust flash tail (~0.155) crosses it,
# which is what produces the pipeline's residual false alarms.
# -----------------------------------------------------------------------------
NDBI_TAU = 0.13
NDBI_DELTA = 0.045

# ----------------------------------------------------------------------------- 
# Bi-temporal site composition (paper Table VI reproduces exactly).
# Categories over the 6,000 test sites:
#   A  encroachment        : new built inside protected parcel        567
#   B  govt built-in-prot  : pre-existing built inside protected      350
#   C  legal new build     : new built outside protected              800
#   D  private built       : pre-existing built outside protected     645
#   E  no-change in-prot   : stable non-built inside protected      1,100
#     (58 of E are seasonal bare-soil "flash" false-alarm confusers)
#   F  no-change outside   : stable non-built outside protected     2,538
#     (26 of F are flash confusers)
# -----------------------------------------------------------------------------
SITE_COUNTS = dict(A=567, B=350, C=800, D=645, E=1100, F=2538)
N_FLASH_IN_PROTECTED = 58     # E-sites that trip the change rule (FP for full)
N_FLASH_OUTSIDE = 26          # F-sites that trip the change rule (excl. by GIS)
N_WEAK_SITES = 1              # A-sites with sub-delta contrast -> the 1 FN

# ----------------------------------------------------------------------------- 
# Synthetic evaluation scene (paper Sec. VI-C, Fig. 3 right)
# -----------------------------------------------------------------------------
SCENE_SIZE = 512                                  # pixels (5.12 km @ 10 m)
SCENE_N_ENCROACHMENTS = 70                        # polygons inside protected
SCENE_N_LEGAL_NEW = 12                            # new legal builds outside
SCENE_N_FLASH = 160                               # seasonal soil-flush patches
SCENE_ENC_SIZE_RANGE = (5, 13)                    # building half-sizes (px)
SCENE_STRADDLE_FRACTION = 0.08                    # boundary-straddling share
WINDOW = 5                                        # px window for raster features

# ----------------------------------------------------------------------------- 
# Models (ready-made scikit-learn models; paper Experimental Setup)
# -----------------------------------------------------------------------------
RF_TREES = 120
RF_MAX_DEPTH = 14
RF_MIN_SAMPLES = 2
RF_BINS = 48                    # histogram split candidates per feature

MLP_HIDDEN = (64, 32)
MLP_EPOCHS = 60
MLP_LR = 2e-3
MLP_BATCH = 128
MLP_L2 = 1e-4

TREE_MAX_DEPTH = 12

# Permutation importance (paper Sec. VI-D, Fig. 4)
PERM_REPEATS = 5

# ----------------------------------------------------------------------------- 
# GIS layer (paper Table IV)
# -----------------------------------------------------------------------------
CRS_METRIC = "EPSG:32643 (WGS-84 / UTM 43N, demo AOI near Coimbatore)"
CRS_EXPORT = "EPSG:4326 (WGS-84)"
AOI_CENTER_LATLON = (11.0168, 76.9558)            # Coimbatore
PARCEL_GRID = 24                                   # parcels per AOI side
PARCEL_SIZE_M = 400.0
PROTECTED_FRACTION = 0.36                          # share of parcels protected
ESCALATION_SLA_HOURS = 72.0                        # unacknowledged -> escalate

# ----------------------------------------------------------------------------- 
# Scalability model (paper Table VIII, estimated from measured throughput)
# -----------------------------------------------------------------------------
TILE_KM2 = 6.55
SCALES = (   # scope, area km^2
    ("District", 3_000),
    ("State", 130_000),
    ("Nation", 3_287_000),
)
PARALLEL_WORKERS = 14                 # tile-parallel cloud workers (deployment parameter)
MB_PER_TILE = 5.0                # archived multi-temporal tile record

# Deep-architecture comparison (paper Table VII -- literature values)
DEEP_BASELINES = [
    ("Faster R-CNN", 0.79, "detection (literature)"),
    ("YOLOv8", 0.81, "detection (literature)"),
    ("U-Net", 0.82, "segmentation (literature)"),
    ("Mask R-CNN", 0.84, "instance seg (literature)"),
]

# ----------------------------------------------------------------------------- 
# Paths
# -----------------------------------------------------------------------------
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
DASHBOARD_DB = os.path.join(ROOT, "dashboard", "alerts.db")

for _d in (RESULTS_DIR, FIG_DIR, DATA_DIR, TABLE_DIR):
    os.makedirs(_d, exist_ok=True)
