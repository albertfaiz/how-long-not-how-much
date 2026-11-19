#!/usr/bin/env python3
"""
================================================================================
 BIGBANG v3 — CHECKPOINTED, LEAKAGE-FREE ML PIPELINE FOR LIFE-EXPECTANCY PAPER
================================================================================
 Input   : Master_Cleaned_NoArtifacts_With_LIVE_Prefix.csv (or any county-year CSV)
 Target  : MeanLifeExpectency
 Output  : RESULTS_<stamp>/  — every metric, prediction, residual, model, SHAP
           value, interval, OOD score, stability table, and statistical test a
           reviewer could ask for, plus manuscript-compatible CSVs for 03_plots.py.

 WHAT v3 ADDS OVER v2 (the "never re-run from scratch again" release)
 ─────────────────────────────────────────────────────────────────────────────
 1. STAGE CHECKPOINTING + RESUME. Every expensive stage persists its result to
    checkpoints/<stage>.pkl guarded by a config-hash marker. Re-running the
    script skips completed stages. Crash at Step 14 → restart resumes at 14.
       --rerun stageA,stageB   force specific stages (downstream auto-invalidated)
       --fresh                 ignore all checkpoints, start clean
    Changing any scientific config (year range, leakage list, folds, seed,
    thresholds, data file) changes the hash and auto-invalidates stale caches.
 2. EXHAUSTIVE ARTIFACT SAVING at every stage: per-fold predictions WITH
    residuals + train metrics + timing, fold-assignment table (GEOID→fold),
    split indices (.npz), holdout GEOID list, feature lists at every pruning
    stage, correlation-pruning audit, config manifest with dataset SHA256.
 3. STATISTICAL TESTING SUITE: bootstrap CI on winner OOF R², permutation test
    (reuses preprocessed fold matrices — valid because preprocessing never
    touches y — so it is ~10× faster), and paired Wilcoxon on per-row |error|
    between the top models.
 4. Everything from v2 kept: honest 20% GEOID holdout, anchor-protected
    collinearity pruning, precomputed fold splits, CONCURRENT model tuning,
    quantile intervals + coverage check, OOD (PCA+Mahalanobis) for 2020-2023,
    SHAP fold-stability (Jaccard/Spearman/CV), top-3 inverse-RMSE ensemble,
    forward-chaining temporal CV, static aggregation of short-coverage
    modalities, diagnostic tables (bracket / temporal / county / residual).
 5. OUR locked scientific decisions preserved (see 01_RESULTS_LEDGER.md):
    - HARD leakage blocklist: exact LE-proxy names + regex net (v2's regexes
      alone would MISS SDOH_CHR_YRS_LIFE_LOST — ours catch it).
    - Default window 2009-2019 (SDOH availability); configurable via CLI.
    - Headline ablation SDOH vs Env vs Combined + smoothed-SDOH IHME baseline
      + incremental modality leaderboard.
    - FAST mode uses the previously-tuned gold-mine params (skips search);
      FULL mode re-tunes with the concurrent tournament.

 USAGE
 ─────────────────────────────────────────────────────────────────────────────
   python 02_pipeline.py --data master.csv                       # FAST, resumable
   python 02_pipeline.py --data master.csv --mode full           # full re-tune
   python 02_pipeline.py --data master.csv --outdir RESULTS_X    # resume that run
   python 02_pipeline.py --data master.csv --outdir RESULTS_X --rerun shap,tables
   python 02_pipeline.py --data master.csv --test-mode           # 3-5 min smoke test
   BIGBANG_N_JOBS=32 python 02_pipeline.py --data master.csv     # thread override

 HPC:  sbatch --cpus-per-task=64 --mem=256G bigbang.sh   (python -u 02_pipeline.py ...)
================================================================================
"""
from __future__ import annotations

# =============================================================================
# 0. SINGLE-TIER THREAD CONTROL  (must precede numpy/sklearn/booster imports)
# =============================================================================
import os
_DEFAULT_N_JOBS = max(1, (os.cpu_count() or 2) - 1)
N_JOBS = max(1, int(os.getenv("BIGBANG_N_JOBS", os.getenv("N_JOBS", str(_DEFAULT_N_JOBS)))))
_BLAS_THREADS = max(1, N_JOBS // 2)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_BLAS_THREADS))

import sys, gc, json, pickle, re, time, warnings, traceback, logging, signal
import platform, hashlib, argparse, random, subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from joblib import Parallel, delayed

try:
    import xgboost as xgb;      XGB_OK = True
except ImportError:             xgb = None; XGB_OK = False
try:
    import lightgbm as lgb;     LGB_OK = True
except ImportError:             lgb = None; LGB_OK = False
try:
    import catboost as cbst;    CB_OK = True
except ImportError:             cbst = None; CB_OK = False
try:
    import shap;                SHAP_OK = True
except ImportError:             shap = None; SHAP_OK = False

warnings.filterwarnings("ignore")

MODEL_AVAILABILITY = {"RF": True, "XGB": XGB_OK, "LGB": LGB_OK, "CB": CB_OK}
MODEL_LABELS = {"RF": "RandomForest", "XGB": "XGBoost",
                "LGB": "LightGBM", "CB": "CatBoost"}

# =============================================================================
# 1. CONFIG  (CLI-driven dataclass; its hash guards every checkpoint)
# =============================================================================
TARGET_COL = "MeanLifeExpectency"
ID_COLS    = ["GEOID", "year", "STATEFP", "COUNTYFP", "NAME", "location_name", "fips"]
EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "78"}   # CONUS only

# --- HARD leakage blocklist: exact names (ledger §4.1) + regex safety net.
#     NOTE: v2's regexes (^yll etc.) would MISS SDOH_CHR_YRS_LIFE_LOST because
#     of the prefix — the union below cannot.
LEAKAGE_EXACT = {
    "SDOH_CHR_AVG_LIFE_EXPEC",
    "SDOH_CHR_YRS_LIFE_LOST", "SDOH_CHR_YRS_LIFE_LOST_AIAN",
    "SDOH_CHR_YRS_LIFE_LOST_ASIAN", "SDOH_CHR_YRS_LIFE_LOST_BLACK",
    "SDOH_CHR_YRS_LIFE_LOST_HISPANIC", "SDOH_CHR_YRS_LIFE_LOST_WHITE",
    "SDOH_CHR_PREMAT_DEATH_RATE", "SDOH_CHR_INFANT_DEATH_RATE",
    "SDOH_CHR_CHILD_DEATH_RATE",
}
LEAKAGE_REGEXES = [
    r"LIFE_EXPEC", r"YRS_LIFE_LOST", r"DEATH_RATE", r"MORTALITY", r"PREMAT",
    r"life.*expect", r"expect.*life", r"(^|_)yll", r"(^|_)yld", r"(^|_)daly",
    r"all_cause_death",
]
_LEAK_RE = [re.compile(p, re.I) for p in LEAKAGE_REGEXES]

def is_leakage(col: str) -> bool:
    if col == TARGET_COL:
        return False
    if col in LEAKAGE_EXACT:
        return True
    return any(rx.search(col) for rx in _LEAK_RE)

# --- residual livestock aggregates that survive dataset rebuilds
EXPLICIT_DROP_COLS = {"total_area_km2", "total_weighted_mean",
                      "avg_weighted_mean", "total_head_count"}

# --- gold-mine hyperparameters from prior Optuna runs (ledger §3) — FAST mode
BEST_PARAMS_GOLD = {
    "XGB": dict(n_estimators=1500, max_depth=8, learning_rate=0.013832442448846112,
                subsample=0.7311989040672405, colsample_bytree=0.7116167224336398,
                min_child_weight=9, reg_alpha=0.2537815508265665,
                reg_lambda=0.679657809075816),
    "CB":  dict(iterations=2000, depth=8, learning_rate=0.051178145982369534,
                l2_leaf_reg=2.7970640394252375, random_strength=0.11400863701127326),
    "RF":  dict(n_estimators=1000, max_depth=25, min_samples_leaf=2, max_features=0.3),
    "LGB": dict(n_estimators=1500, num_leaves=127, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
}

@dataclass
class Config:
    data_path: str
    outdir: str
    mode: str = "fast"                    # fast (gold params) | full (re-tune)
    test_mode: bool = False
    seed: int = 42
    n_folds: int = 5
    n_iter_search: int = 40
    year_start: int = 2009               # OUR locked default (SDOH availability)
    year_end: int = 2019
    forecast_start: int = 2020
    forecast_end: int = 2023
    holdout_frac: float = 0.20
    max_missing_frac: float = 0.50
    nzv_thresh: float = 1e-8
    corr_thresh: float = 0.97
    importance_prune_thr: float = 0.001
    static_modality_thresh: float = 0.70
    shap_sample: int = 5000
    shap_bg: int = 500
    shap_topk: int = 30
    ood_pca: int = 20
    ood_quantile: float = 0.99
    bootstrap_n: int = 1000
    perm_n: int = 30
    loso_max_states: int = 10            # capped in fast; full = all
    exclude_prefixes: tuple = ("LIVE_",) # S1/S2/AIR handled by static-agg, not dropped
    rerun: list = field(default_factory=list)
    fresh: bool = False
    skip: list = field(default_factory=list)

    def apply_test_mode(self):
        if self.test_mode:
            self.n_folds = 2; self.n_iter_search = 3
            self.shap_sample = 200; self.shap_bg = 50
            self.bootstrap_n = 100; self.perm_n = 3
            self.loso_max_states = 3

    def science_hash(self) -> str:
        """Hash of everything that changes the SCIENCE (not runtime knobs).
        A checkpoint is valid only if this hash matches."""
        keys = ["data_path", "mode", "test_mode", "seed", "n_folds",
                "year_start", "year_end", "forecast_start", "forecast_end",
                "holdout_frac", "max_missing_frac", "nzv_thresh", "corr_thresh",
                "importance_prune_thr", "static_modality_thresh",
                "exclude_prefixes"]
        d = {k: getattr(self, k) for k in keys}
        d["leakage_exact"] = sorted(LEAKAGE_EXACT)
        d["leakage_regex"] = LEAKAGE_REGEXES
        d["target"] = TARGET_COL
        try:
            st = os.stat(self.data_path)
            d["data_sig"] = f"{st.st_size}_{int(st.st_mtime)}"
        except OSError:
            d["data_sig"] = "missing"
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str)
                              .encode()).hexdigest()[:16]


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="BIGBANG v3 checkpointed LE pipeline")
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default=None,
                    help="reuse an existing RESULTS dir to RESUME it")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--year-start", type=int, default=2009)
    ap.add_argument("--year-end", type=int, default=2019)
    ap.add_argument("--rerun", default="", help="comma list of stages to force")
    ap.add_argument("--skip", default="", help="comma list of stages to skip")
    ap.add_argument("--fresh", action="store_true", help="ignore all checkpoints")
    ap.add_argument("--list-stages", action="store_true")
    a = ap.parse_args()
    outdir = a.outdir or f"RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg = Config(data_path=a.data, outdir=outdir, mode=a.mode,
                 test_mode=a.test_mode, seed=a.seed,
                 year_start=a.year_start, year_end=a.year_end,
                 rerun=[s for s in a.rerun.split(",") if s],
                 skip=[s for s in a.skip.split(",") if s],
                 fresh=a.fresh)
    cfg.apply_test_mode()
    cfg._list_stages = a.list_stages
    return cfg

# =============================================================================
# 2. LOGGING
# =============================================================================
logger = logging.getLogger("bigbang")

def setup_logging(outdir: Path):
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)-5s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(outdir / "bigbang_pipeline.log", mode="a")
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

def log(msg, level="INFO"): getattr(logger, level.lower(), logger.info)(msg)

def log_sep(title=""):
    log("=" * 78)
    if title: log(f"  {title.upper()}"); log("=" * 78)

def log_exc(prefix="Unhandled exception"):
    log(f"{prefix}:\n{traceback.format_exc()}", "ERROR")


# =============================================================================
# 3. CHECKPOINT MANAGER — the "never start over" machinery
# =============================================================================
# Design:
#   • Each stage is a named function returning a picklable dict.
#   • mgr.run("name", fn) → if checkpoints/name.pkl exists AND its marker's
#     science_hash matches the current config AND the stage isn't forced,
#     the pickle is loaded and the stage skipped.
#   • Stages form a LINEAR dependency chain: once any stage actually
#     recomputes, every later stage is invalidated ("dirty cascade") — this
#     prevents a forced early stage from silently feeding stale caches
#     downstream.
#   • Artifacts (CSVs, models) are written inside stage functions; on resume
#     they already exist on disk, so nothing is lost.
class CheckpointManager:
    def __init__(self, outdir: Path, cfg: Config):
        self.dir = outdir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hash = cfg.science_hash()
        self.force = set(cfg.rerun)
        self.skip = set(cfg.skip)
        self.fresh = cfg.fresh
        self.dirty = False          # once True, all later stages recompute
        self.timings = []

    def _paths(self, name):
        return self.dir / f"{name}.pkl", self.dir / f"{name}.done.json"

    def _valid(self, name) -> bool:
        pkl, marker = self._paths(name)
        if self.fresh or self.dirty or (name in self.force):
            return False
        if not (pkl.exists() and marker.exists()):
            return False
        try:
            meta = json.loads(marker.read_text())
            return meta.get("science_hash") == self.hash
        except Exception:
            return False

    def run(self, name: str, fn, *args, **kwargs):
        if name in self.skip:
            log(f"⏭  Stage [{name}] SKIPPED by --skip")
            return None
        pkl, marker = self._paths(name)
        if self._valid(name):
            t0 = time.time()
            with open(pkl, "rb") as f:
                out = pickle.load(f)
            log(f"⏩ Stage [{name}] restored from checkpoint "
                f"({time.time()-t0:.1f}s load) — skipping compute")
            self.timings.append({"stage": name, "seconds": 0.0, "cached": True})
            return out
        # compute
        reason = ("forced" if name in self.force else
                  "downstream-of-recomputed" if self.dirty else
                  "fresh" if self.fresh else "no valid checkpoint")
        log_sep(f"STAGE [{name}]  ({reason})")
        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = time.time() - t0
        try:
            with open(pkl, "wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            marker.write_text(json.dumps({
                "science_hash": self.hash, "completed": datetime.now().isoformat(),
                "seconds": round(dt, 1)}))
        except Exception:
            log_exc(f"checkpoint save failed for [{name}] (stage output still returned)")
        self.dirty = True   # everything after a recomputed stage must recompute
        self.timings.append({"stage": name, "seconds": round(dt, 1), "cached": False})
        log(f"✔ Stage [{name}] complete in {dt:.1f}s (checkpointed)")
        return out

    def save_timing_report(self, outdir: Path):
        pd.DataFrame(self.timings).to_csv(outdir / "stage_timings.csv", index=False)


# =============================================================================
# 4. DEVICE DETECTION  (CUDA probe with timeout; env overrides)
# =============================================================================
def _probe_with_timeout(fn, timeout_s=15, fallback="cpu"):
    if not hasattr(signal, "SIGALRM"):
        try: return fn()
        except Exception: return fallback
    def _h(s, f): raise TimeoutError()
    old = signal.signal(signal.SIGALRM, _h); signal.alarm(timeout_s)
    try: return fn()
    except Exception: return fallback
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)

def _has_nvidia() -> bool:
    import shutil
    if shutil.which("nvidia-smi") is None: return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                           text=True, timeout=5)
        return r.returncode == 0 and "GPU" in (r.stdout or "")
    except Exception:
        return False

def detect_devices(seed: int):
    xgb_dev, cb_task = "cpu", "CPU"
    user = os.getenv("BIGBANG_DEVICE", "").lower()
    if XGB_OK:
        u = os.getenv("BIGBANG_XGB_DEVICE", user).lower()
        if u in ("cpu", "cuda"): xgb_dev = u
        elif _has_nvidia():
            def _p():
                m = xgb.XGBRegressor(tree_method="hist", device="cuda",
                                     n_estimators=2, verbosity=0,
                                     random_state=seed, n_jobs=1)
                m.fit(np.random.rand(20, 3), np.random.rand(20))
                m.predict(np.random.rand(5, 3)); return "cuda"
            xgb_dev = _probe_with_timeout(_p, 15)
    if CB_OK:
        u = os.getenv("BIGBANG_CB_DEVICE", user).upper()
        if u == "CUDA": u = "GPU"
        if u in ("CPU", "GPU"): cb_task = u
        elif _has_nvidia():
            def _p():
                m = cbst.CatBoostRegressor(iterations=2, task_type="GPU",
                                           verbose=False, allow_writing_files=False)
                m.fit(np.random.rand(40, 3), np.random.rand(40)); return "GPU"
            cb_task = _probe_with_timeout(_p, 20, fallback="CPU")
    return xgb_dev, cb_task

XGB_DEVICE, CB_TASK_TYPE = "cpu", "CPU"     # set properly in main()

# =============================================================================
# 5. MODALITY DEFINITIONS  (order: specific prefixes before general)
# =============================================================================
MODALITY_PREFIX_MAP = {
    "USDA_Cropland": ["USDA_"], "JRC_Water": ["JRC_"], "DEM_Topography": ["DEM_"],
    "Landsat_OLI": ["Landsat_"], "Sentinel2": ["S2_"], "Sentinel1_SAR": ["S1_"],
    "Landsat_LS57": ["LS57_"], "MODIS_LST": ["LST_"], "MODIS_NDVI": ["NDVI_EVI_", "NDVI_"],
    "Engineered": ["ENG_"], "AirQuality": ["AIR_"], "Economic": ["ECON_"],
    "Human_Modification": ["GHM_"], "Meta_Canopy": ["META_canopy_", "META_"],
    "PRISM_Climate": ["PRISM_"], "Soil_Texture": ["SOIL_"], "Soil_USDA": ["Soil_"],
    "TerraClimate": ["TERRA_"], "NightLights": ["NTL_"], "PM25": ["PM25_"],
    "SDOH": ["SDOH_"],
}
MODALITY_YEAR_RANGES = {
    "USDA_Cropland": (2000, 2019), "JRC_Water": (2000, 2019), "DEM_Topography": (2000, 2019),
    "Landsat_OLI": (2013, 2019), "Sentinel2": (2015, 2019), "Sentinel1_SAR": (2015, 2019),
    "Landsat_LS57": (2000, 2019), "MODIS_LST": (2000, 2019), "MODIS_NDVI": (2000, 2019),
    "Engineered": (2000, 2019), "AirQuality": (2019, 2023), "Economic": (2014, 2023),
    "Human_Modification": (2000, 2023), "Meta_Canopy": (2000, 2023),
    "PRISM_Climate": (2000, 2023), "Soil_Texture": (2000, 2023), "Soil_USDA": (2000, 2019),
    "TerraClimate": (2000, 2023), "NightLights": (2012, 2023), "PM25": (2000, 2019),
    "SDOH": (2009, 2019),
}
# High-level groups for the headline SDOH-vs-Env story
ENV_MODALITIES = {m for m in MODALITY_PREFIX_MAP if m != "SDOH"}

def assign_modality(col: str) -> str:
    for mod, prefixes in MODALITY_PREFIX_MAP.items():
        for p in prefixes:
            if col.startswith(p): return mod
    return "Other"

def overlap_fraction(modality, target_range) -> float:
    if modality not in MODALITY_YEAR_RANGES: return 1.0
    m_lo, m_hi = MODALITY_YEAR_RANGES[modality]; t_lo, t_hi = target_range
    ov = max(0, min(m_hi, t_hi) - max(m_lo, t_lo) + 1); span = t_hi - t_lo + 1
    return ov / span if span > 0 else 0.0

# =============================================================================
# 6. PROTECTED ANCHORS  (canonical, well-cited features immune to pruning)
# =============================================================================
PROTECTED_PATTERNS = [
    r"^PRISM_tmean_mean$", r"^PRISM_tmax_mean$", r"^PRISM_tmin_mean$",
    r"^PRISM_ppt_sum_mean$", r"^PRISM_summer_tmean_mean$", r"^PRISM_vpdmax_mean$",
    r"^PRISM_days_tmax_over_35C_mean$", r"^PRISM_days_tmin_under_0C_mean$",
    r"^NDVI_mean$", r"^NDVI_EVI_mean$", r"^Landsat_NDVI_mean$", r"^LS57_NDVI_mean$",
    r"^LST_Day_1km_mean$", r"^LST_Night_1km_mean$", r"^DEM_mean$",
    r"^SOIL_org_carbon_0cm_mean$", r"^SOIL_ph_0cm_mean$", r"^Soil_mean$",
    r"^TERRA_aet_mean$", r"^TERRA_pdsi_worst_month_mean$", r"^TERRA_def_mean$",
    r"^AIR_no2_mean_mean$", r"^AIR_aerosol_index_mean$", r"^PM25_mean$",
    r"^GHM_mean$", r"^ECON_mean$", r"^NTL_mean$",
    r"^META_canopy_height_mean$", r"^S1_VV_mean$", r"^S1_VH_mean$",
    r"^ENG_Diurnal_Temp_Range$", r"^ENG_NDVI_LST_Divergence$", r"^ENG_Wet_Bulb_Proxy$",
    # a few canonical SDOH anchors so the SDOH-only model keeps its backbone
    r"^SDOH_SAIPE_MEDIAN_HH_INCOME$", r"^SDOH_ACS_PCT_LT_HS$",
    r"^SDOH_CHR_PCT_SMOKING$", r"^SDOH_CHR_PCT_ADULT_OBESITY$",
]
_PROT_RE = [re.compile(p) for p in PROTECTED_PATTERNS]
def is_protected(name: str) -> bool:
    return any(rx.match(name) for rx in _PROT_RE)

# =============================================================================
# 7. METRICS
# =============================================================================
def compute_all_metrics(y_true, y_pred, label="") -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    n = len(y_true)
    if n < 3:
        return {"label": label, "n": n, "R2": np.nan, "MAE": np.nan,
                "MedAE": np.nan, "RMSE": np.nan, "MSE": np.nan,
                "MAPE_pct": np.nan, "Pearson_r": np.nan}
    try:
        mse = mean_squared_error(y_true, y_pred)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.where(y_true == 0, np.nan, y_true)
            mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
        pr, _ = sp_stats.pearsonr(y_true, y_pred)
        return {"label": label, "n": n,
                "R2": round(float(r2_score(y_true, y_pred)), 6),
                "MAE": round(float(mean_absolute_error(y_true, y_pred)), 6),
                "MedAE": round(float(np.median(np.abs(y_true - y_pred))), 6),
                "RMSE": round(float(np.sqrt(mse)), 6),
                "MSE": round(float(mse), 6),
                "MAPE_pct": round(float(mape), 4),
                "Pearson_r": round(float(pr), 6)}
    except Exception:
        log_exc("metric computation failed")
        return {"label": label, "n": n, "R2": np.nan, "MAE": np.nan,
                "MedAE": np.nan, "RMSE": np.nan, "MSE": np.nan,
                "MAPE_pct": np.nan, "Pearson_r": np.nan}

def summarise_folds(fold_metrics: list) -> dict:
    if not fold_metrics: return {}
    keys = [k for k in fold_metrics[0] if k not in ("label", "n")]
    out = {"n_folds": len(fold_metrics)}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics if not pd.isna(fm[k])]
        out[f"{k}_mean"] = round(float(np.mean(vals)), 6) if vals else np.nan
        out[f"{k}_std"]  = round(float(np.std(vals)), 6) if vals else np.nan
    return out

# =============================================================================
# 8. PREPROCESSING  (fit on train fold only — leakage-free)
# =============================================================================
def preprocess_fit(X_train: pd.DataFrame):
    Xf = X_train.replace([np.inf, -np.inf], np.nan)
    return (Xf.median(numeric_only=True),
            Xf.quantile(0.001), Xf.quantile(0.999))

def preprocess_transform(X, medians, lo, hi):
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians)
    X = X.clip(lower=lo, upper=hi, axis=1)
    return X.fillna(0.0)

# =============================================================================
# 9. STATIC AGGREGATION  (kills flatline-imputation anachronism)
# =============================================================================
def apply_static_aggregation(df, feature_cols, target_range, threshold):
    static_cols, actions = [], []
    for mod, yrng in MODALITY_YEAR_RANGES.items():
        cov = overlap_fraction(mod, target_range)
        if cov >= threshold:
            actions.append({"modality": mod, "coverage_frac": round(cov, 3),
                            "action": "kept_temporal", "n_features": 0}); continue
        mod_cols = [c for c in feature_cols if assign_modality(c) == mod]
        if not mod_cols: continue
        m_lo, m_hi = yrng
        mask = (df["year"] >= m_lo) & (df["year"] <= m_hi)
        cty = df.loc[mask, ["GEOID"] + mod_cols].groupby("GEOID").mean()
        for c in mod_cols:
            df[c] = df["GEOID"].map(cty[c])
        static_cols += mod_cols
        actions.append({"modality": mod, "coverage_frac": round(cov, 3),
                        "action": "static_per_county_mean", "n_features": len(mod_cols),
                        "observed_years": f"{m_lo}-{m_hi}"})
    return df, static_cols, pd.DataFrame(actions)

# =============================================================================
# 10. COLLINEARITY PRUNER  (numpy-accelerated, anchor-protected)
# =============================================================================
def collinearity_pruning_report(X, cfg: Config):
    audit = []
    X_ = X.replace([np.inf, -np.inf], np.nan)
    miss = X_.isna().mean()
    too_missing = miss[miss > cfg.max_missing_frac].index.tolist()
    for c in too_missing:
        audit.append({"feature": c, "stage": "missingness",
                      "score": float(miss[c]), "kept_partner": ""})
    keep = [c for c in X_.columns if c not in too_missing]
    Xv = X_[keep].fillna(X_[keep].median(numeric_only=True))
    var = Xv.var(numeric_only=True)
    nzv = var[var < cfg.nzv_thresh].index.tolist()
    for c in nzv:
        audit.append({"feature": c, "stage": "near_zero_variance",
                      "score": float(var.get(c, 0.0)), "kept_partner": ""})
    keep = [c for c in keep if c not in nzv]
    if len(keep) > 1:
        sub = Xv[keep]
        if len(sub) > 20000:
            sub = sub.sample(n=20000, random_state=cfg.seed)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = np.abs(np.corrcoef(sub.values, rowvar=False))
        np.fill_diagonal(cm, 0.0); cm = np.where(np.isnan(cm), 0.0, cm)
        cols = sub.columns.tolist(); n = len(cols)
        keep_mask = np.ones(n, dtype=bool)
        def _score(name):
            s = 0.0
            if name.endswith("_mean"): s -= 5
            if "weighted_mean" in name: s += 3
            if "_stdDev" in name or "_std" in name: s += 4
            if "growth_rate_" in name: s += 6
            for tag in ("_p10","_p20","_p25","_p30","_p40","_p60","_p70","_p75","_p80","_p90","_p99"):
                if tag in name: s += 2; break
            if name.endswith("_max") or name.endswith("_min"): s += 1
            return s + len(name) * 0.01
        order = np.argsort(-cm.mean(axis=0))
        for ic in order:
            if not keep_mask[ic]: continue
            for ip in np.where((cm[ic] > cfg.corr_thresh) & keep_mask)[0]:
                if ip == ic or not keep_mask[ip]: continue
                c, p = cols[ic], cols[ip]
                ca, pa = is_protected(c), is_protected(p)
                if ca and not pa: drop, kept = ip, c
                elif pa and not ca: drop, kept = ic, p
                else:
                    drop = ip if _score(c) <= _score(p) else ic
                    kept = cols[ic if drop == ip else ip]
                keep_mask[drop] = False
                audit.append({"feature": cols[drop], "stage": f"corr>{cfg.corr_thresh:.2f}",
                              "score": float(cm[ic, ip]), "kept_partner": kept})
                if drop == ic: break
        keep = [cols[i] for i in range(n) if keep_mask[i]]
    return keep, pd.DataFrame(audit)

# =============================================================================
# 11. MODEL FACTORY + SEARCH SPACES
# =============================================================================
def make_estimator(name, params=None, threads=N_JOBS, seed=42):
    p = dict(params or {})
    if name == "RF":
        return RandomForestRegressor(random_state=seed, n_jobs=threads, **p)
    if name == "XGB" and XGB_OK:
        base = dict(verbosity=0, random_state=seed, tree_method="hist",
                    device=XGB_DEVICE, n_jobs=threads)
        return xgb.XGBRegressor(**{**base, **p})
    if name == "LGB" and LGB_OK:
        base = dict(verbosity=-1, random_state=seed, n_jobs=threads, force_col_wise=True)
        obj = p.pop("objective", None); alpha = p.pop("alpha", None)
        if obj: base["objective"] = obj
        if alpha is not None: base["alpha"] = alpha
        return lgb.LGBMRegressor(**{**base, **p})
    if name == "CB" and CB_OK:
        base = dict(verbose=False, random_seed=seed, thread_count=threads,
                    allow_writing_files=False, task_type=CB_TASK_TYPE)
        return cbst.CatBoostRegressor(**{**base, **p})
    raise ValueError(f"model {name} unavailable")

def search_space(name, test_mode):
    if test_mode:
        return {"RF": {"n_estimators":[50,100],"max_depth":[8,12],"min_samples_leaf":[1,2],"max_features":["sqrt"]},
                "XGB":{"n_estimators":[80,120],"max_depth":[4,6],"learning_rate":[0.05,0.1],"subsample":[0.8,1.0]},
                "LGB":{"n_estimators":[80,120],"max_depth":[6,10],"learning_rate":[0.05,0.1],"num_leaves":[15,31]},
                "CB": {"iterations":[80,120],"depth":[4,6],"learning_rate":[0.05,0.1],"l2_leaf_reg":[1,3]}}[name]
    return {"RF": {"n_estimators":[800,1200,1600,2000],"max_depth":[None,25,35,50],
                   "min_samples_split":[2,4,8],"min_samples_leaf":[1,2,4],
                   "max_features":["sqrt",0.3,0.5],"max_samples":[0.7,0.85,0.95]},
            "XGB":{"n_estimators":[1000,1500,2000,2500],"max_depth":[6,8,10,12],
                   "learning_rate":[0.01,0.02,0.05],"subsample":[0.7,0.8,0.9],
                   "colsample_bytree":[0.4,0.5,0.6,0.8],"reg_alpha":[0.0,0.1,1.0],
                   "reg_lambda":[1.0,5.0,10.0],"min_child_weight":[1,3,5],"gamma":[0.0,0.1,1.0]},
            "LGB":{"n_estimators":[1000,1500,2000,2500],"max_depth":[-1,8,12,16],
                   "learning_rate":[0.01,0.02,0.05],"num_leaves":[31,63,95,127],
                   "subsample":[0.6,0.8,1.0],"colsample_bytree":[0.4,0.6,0.8],
                   "reg_alpha":[0.0,0.1,1.0],"reg_lambda":[0.0,1.0,5.0,10.0],
                   "min_child_samples":[10,20,40]},
            "CB": {"iterations":[800,1200,1600,2000],"depth":[6,8,10],
                   "learning_rate":[0.03,0.05,0.1],"l2_leaf_reg":[1,3,5,7],
                   "subsample":[0.7,0.85,1.0],"colsample_bylevel":[0.6,0.8,1.0],
                   "min_data_in_leaf":[1,5,20]}}[name]

DEFAULT_PARAMS_FOR_ABLATION = {
    "RF": dict(n_estimators=500, max_depth=30, min_samples_leaf=2, max_features="sqrt"),
    "XGB": dict(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
    "LGB": dict(n_estimators=800, max_depth=-1, num_leaves=63, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20),
    "CB": dict(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=3),
}

# =============================================================================
# 12. PRECOMPUTED FOLD SPLITS  (saved to disk so folds are auditable/reusable)
# =============================================================================
def precompute_fold_splits(X, y, groups, cfg: Config, outdir: Path = None,
                           tag: str = "main"):
    gkf = GroupKFold(n_splits=cfg.n_folds)
    splits = []
    fold_assign = np.full(len(X), -1, dtype=int)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        m, lo, hi = preprocess_fit(X.iloc[tr])
        splits.append({"fold": k, "tr_idx": tr, "te_idx": te,
                       "X_train": preprocess_transform(X.iloc[tr], m, lo, hi),
                       "y_train": y.iloc[tr],
                       "X_test": preprocess_transform(X.iloc[te], m, lo, hi),
                       "y_test": y.iloc[te], "medians": m, "lo": lo, "hi": hi})
        fold_assign[te] = k
    if outdir is not None:
        # persist fold assignment (GEOID→fold) + raw indices so any later
        # re-analysis uses IDENTICAL folds without recomputing
        pd.DataFrame({"row": np.arange(len(X)), "GEOID": groups.values,
                      "fold": fold_assign}).to_csv(
                          outdir / f"fold_assignment_{tag}.csv", index=False)
        np.savez_compressed(outdir / f"fold_indices_{tag}.npz",
                            **{f"train_{s['fold']}": s["tr_idx"] for s in splits},
                            **{f"test_{s['fold']}": s["te_idx"] for s in splits})
    return splits

# =============================================================================
# 13. CV RUNNER  (saves residuals + train metrics + timing for every fold)
# =============================================================================
def run_cv_on_splits(splits, name, params, cfg, label="", years=None,
                     groups=None, return_models=False):
    fold_metrics, all_preds, fold_models, fold_rows = [], [], [], []
    for spl in splits:
        t0 = time.time()
        mdl = make_estimator(name, params, seed=cfg.seed)
        mdl.fit(spl["X_train"], spl["y_train"])
        pred_te = mdl.predict(spl["X_test"])
        pred_tr = mdl.predict(spl["X_train"])
        dt = time.time() - t0
        fm = compute_all_metrics(spl["y_test"], pred_te, f"{label}_fold{spl['fold']}")
        tr_r2 = r2_score(spl["y_train"], pred_tr)
        fold_metrics.append(fm)
        fold_rows.append({"fold": spl["fold"], "label": label,
                          "test_R2": fm["R2"], "test_RMSE": fm["RMSE"],
                          "test_MAE": fm["MAE"], "train_R2": round(float(tr_r2), 6),
                          "overfit_gap": round(float(tr_r2 - (fm["R2"] or 0)), 6),
                          "n_test": fm["n"], "seconds": round(dt, 1)})
        log(f"  [{label}] fold {spl['fold']}/{len(splits)}: "
            f"test R²={fm['R2']:.4f} RMSE={fm['RMSE']:.4f} "
            f"(train R²={tr_r2:.4f}, gap={tr_r2-(fm['R2'] or 0):.3f}, {dt:.1f}s)")
        te = spl["te_idx"]
        all_preds.append(pd.DataFrame({
            "GEOID": groups.iloc[te].values if groups is not None else te,
            "year": years.iloc[te].values if years is not None else np.nan,
            "actual": spl["y_test"].values, "predicted": pred_te,
            "residual": spl["y_test"].values - pred_te,
            "abs_error": np.abs(spl["y_test"].values - pred_te), "fold": spl["fold"]}))
        if return_models: fold_models.append(mdl)
        else: del mdl; gc.collect()
    return (fold_metrics, summarise_folds(fold_metrics),
            pd.concat(all_preds, ignore_index=True),
            pd.DataFrame(fold_rows), fold_models)

# =============================================================================
# 14. TUNING  (reuses precomputed splits; falls back to gold params in FAST)
# =============================================================================
def tune_model_on_splits(splits, name, cfg, threads=N_JOBS):
    space = search_space(name, cfg.test_mode)
    keys, values = list(space.keys()), list(space.values())
    space_size = int(np.prod([len(v) for v in values]))
    n_eff = min(cfg.n_iter_search, max(1, space_size))
    rng = random.Random(cfg.seed)
    seen, trials, attempts = set(), [], 0
    while len(trials) < n_eff and attempts < max(50, n_eff * 20):
        attempts += 1
        combo = tuple(rng.choice(v) for v in values)
        if combo in seen: continue
        seen.add(combo); trials.append(dict(zip(keys, combo)))
    best_rmse, best_params, trial_log = np.inf, None, []
    for i, params in enumerate(trials, 1):
        try:
            rmses = []
            for spl in splits:
                mdl = make_estimator(name, params, threads=threads, seed=cfg.seed)
                mdl.fit(spl["X_train"], spl["y_train"])
                rmses.append(np.sqrt(mean_squared_error(
                    spl["y_test"], mdl.predict(spl["X_test"]))))
                del mdl
            mean_rmse = float(np.mean(rmses))
        except Exception:
            log_exc(f"trial {i} failed for {name}"); continue
        trial_log.append({"trial": i, "mean_rmse": mean_rmse, **params})
        if mean_rmse < best_rmse:
            best_rmse, best_params = mean_rmse, params
            log(f"    [{name}] trial {i}: RMSE={mean_rmse:.4f} ★")
    return best_params, best_rmse, pd.DataFrame(trial_log)

# =============================================================================
# 15. FEATURE IMPORTANCE
# =============================================================================
def get_feature_importances(model, names):
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, float)
        if imp.sum() > 0: imp = imp / imp.sum()
    else:
        imp = np.zeros(len(names))
    df = pd.DataFrame({"Feature": names, "Importance": imp}) \
           .sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df

# =============================================================================
# 16. SHAP + cross-fold stability
# =============================================================================
def compute_shap_for_model(model, X_full, cfg):
    if not SHAP_OK: return None, None, None
    Xs = X_full.sample(n=min(cfg.shap_sample, len(X_full)), random_state=cfg.seed)
    Xbg = X_full.sample(n=min(cfg.shap_bg, len(X_full)), random_state=cfg.seed + 1)
    try:
        ex = shap.TreeExplainer(model, data=Xbg, feature_perturbation="interventional")
    except Exception:
        ex = shap.TreeExplainer(model)
    try:
        sv = ex.shap_values(Xs, check_additivity=False)
    except TypeError:
        sv = ex.shap_values(Xs)
    except Exception:
        log_exc("SHAP failed"); return None, None, None
    imp = pd.DataFrame({"Feature": Xs.columns.tolist(),
                        "Mean_AbsSHAP": np.abs(sv).mean(0)}) \
            .sort_values("Mean_AbsSHAP", ascending=False).reset_index(drop=True)
    imp["SHAP_Rank"] = imp.index + 1
    return sv, Xs, imp

def compute_shap_stability(fold_models, fold_X_test, cfg):
    if not SHAP_OK or not fold_models: return pd.DataFrame(), pd.DataFrame()
    topk = cfg.shap_topk; dfs = []
    for fi, (mdl, Xte) in enumerate(zip(fold_models, fold_X_test), 1):
        Xs = Xte.sample(n=min(2000 if not cfg.test_mode else 100, len(Xte)),
                        random_state=cfg.seed + fi)
        Xbg = Xte.sample(n=min(200 if not cfg.test_mode else 50, len(Xte)),
                         random_state=cfg.seed + 100 + fi)
        try: ex = shap.TreeExplainer(mdl, data=Xbg, feature_perturbation="interventional")
        except Exception: ex = shap.TreeExplainer(mdl)
        try: sv = ex.shap_values(Xs, check_additivity=False)
        except TypeError: sv = ex.shap_values(Xs)
        except Exception: log_exc(f"SHAP fold {fi} failed"); continue
        d = pd.DataFrame({"Feature": Xs.columns.tolist(),
                          "Mean_AbsSHAP": np.abs(sv).mean(0)})
        d["fold"] = fi
        d = d.sort_values("Mean_AbsSHAP", ascending=False).reset_index(drop=True)
        d["Rank"] = d.index + 1; dfs.append(d)
    if not dfs: return pd.DataFrame(), pd.DataFrame()
    long = pd.concat(dfs, ignore_index=True)
    rw = long.pivot_table(index="Feature", columns="fold", values="Rank", aggfunc="first")
    vw = long.pivot_table(index="Feature", columns="fold", values="Mean_AbsSHAP", aggfunc="first")
    stab = pd.DataFrame({"Feature": rw.index, "rank_mean": rw.mean(1).values,
                         "rank_std": rw.std(1).values, "AbsSHAP_mean": vw.mean(1).values,
                         "AbsSHAP_CV": (vw.std(1) / vw.mean(1).replace(0, np.nan)).values,
                         "n_folds_topK": (rw <= topk).sum(1).values,
                         "n_folds_present": rw.notna().sum(1).values}) \
             .sort_values("rank_mean").reset_index(drop=True)
    rows = []; fp = sorted(rw.columns.tolist())
    for i in range(len(fp)):
        for j in range(i + 1, len(fp)):
            fi, fj = fp[i], fp[j]
            ti = set(rw[rw[fi] <= topk].index); tj = set(rw[rw[fj] <= topk].index)
            jac = len(ti & tj) / len(ti | tj) if (ti | tj) else np.nan
            common = rw.dropna(subset=[fi, fj])
            rho = sp_stats.spearmanr(common[fi], common[fj])[0] if len(common) > 5 else np.nan
            rows.append({"fold_i": fi, "fold_j": fj,
                         f"Jaccard_top{topk}": round(float(jac), 4),
                         "Spearman_rho": round(float(rho), 4)})
    return stab, pd.DataFrame(rows)

# =============================================================================
# 17. QUANTILE INTERVALS
# =============================================================================
def fit_quantile_models(Xtr, ytr, params, cfg, alphas=(0.1, 0.5, 0.9)):
    if not LGB_OK: return {}
    out = {}
    for a in alphas:
        m = make_estimator("LGB", {**params, "objective": "quantile", "alpha": a}, seed=cfg.seed)
        m.fit(Xtr, ytr); out[a] = m
    return out

def predict_with_intervals(qmodels, X):
    if not qmodels: return pd.DataFrame(index=X.index)
    out = {f"q{int(a*100):02d}": m.predict(X) for a, m in qmodels.items()}
    df = pd.DataFrame(out, index=X.index)
    if "q10" in df and "q90" in df: df["interval_width_80"] = df["q90"] - df["q10"]
    return df

# =============================================================================
# 18. OOD DETECTION
# =============================================================================
def fit_ood_detector(Xtr, cfg):
    nc = min(cfg.ood_pca, Xtr.shape[1], Xtr.shape[0] - 1)
    if nc < 2: return {"available": False}
    pca = PCA(n_components=nc, random_state=cfg.seed); Z = pca.fit_transform(Xtr.values)
    try:
        cov = MinCovDet(random_state=cfg.seed).fit(Z) if len(Z) <= 30000 else EmpiricalCovariance().fit(Z)
    except Exception:
        cov = EmpiricalCovariance().fit(Z)
    d = cov.mahalanobis(Z)
    return {"available": True, "pca": pca, "cov": cov,
            "threshold": float(np.quantile(d, cfg.ood_quantile)), "n_components": nc}

def score_ood(det, X):
    if not det.get("available"): return np.full(len(X), np.nan), np.zeros(len(X), bool)
    Z = det["pca"].transform(X.values); d = det["cov"].mahalanobis(Z)
    return d, d > det["threshold"]

# =============================================================================
# 19. ENSEMBLE (top-3 inverse-RMSE)
# =============================================================================
def fit_top_k_ensemble(splits, top_models, cfg, years, groups):
    fold_preds, weights = [], {}
    for name, params in top_models:
        _, summ, preds, _, _ = run_cv_on_splits(splits, name, params, cfg,
                                                 label=f"ENS_{name}", years=years, groups=groups)
        fold_preds.append((name, preds))
        weights[name] = 1.0 / max(summ.get("RMSE_mean", np.inf), 1e-6)
    z = sum(weights.values()); weights = {k: v / z for k, v in weights.items()}
    base = fold_preds[0][1].rename(columns={"predicted": f"pred_{fold_preds[0][0]}"}) \
                           .drop(columns=["residual", "abs_error"])
    for name, preds in fold_preds[1:]:
        base = base.merge(preds[["GEOID", "year", "fold", "predicted"]]
                          .rename(columns={"predicted": f"pred_{name}"}),
                          on=["GEOID", "year", "fold"], how="inner")
    base["predicted"] = sum(weights[n] * base[f"pred_{n}"] for n, _ in top_models)
    base["residual"] = base["actual"] - base["predicted"]
    base["abs_error"] = base["residual"].abs()
    fm = [compute_all_metrics(base[base.fold == f]["actual"], base[base.fold == f]["predicted"],
                              f"ens_fold{f}") for f in sorted(base["fold"].unique())]
    return base, summarise_folds(fm), weights

# =============================================================================
# 20. FORWARD-CHAINING TEMPORAL CV
# =============================================================================
def forward_chaining_temporal_cv(df, feats, name, params, cfg):
    lo, hi = cfg.year_start, cfg.year_end; rows = []; block = 3 if (hi - lo) < 12 else 4
    train_end = lo + max(5, (hi - lo) // 2)
    while train_end + 1 <= hi:
        test_end = min(train_end + block, hi)
        tr = df[(df.year >= lo) & (df.year <= train_end)]
        te = df[(df.year > train_end) & (df.year <= test_end)]
        if len(tr) < 500 or len(te) < 100: break
        m, l, h = preprocess_fit(tr[feats])
        mdl = make_estimator(name, params, seed=cfg.seed)
        mdl.fit(preprocess_transform(tr[feats], m, l, h), tr[TARGET_COL].astype(float))
        pred = mdl.predict(preprocess_transform(te[feats], m, l, h))
        fm = compute_all_metrics(te[TARGET_COL].astype(float), pred,
                                 f"fwd_{lo}-{train_end}_test_{train_end+1}-{test_end}")
        fm.update({"train_years": f"{lo}-{train_end}", "test_years": f"{train_end+1}-{test_end}",
                   "n_train": len(tr), "n_test": len(te)})
        rows.append(fm)
        log(f"  fwd-chain {fm['label']}: R²={fm['R2']:.4f} RMSE={fm['RMSE']:.4f}")
        train_end += block
    return pd.DataFrame(rows)

# =============================================================================
# 21. STATISTICAL TESTS  (bootstrap CI, permutation, paired Wilcoxon)
# =============================================================================
def bootstrap_r2_ci(y_true, y_pred, cfg):
    rng = np.random.default_rng(cfg.seed)
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    m = ~(np.isnan(yt) | np.isnan(yp)); yt, yp = yt[m], yp[m]
    idx = np.arange(len(yt)); r2s = []
    for _ in range(cfg.bootstrap_n):
        b = rng.choice(idx, len(idx), replace=True)
        r2s.append(r2_score(yt[b], yp[b]))
    lo, hi = np.percentile(r2s, [2.5, 97.5])
    return {"r2": float(r2_score(yt, yp)), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": cfg.bootstrap_n}

def permutation_test(splits, name, params, cfg, real_oof_r2):
    """Shuffle y, re-run CV on the SAME preprocessed fold matrices (valid
    because preprocessing never used y), collect null R² distribution."""
    rng = np.random.default_rng(cfg.seed); null = []
    for _ in range(cfg.perm_n):
        preds_all, ytrue_all = [], []
        for spl in splits:
            yptr = pd.Series(rng.permutation(spl["y_train"].values), index=spl["y_train"].index)
            mdl = make_estimator(name, params, seed=cfg.seed)
            mdl.fit(spl["X_train"], yptr)
            preds_all.append(mdl.predict(spl["X_test"])); ytrue_all.append(spl["y_test"].values)
            del mdl
        null.append(r2_score(np.concatenate(ytrue_all), np.concatenate(preds_all)))
    null = np.array(null)
    p = (np.sum(null >= real_oof_r2) + 1) / (cfg.perm_n + 1)
    return {"real_r2": float(real_oof_r2), "null_mean": float(null.mean()),
            "null_std": float(null.std()), "p_value": float(p), "n_perm": cfg.perm_n}, null

def paired_wilcoxon(preds_a, preds_b, label_a, label_b):
    """Paired test on per-row absolute error between two models (aligned by
    GEOID+year+fold)."""
    key = ["GEOID", "year", "fold"]
    m = preds_a[key + ["abs_error"]].merge(
        preds_b[key + ["abs_error"]], on=key, suffixes=("_a", "_b"))
    if len(m) < 10:
        return {"model_a": label_a, "model_b": label_b, "n": len(m),
                "wilcoxon_stat": np.nan, "p_value": np.nan, "median_ae_a": np.nan,
                "median_ae_b": np.nan}
    try:
        stat, p = sp_stats.wilcoxon(m["abs_error_a"], m["abs_error_b"])
    except Exception:
        stat, p = np.nan, np.nan
    return {"model_a": label_a, "model_b": label_b, "n": len(m),
            "wilcoxon_stat": float(stat), "p_value": float(p),
            "median_ae_a": float(m["abs_error_a"].median()),
            "median_ae_b": float(m["abs_error_b"].median())}

# =============================================================================
# 22. DIAGNOSTIC TABLES
# =============================================================================
def make_bracket_table(preds):
    bins = [60, 70, 74, 76.5, 79, 95]; labels = ["60-70","70-74","74-76.5","76.5-79","79-95"]
    p = preds.copy(); p["LE_Bracket"] = pd.cut(p["actual"], bins=bins, labels=labels)
    def _rmse(x): return float(np.sqrt((np.asarray(x, float) ** 2).mean()))
    return (p.groupby("LE_Bracket", observed=True)
              .agg(Unique_Counties=("GEOID","nunique"), Count=("actual","count"),
                   Mean_LE=("actual","mean"), MAE=("abs_error","mean"),
                   RMSE=("residual", _rmse), Mean_Residual=("residual","mean")).reset_index())

def make_temporal_table(preds):
    if "year" not in preds.columns: return pd.DataFrame()
    rows = []
    for yr, g in preds.groupby("year", observed=True):
        if pd.isna(yr): continue
        rows.append({"year": yr, **compute_all_metrics(g["actual"], g["predicted"], str(yr)),
                     "n_counties": g["GEOID"].nunique()})
    return pd.DataFrame(rows).sort_values("year") if rows else pd.DataFrame()

def make_county_error_table(preds):
    t = (preds.groupby("GEOID").agg(Mean_LE=("actual","mean"), MAE=("abs_error","mean"),
         Mean_Resid=("residual","mean"), N_Years=("actual","count")).reset_index().sort_values("MAE"))
    return t.head(20), t.tail(20).sort_values("MAE", ascending=False)

def make_residual_stats(preds):
    r = preds["residual"].dropna()
    if len(r) == 0: return pd.DataFrame(columns=["Statistic", "Value"])
    sw = sp_stats.shapiro(r.sample(min(5000, len(r)), random_state=42))
    return pd.DataFrame([("Mean_Residual", float(r.mean())), ("Std_Residual", float(r.std())),
        ("Skewness", float(sp_stats.skew(r))), ("Kurtosis", float(sp_stats.kurtosis(r))),
        ("Shapiro_W", float(sw[0])), ("Shapiro_p", float(sw[1])),
        ("Min", float(r.min())), ("Max", float(r.max())),
        ("CI95_lower", float(np.percentile(r, 2.5))),
        ("CI95_upper", float(np.percentile(r, 97.5)))], columns=["Statistic", "Value"])

# =============================================================================
# 23. HEADLINE-SPECIFIC BLOCKS (SDOH-vs-Env, smoothed-SDOH baseline, ablation)
# =============================================================================
def sdoh_env_split(feats):
    sdoh = [c for c in feats if assign_modality(c) == "SDOH"]
    env  = [c for c in feats if assign_modality(c) in ENV_MODALITIES]
    return sdoh, env

def baseline_smoothed_sdoh(df, sdoh_cols, name, params, cfg, groups, years):
    """IHME apples-to-apples: SDOH + fold-safe state-mean-LE spatial feature."""
    y = df[TARGET_COL].astype(float).values
    state = (pd.to_numeric(df["GEOID"]) // 1000).values
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(cfg.n_folds).split(df, y, groups):
        sm = pd.Series(y[tr]).groupby(state[tr]).mean(); gm = y[tr].mean()
        smt = np.array([sm.get(s, gm) for s in state[tr]])
        sme = np.array([sm.get(s, gm) for s in state[te]])
        Xtr = np.column_stack([df.iloc[tr][sdoh_cols].values, smt])
        Xte = np.column_stack([df.iloc[te][sdoh_cols].values, sme])
        m, lo, hi = preprocess_fit(pd.DataFrame(Xtr))
        mdl = make_estimator(name, params, seed=cfg.seed)
        mdl.fit(preprocess_transform(pd.DataFrame(Xtr), m, lo, hi), y[tr])
        oof[te] = mdl.predict(preprocess_transform(pd.DataFrame(Xte), m, lo, hi))
    return compute_all_metrics(y, oof, "smoothed_sdoh_IHME_comparable"), oof

# =============================================================================
# 24. MAIN — fully checkpointed stage graph
# =============================================================================
STAGE_LIST = ["load", "features", "holdout", "prune", "splits", "tournament",
              "cv_all", "final_fit", "importance", "shap", "quantile",
              "ensemble", "ablation", "headline", "temporal", "forecast",
              "stats", "tables", "export"]

def main():
    global XGB_DEVICE, CB_TASK_TYPE
    cfg = parse_args()
    if getattr(cfg, "_list_stages", False):
        print("Stages:", ", ".join(STAGE_LIST)); return
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)
    XGB_DEVICE, CB_TASK_TYPE = detect_devices(cfg.seed)
    mgr = CheckpointManager(outdir, cfg)
    t_start = time.time()

    log_sep("BIGBANG v3 — CHECKPOINTED LEAKAGE-FREE PIPELINE")
    log(f"OUTPUT      : {outdir}")
    log(f"DATA        : {cfg.data_path}")
    log(f"MODE        : {cfg.mode}  | TEST_MODE={cfg.test_mode}")
    log(f"WINDOW      : {cfg.year_start}-{cfg.year_end}  forecast {cfg.forecast_start}-{cfg.forecast_end}")
    log(f"SCIENCE HASH: {mgr.hash}  (checkpoints validated against this)")
    log(f"N_JOBS={N_JOBS} BLAS={_BLAS_THREADS} | XGB={XGB_DEVICE} CB={CB_TASK_TYPE}")
    log(f"Models: {MODEL_AVAILABILITY} | SHAP={SHAP_OK}")

    # ---- STAGE load ----
    def _load():
        df = pd.read_csv(cfg.data_path, low_memory=False)
        if "GEOID" not in df.columns and "fips" in df.columns:
            df["GEOID"] = df["fips"]
        df["GEOID"] = df["GEOID"].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(5)
        if "STATEFP" not in df.columns:
            df["STATEFP"] = df["GEOID"].str[:2]
        df["STATEFP"] = df["STATEFP"].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(2)
        n0 = len(df)
        df = df[~df["STATEFP"].isin(EXCLUDE_STATEFP)].reset_index(drop=True)
        log(f"CONUS filter: {n0-len(df):,} dropped -> {len(df):,} rows")
        if TARGET_COL not in df.columns:
            log(f"FATAL target missing", "ERROR"); sys.exit(1)
        sha = hashlib.sha256(open(cfg.data_path, "rb").read()).hexdigest()
        # leakage audit
        leak = sorted([c for c in df.columns if is_leakage(c)])
        log(f"⚠ leakage cols dropped ({len(leak)}): {leak}")
        (outdir / "leakage_dropped.json").write_text(json.dumps(leak, indent=2))
        # missingness
        miss = df.isnull().sum(); miss = miss[miss > 0].sort_values(ascending=False)
        md = miss.reset_index(); md.columns = ["col", "n_missing"]
        md["pct"] = md["n_missing"] / len(df) * 100
        md.to_csv(outdir / "data_quality_missing_report.csv", index=False)
        return {"df": df, "sha256": sha, "leakage": leak}
    S = mgr.run("load", _load)
    df_raw, sha256, leakage_cols = S["df"], S["sha256"], S["leakage"]

    # ---- STAGE features ----
    def _features():
        yr = pd.to_numeric(df_raw["year"], errors="coerce")
        dft = df_raw[(yr >= cfg.year_start) & (yr <= cfg.year_end)].dropna(
            subset=[TARGET_COL]).reset_index(drop=True)
        if cfg.test_mode:
            keep_g = dft["GEOID"].unique()[:50]
            dft = dft[dft["GEOID"].isin(keep_g)].reset_index(drop=True)
        forbidden = set(ID_COLS) | {TARGET_COL} | set(leakage_cols) | EXPLICIT_DROP_COLS
        cand = [c for c in dft.columns if c not in forbidden
                and not any(c.startswith(p) for p in cfg.exclude_prefixes)]
        num = [c for c in cand if pd.api.types.is_numeric_dtype(dft[c])]
        # static aggregation
        dft, static_cols, agg = apply_static_aggregation(
            dft, num, (cfg.year_start, cfg.year_end), cfg.static_modality_thresh)
        agg.to_csv(outdir / "static_aggregation_actions.csv", index=False)
        # modality coverage report
        cov_rows = []
        for mod, cnt in Counter(assign_modality(c) for c in num).items():
            yl, yh = MODALITY_YEAR_RANGES.get(mod, ("?", "?"))
            cov_rows.append({"modality": mod, "n_features": cnt, "year_lo": yl,
                             "year_hi": yh, "coverage": round(overlap_fraction(mod, (cfg.year_start, cfg.year_end)), 3)})
        pd.DataFrame(cov_rows).to_csv(outdir / "modality_coverage_report.csv", index=False)
        pd.DataFrame({"feature": num}).to_csv(outdir / "feature_list_initial.csv", index=False)
        log(f"feature matrix: {len(dft):,} rows × {len(num):,} feats | static-agg {len(static_cols)}")
        return {"df": dft, "feats": num, "static_cols": static_cols}
    S = mgr.run("features", _features)
    df, feats_initial, static_cols = S["df"], S["feats"], S["static_cols"]
    y = df[TARGET_COL].astype(float); groups = df["GEOID"]; years = df["year"]

    # ---- STAGE holdout ----
    def _holdout():
        rng = np.random.default_rng(cfg.seed)
        ug = np.array(sorted(df["GEOID"].unique())); rng.shuffle(ug)
        n_hold = max(5 if cfg.test_mode else 50, int(len(ug) * cfg.holdout_frac))
        hold = set(ug[:n_hold])
        is_h = df["GEOID"].isin(hold).values
        pd.DataFrame({"GEOID": sorted(hold)}).to_csv(outdir / "holdout_geoids.csv", index=False)
        log(f"holdout {len(hold)} counties / {is_h.sum():,} rows; dev {(~is_h).sum():,} rows")
        return {"is_holdout": is_h, "holdout_geoids": sorted(hold)}
    S = mgr.run("holdout", _holdout)
    is_h = np.asarray(S["is_holdout"]); holdout_geoids = S["holdout_geoids"]
    X_full = df[feats_initial]
    X_dev, y_dev = X_full.loc[~is_h].reset_index(drop=True), y.loc[~is_h].reset_index(drop=True)
    g_dev, yr_dev = groups.loc[~is_h].reset_index(drop=True), years.loc[~is_h].reset_index(drop=True)
    X_hold, y_hold = X_full.loc[is_h].reset_index(drop=True), y.loc[is_h].reset_index(drop=True)
    g_hold, yr_hold = groups.loc[is_h].reset_index(drop=True), years.loc[is_h].reset_index(drop=True)

    # ---- STAGE prune ----
    def _prune():
        kept, audit = collinearity_pruning_report(X_dev, cfg)
        audit.to_csv(outdir / "feature_pruning_audit.csv", index=False)
        pd.DataFrame({"feature": kept}).to_csv(outdir / "feature_list_kept.csv", index=False)
        log(f"pruned to {len(kept)} feats (anchors kept: {sum(is_protected(c) for c in kept)})")
        return {"kept": kept}
    kept_cols = mgr.run("prune", _prune)["kept"]
    X_dev, X_hold = X_dev[kept_cols], X_hold[kept_cols]

    # ---- STAGE splits ----
    def _splits():
        sp = precompute_fold_splits(X_dev, y_dev, g_dev, cfg, outdir=outdir, tag="main")
        return {"splits": sp}
    splits = mgr.run("splits", _splits)["splits"]

    # ---- STAGE tournament ----
    def _tournament():
        candidates = [m for m, ok in MODEL_AVAILABILITY.items() if ok]
        if cfg.mode == "fast":
            log("FAST mode: using gold-mine params (no search)")
            rows, params_all, trials = [], {}, {}
            for m in candidates:
                p = BEST_PARAMS_GOLD.get(m, DEFAULT_PARAMS_FOR_ABLATION[m])
                _, summ, _, _, _ = run_cv_on_splits(splits, m, p, cfg, label=f"gold_{m}",
                                                    years=yr_dev, groups=g_dev)
                rows.append({"model": m, "model_name": MODEL_LABELS[m],
                             "CV_RMSE": summ.get("RMSE_mean", np.nan),
                             "CV_R2": summ.get("R2_mean", np.nan), "best_params": p})
                params_all[m] = p
            tour = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
        else:
            log("FULL mode: concurrent tuning tournament")
            cpu_models = [m for m in candidates if m in ("RF", "LGB") or (m == "CB" and CB_TASK_TYPE == "CPU")]
            tpm = max(1, N_JOBS // max(1, len(cpu_models)))
            def _tune(m):
                thr = N_JOBS if (m == "XGB" and XGB_DEVICE == "cuda") else tpm
                try:
                    p, rmse, tl = tune_model_on_splits(splits, m, cfg, threads=thr)
                    tl.to_csv(outdir / f"tuning_trials_{m}.csv", index=False)
                    return {"model": m, "model_name": MODEL_LABELS[m], "CV_RMSE": rmse,
                            "best_params": p} if p else None
                except Exception:
                    log_exc(f"tuning failed {m}"); return None
            res = Parallel(n_jobs=len(candidates), backend="threading")(
                delayed(_tune)(m) for m in candidates)
            rows = [r for r in res if r]
            params_all = {r["model"]: r["best_params"] for r in rows}
            tour = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
        tour.drop(columns=["best_params"]).to_csv(outdir / "model_tournament.csv", index=False)
        (outdir / "best_params_all.json").write_text(json.dumps(params_all, indent=2, default=str))
        log(f"🏆 winner: {tour.iloc[0]['model']}  CV-RMSE={tour.iloc[0]['CV_RMSE']:.4f}")
        return {"tournament": tour, "params_all": params_all,
                "winner": tour.iloc[0]["model"]}
    S = mgr.run("tournament", _tournament)
    tournament_df, best_params_all, winner = S["tournament"], S["params_all"], S["winner"]
    winner_params = best_params_all[winner]

    # ---- STAGE cv_all ----
    def _cv_all():
        summaries, preds_by, foldrows_by = [], {}, []
        fmodels_winner, fXtest_winner = [], []
        for m, p in best_params_all.items():
            ret = (m == winner) and SHAP_OK
            _, summ, preds, foldrows, fmods = run_cv_on_splits(
                splits, m, p, cfg, label=MODEL_LABELS[m], years=yr_dev,
                groups=g_dev, return_models=ret)
            summ["model"] = m; summaries.append(summ)
            preds_by[m] = preds
            preds.to_csv(outdir / f"cv_predictions_{m}.csv", index=False)
            foldrows.to_csv(outdir / f"cv_fold_details_{m}.csv", index=False)
            foldrows_by.append(foldrows)
            if ret:
                fmodels_winner = fmods
                fXtest_winner = [s["X_test"] for s in splits]
        pd.DataFrame(summaries).to_csv(outdir / "cv_summary_all_models.csv", index=False)
        pd.concat(foldrows_by, ignore_index=True).to_csv(
            outdir / "cv_fold_details_ALL.csv", index=False)
        return {"summaries": summaries, "preds_by": preds_by,
                "fmodels_winner": fmodels_winner, "fXtest_winner": fXtest_winner}
    S = mgr.run("cv_all", _cv_all)
    all_summaries, preds_by_model = S["summaries"], S["preds_by"]
    fold_models_winner, fold_X_test_winner = S["fmodels_winner"], S["fXtest_winner"]

    # ---- STAGE final_fit ----
    def _final_fit():
        m, lo, hi = preprocess_fit(X_dev)
        Xd, Xh = preprocess_transform(X_dev, m, lo, hi), preprocess_transform(X_hold, m, lo, hi)
        mdl = make_estimator(winner, winner_params, seed=cfg.seed)
        mdl.fit(Xd, y_dev)
        hp = mdl.predict(Xh)
        hm = compute_all_metrics(y_hold, hp, "holdout")
        log(f"HOLDOUT: R²={hm['R2']:.4f} RMSE={hm['RMSE']:.4f} MAE={hm['MAE']:.4f}")
        hdf = pd.DataFrame({"GEOID": g_hold.values, "year": yr_hold.values,
                            "actual": y_hold.values, "predicted": hp,
                            "residual": y_hold.values - hp, "abs_error": np.abs(y_hold.values - hp)})
        hdf.to_csv(outdir / "holdout_predictions.csv", index=False)
        with open(outdir / "final_model.pkl", "wb") as f:
            pickle.dump({"model": mdl, "medians": m, "lo": lo, "hi": hi,
                         "features": kept_cols}, f)
        return {"model": mdl, "medians": m, "lo": lo, "hi": hi,
                "holdout_metrics": hm, "holdout_preds": hdf, "Xd": Xd, "Xh": Xh}
    S = mgr.run("final_fit", _final_fit)
    final_model = S["model"]; medians_f, lo_f, hi_f = S["medians"], S["lo"], S["hi"]
    holdout_metrics, holdout_preds_df = S["holdout_metrics"], S["holdout_preds"]
    X_dev_imp, X_hold_imp = S["Xd"], S["Xh"]

    # ---- STAGE importance (+ post-fit pruned model) ----
    def _importance():
        imp = get_feature_importances(final_model, kept_cols)
        imp.to_csv(outdir / "feature_importance_full.csv", index=False)
        keep2 = imp.loc[(imp.Importance >= cfg.importance_prune_thr)
                        | imp.Feature.apply(is_protected), "Feature"].tolist()
        pd.DataFrame({"feature": keep2}).to_csv(outdir / "feature_list_pruned.csv", index=False)
        mp = make_estimator(winner, winner_params, seed=cfg.seed)
        mp.fit(X_dev_imp[keep2], y_dev)
        phm = compute_all_metrics(y_hold, mp.predict(X_hold_imp[keep2]), "holdout_pruned")
        ps = precompute_fold_splits(X_dev[keep2], y_dev, g_dev, cfg, outdir=outdir, tag="pruned")
        _, sp, pp, fr, _ = run_cv_on_splits(ps, winner, winner_params, cfg,
                                            label=f"{winner}_pruned", years=yr_dev, groups=g_dev)
        pp.to_csv(outdir / "cv_predictions_winner_pruned.csv", index=False)
        log(f"importance-pruned to {len(keep2)} | pruned holdout R²={phm['R2']:.4f}")
        return {"imp": imp, "keep2": keep2, "model_pruned": mp,
                "pruned_holdout": phm, "pruned_oof": pp, "pruned_summary": sp}
    S = mgr.run("importance", _importance)
    imp_df, keep_after_imp = S["imp"], S["keep2"]
    final_model_pruned, pruned_holdout_metrics = S["model_pruned"], S["pruned_holdout"]
    preds_pruned = S["pruned_oof"]
    X_pruned, X_pruned_h = X_dev_imp[keep_after_imp], X_hold_imp[keep_after_imp]

    # ---- STAGE shap ----
    def _shap():
        sv, Xs, imp = compute_shap_for_model(final_model_pruned, X_pruned, cfg)
        if sv is not None:
            np.save(outdir / "shap_values.npy", sv)
            Xs.to_csv(outdir / "shap_sample_X.csv", index=False)
            imp.to_csv(outdir / "shap_importance.csv", index=False)
        stab, pair = compute_shap_stability(fold_models_winner, fold_X_test_winner, cfg)
        if not stab.empty:
            stab.to_csv(outdir / "shap_stability_per_feature.csv", index=False)
            pair.to_csv(outdir / "shap_stability_pairwise.csv", index=False)
        return {"shap_imp": imp, "stability": stab, "pairwise": pair}
    S = mgr.run("shap", _shap)
    shap_imp_df, stability_df = S.get("shap_imp"), S.get("stability")

    # ---- STAGE quantile ----
    def _quantile():
        if not LGB_OK: return {"ok": False}
        qp = best_params_all.get("LGB", DEFAULT_PARAMS_FOR_ABLATION["LGB"])
        qp = {k: v for k, v in qp.items() if k != "objective"}
        qm = fit_quantile_models(X_pruned, y_dev, qp, cfg)
        qd = predict_with_intervals(qm, X_pruned); qh = predict_with_intervals(qm, X_pruned_h)
        cov = np.nan
        if {"q10", "q90"}.issubset(qh.columns):
            cov = float(((y_hold.values >= qh["q10"].values) &
                         (y_hold.values <= qh["q90"].values)).mean())
            log(f"holdout 80% interval coverage: {cov*100:.1f}%")
        qd.to_csv(outdir / "quantile_predictions_dev.csv", index=False)
        qh.to_csv(outdir / "quantile_predictions_holdout.csv", index=False)
        return {"ok": True, "qmodels": qm, "coverage80": cov}
    S = mgr.run("quantile", _quantile)
    qmodels = S.get("qmodels", {}) if S else {}

    # ---- STAGE ensemble ----
    def _ensemble():
        top = [(r["model"], best_params_all[r["model"]]) for _, r in tournament_df.head(3).iterrows()]
        if len(top) < 2: return {"ok": False}
        ep, es, ew = fit_top_k_ensemble(splits, top, cfg, years=yr_dev, groups=g_dev)
        ep.to_csv(outdir / "ensemble_cv_predictions.csv", index=False)
        ehp = np.zeros(len(X_hold_imp))
        for name, params in top:
            mm = make_estimator(name, params, seed=cfg.seed); mm.fit(X_dev_imp, y_dev)
            ehp += ew[name] * mm.predict(X_hold_imp); del mm; gc.collect()
        ehm = compute_all_metrics(y_hold, ehp, "ensemble_holdout")
        log(f"ensemble CV R²={es.get('R2_mean', np.nan):.4f} | holdout R²={ehm['R2']:.4f}")
        return {"ok": True, "summary": es, "weights": ew, "holdout_metrics": ehm}
    ens = mgr.run("ensemble", _ensemble) or {"ok": False}

    # ---- STAGE ablation (modality single + LOO with sensible defaults) ----
    def _ablation():
        abl_params = DEFAULT_PARAMS_FOR_ABLATION[winner]
        rows = []
        full = next((s for s in all_summaries if s["model"] == winner), None)
        rows.append({"ablation_type": "FULL_MODEL", "modality": "ALL",
                     "n_features": len(kept_cols),
                     "R2_mean": full.get("R2_mean") if full else np.nan,
                     "RMSE_mean": full.get("RMSE_mean") if full else np.nan})
        f2m = {c: assign_modality(c) for c in kept_cols}
        for mk in sorted(set(f2m.values())):
            mc = [c for c in kept_cols if f2m[c] == mk]
            if not mc: continue
            try:
                sm = run_cv_on_splits(precompute_fold_splits(X_dev[mc], y_dev, g_dev, cfg),
                                      winner, abl_params, cfg, label=f"SINGLE_{mk}",
                                      years=yr_dev, groups=g_dev)[1]
                rows.append({"ablation_type": "SINGLE_MODALITY", "modality": mk,
                             "n_features": len(mc), "R2_mean": sm.get("R2_mean"),
                             "RMSE_mean": sm.get("RMSE_mean")})
            except Exception: log_exc(f"single {mk} failed")
            loo = [c for c in kept_cols if c not in mc]
            if not loo: continue
            try:
                lm = run_cv_on_splits(precompute_fold_splits(X_dev[loo], y_dev, g_dev, cfg),
                                      winner, abl_params, cfg, label=f"LOO_{mk}",
                                      years=yr_dev, groups=g_dev)[1]
                rows.append({"ablation_type": "LEAVE_ONE_OUT", "modality": mk,
                             "n_features": len(loo), "R2_mean": lm.get("R2_mean"),
                             "RMSE_mean": lm.get("RMSE_mean")})
            except Exception: log_exc(f"LOO {mk} failed")
        adf = pd.DataFrame(rows).sort_values(["ablation_type", "R2_mean"],
                                             ascending=[True, False]).reset_index(drop=True)
        adf.to_csv(outdir / "ablation_study.csv", index=False)
        return {"ablation": adf}
    mgr.run("ablation", _ablation)

    # ---- STAGE headline (SDOH vs Env vs Combined + smoothed-SDOH + incremental) ----
    def _headline():
        sdoh, env = sdoh_env_split(kept_cols)
        rows = []
        for scen, cols in [("SDOH_only", sdoh), ("Env_only", env), ("Combined", kept_cols)]:
            if not cols: continue
            _, summ, _, _, _ = run_cv_on_splits(
                precompute_fold_splits(X_dev[cols], y_dev, g_dev, cfg),
                winner, winner_params, cfg, label=scen, years=yr_dev, groups=g_dev)
            rows.append({"scenario": scen, "n_features": len(cols),
                         "r2": summ.get("R2_mean"), "rmse": summ.get("RMSE_mean"),
                         "mae": summ.get("MAE_mean")})
        pd.DataFrame(rows).to_csv(outdir / "modality_ablation.csv", index=False)
        # smoothed SDOH baseline (IHME comparable)
        base_rows = []
        if sdoh:
            m, oof = baseline_smoothed_sdoh(df.loc[~is_h].reset_index(drop=True), sdoh,
                                            winner, winner_params, cfg, g_dev, yr_dev)
            base_rows.append({"baseline": "smoothed_sdoh_IHME_comparable", **m})
        pd.DataFrame(base_rows).to_csv(outdir / "baselines.csv", index=False)
        # incremental leaderboard
        anchor_r2 = None
        inc_rows = []
        if sdoh:
            _, sa, _, _, _ = run_cv_on_splits(
                precompute_fold_splits(X_dev[sdoh], y_dev, g_dev, cfg),
                winner, winner_params, cfg, label="anchorSDOH", years=yr_dev, groups=g_dev)
            anchor_r2 = sa.get("R2_mean")
        for mk in sorted(ENV_MODALITIES):
            mc = [c for c in kept_cols if assign_modality(c) == mk]
            if not mc: continue
            combo = sdoh + mc
            _, sm, _, _, _ = run_cv_on_splits(
                precompute_fold_splits(X_dev[combo], y_dev, g_dev, cfg),
                winner, winner_params, cfg, label=f"+{mk}", years=yr_dev, groups=g_dev)
            inc_rows.append({"modality_added": mk, "n_features": len(mc),
                             "r2": sm.get("R2_mean"),
                             "gain_vs_sdoh": (sm.get("R2_mean") - anchor_r2) if anchor_r2 else np.nan,
                             "rmse": sm.get("RMSE_mean")})
        pd.DataFrame(inc_rows).sort_values("r2", ascending=False).to_csv(
            outdir / "incremental_modality.csv", index=False)
        return {"scenarios": rows}
    mgr.run("headline", _headline)

    # ---- STAGE temporal ----
    def _temporal():
        # true temporal holdout: train < (year_end-1), test last 2 yrs
        split_year = cfg.year_end - 1
        tr = df.loc[~is_h & (df["year"] < split_year).values].reset_index(drop=True)
        te = df.loc[~is_h & (df["year"] >= split_year).values].reset_index(drop=True)
        thr = None
        if len(tr) > 100 and len(te) > 50:
            m, lo, hi = preprocess_fit(tr[kept_cols])
            mdl = make_estimator(winner, winner_params, seed=cfg.seed)
            mdl.fit(preprocess_transform(tr[kept_cols], m, lo, hi), tr[TARGET_COL].astype(float))
            pred = mdl.predict(preprocess_transform(te[kept_cols], m, lo, hi))
            thr = compute_all_metrics(te[TARGET_COL].astype(float), pred, "temporal_holdout")
            pd.DataFrame([thr]).to_csv(outdir / "temporal_holdout.csv", index=False)
            tdf = te[["GEOID", "year", TARGET_COL]].copy(); tdf["pred"] = pred
            tdf.to_csv(outdir / "temporal_holdout_predictions.csv", index=False)
            log(f"temporal holdout (train<{split_year}): R²={thr['R2']:.4f}")
        fwd = forward_chaining_temporal_cv(df.loc[~is_h].reset_index(drop=True),
                                           kept_cols, winner, winner_params, cfg)
        if not fwd.empty: fwd.to_csv(outdir / "forward_chaining_temporal_cv.csv", index=False)
        return {"temporal": thr, "forward": fwd}
    mgr.run("temporal", _temporal)

    # ---- STAGE forecast (2020-2023 + OOD + intervals) ----
    def _forecast():
        yr = pd.to_numeric(df_raw["year"], errors="coerce")
        fc = df_raw[(yr >= cfg.forecast_start) & (yr <= cfg.forecast_end)].copy()
        if not len(fc): return {"ok": False}
        fca = fc.copy()
        for mod in MODALITY_YEAR_RANGES:
            if overlap_fraction(mod, (cfg.year_start, cfg.year_end)) >= cfg.static_modality_thresh:
                continue
            for c in [c for c in kept_cols if assign_modality(c) == mod]:
                if c in df.columns:
                    fca[c] = fca["GEOID"].map(df.groupby("GEOID")[c].mean())
        Xfc = preprocess_transform(fca.reindex(columns=kept_cols), medians_f, lo_f, hi_f)
        ood = fit_ood_detector(X_dev_imp, cfg)
        d, flag = score_ood(ood, Xfc)
        pred = final_model.predict(Xfc)
        out = fc[[c for c in ID_COLS if c in fc.columns]].copy()
        out["Predicted_LE"] = pred; out["OOD_Mahalanobis"] = d; out["OOD_flag"] = flag
        if qmodels:
            qi = predict_with_intervals(qmodels, Xfc[keep_after_imp])
            for c in qi.columns: out[c] = qi[c].values
        out.to_csv(outdir / "forecast_2020_2023.csv", index=False)
        if ood.get("available"):
            log(f"forecast OOD flagged: {flag.sum():,}/{len(flag):,} ({flag.mean()*100:.1f}%)")
        return {"ok": True, "n": len(out), "ood_threshold": ood.get("threshold")}
    mgr.run("forecast", _forecast)

    # ---- STAGE stats (bootstrap, permutation, wilcoxon) ----
    def _stats():
        winner_oof = preds_by_model.get(winner)
        results = {}
        if winner_oof is not None:
            boot = bootstrap_r2_ci(winner_oof["actual"], winner_oof["predicted"], cfg)
            pd.DataFrame([boot]).to_csv(outdir / "bootstrap_ci.csv", index=False)
            results["bootstrap"] = boot
            real_r2 = r2_score(winner_oof["actual"], winner_oof["predicted"])
            perm, null = permutation_test(splits, winner, winner_params, cfg, real_r2)
            pd.DataFrame([perm]).to_csv(outdir / "permutation_test.csv", index=False)
            np.save(outdir / "permutation_null_r2.npy", null)
            results["permutation"] = perm
            log(f"bootstrap R²={boot['r2']:.4f} [{boot['ci_low']:.4f},{boot['ci_high']:.4f}] | "
                f"perm p={perm['p_value']:.4f}")
        # paired wilcoxon across top models
        wrows = []
        models = [m for m in best_params_all if m in preds_by_model]
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                wrows.append(paired_wilcoxon(preds_by_model[models[i]], preds_by_model[models[j]],
                                             models[i], models[j]))
        if wrows: pd.DataFrame(wrows).to_csv(outdir / "paired_wilcoxon.csv", index=False)
        return results
    mgr.run("stats", _stats)

    # ---- STAGE tables ----
    def _tables():
        wp = preds_by_model.get(winner, preds_pruned)
        make_bracket_table(wp).to_csv(outdir / "table_bracket_error.csv", index=False)
        make_temporal_table(wp).to_csv(outdir / "table_temporal_performance.csv", index=False)
        bc, wc = make_county_error_table(wp)
        bc.to_csv(outdir / "table_best_counties.csv", index=False)
        wc.to_csv(outdir / "table_worst_counties.csv", index=False)
        make_residual_stats(wp).to_csv(outdir / "table_residual_stats.csv", index=False)
        # full in-sample predictions with holdout flag
        fx = preprocess_transform(X_full[kept_cols], medians_f, lo_f, hi_f)
        fp = final_model.predict(fx)
        ex = df[[c for c in ID_COLS if c in df.columns] + [TARGET_COL]].copy()
        ex["Predicted_LE"] = fp; ex["Residual"] = ex[TARGET_COL] - fp
        ex["was_in_holdout"] = is_h
        ex.to_csv(outdir / "full_dataset_predictions.csv", index=False)
        return {"ok": True}
    mgr.run("tables", _tables)

    # ---- STAGE export (master manifest + metrics_master) ----
    def _export():
        # flat metrics_master.csv aggregating headline numbers
        rows = []
        for m in all_summaries:
            rows.append({"experiment": "cv_tournament", "condition": m["model"],
                         "r2": m.get("R2_mean"), "rmse": m.get("RMSE_mean"),
                         "mae": m.get("MAE_mean")})
        rows.append({"experiment": "holdout", "condition": winner,
                     "r2": holdout_metrics["R2"], "rmse": holdout_metrics["RMSE"],
                     "mae": holdout_metrics["MAE"]})
        for f in ["modality_ablation.csv", "baselines.csv", "temporal_holdout.csv"]:
            p = outdir / f
            if p.exists():
                d = pd.read_csv(p)
                for _, r in d.iterrows():
                    rows.append({"experiment": f.replace(".csv", ""),
                                 "condition": r.get("scenario") or r.get("baseline") or r.get("label", "-"),
                                 "r2": r.get("r2") or r.get("R2"),
                                 "rmse": r.get("rmse") or r.get("RMSE"),
                                 "mae": r.get("mae") or r.get("MAE")})
        pd.DataFrame(rows).to_csv(outdir / "metrics_master.csv", index=False)
        manifest = {
            "timestamp": datetime.now().isoformat(), "science_hash": mgr.hash,
            "config": asdict(cfg), "dataset_sha256": sha256,
            "git_hash": (subprocess.getoutput("git rev-parse HEAD") or "n/a")[:40],
            "winner": winner, "winner_params": winner_params,
            "holdout_metrics": holdout_metrics,
            "pruned_holdout_metrics": pruned_holdout_metrics,
            "n_features_initial": len(feats_initial), "n_features_kept": len(kept_cols),
            "n_features_pruned": len(keep_after_imp),
            "n_counties_dev": int(g_dev.nunique()), "n_counties_holdout": len(holdout_geoids),
            "leakage_dropped": leakage_cols, "static_aggregated": static_cols,
            "package_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                 "xgboost": getattr(xgb, "__version__", None),
                                 "lightgbm": getattr(lgb, "__version__", None),
                                 "catboost": getattr(cbst, "__version__", None),
                                 "shap": getattr(shap, "__version__", None)},
            "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                         "xgb_device": XGB_DEVICE, "cb_task_type": CB_TASK_TYPE},
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return {"ok": True}
    mgr.run("export", _export)

    mgr.save_timing_report(outdir)
    log_sep("PIPELINE COMPLETE")
    log(f"Total wall time: {(time.time()-t_start)/60:.1f} min")
    log(f"All artifacts in: {outdir}")
    log(f"Resume/extend with: --outdir {outdir} [--rerun stage]")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        log_exc("PIPELINE CRASHED")
        sys.exit(1)
