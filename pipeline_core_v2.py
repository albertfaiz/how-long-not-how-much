#!/usr/bin/env python3
# =============================================================================
#  pipeline_core_v2.py
#  Shared machinery for BOTH studies on the 1,318-column superset dataset
#  (now including ERA5 reanalysis and CAMS atmospheric composition).
#
#  WHAT CHANGED FROM v1, AND WHY
#  -----------------------------------------------------------------------
#  1. NEW MODALITIES. ERA5 (23 features, 2003-2019) and CAMS (97 features,
#     2003-2019, including formaldehyde, NO2, ozone, aerosol speciation).
#     Because both start in 2003, Study A now runs 2003-2019.
#  2. PER-FEATURE GATING IS NOW MANDATORY, not a nicety. In this dataset the
#     "Sentinel-5P & Particulates" label mixes pm25_* (2000-2019, keep) with
#     air_* (2019-2023, drop). Gating by modality would be wrong either way.
#  3. LEAKAGE BLOCKLIST EXPANDED. The superset contains seven columns that are
#     functions of mortality, including sdoh_chr_avg_life_expec, which IS the
#     target under another name. All are blocked before any modelling.
#  4. DEGENERATE-FEATURE REMOVAL. 22 columns carry <=2 distinct values across
#     74,617 rows (20 of them meta_canopy_cover_*). They are dropped.
#  5. MODEL TOURNAMENT. Instead of assuming LightGBM, several learners compete
#     under the identical protocol and SHAP runs on the winner.
#  6. FULL ARTEFACT CAPTURE. Every run writes the fitted model, out-of-fold
#     predictions, per-fold metrics, the SHAP ranking, the gate report, the
#     coverage report, and a machine-readable metrics JSON, so downstream
#     plotting and analysis never needs to re-run the model.
#
#  INVARIANTS (unchanged, and enforced)
#    * County-grouped cross-validation; all years of a county in one fold.
#    * Imputation fitted on the TRAIN fold only. No global imputation anywhere.
#    * Variance and collinearity filters never see the target.
#    * A protected list shields physiologically motivated features from pruning.
# =============================================================================

from __future__ import annotations
import json, re, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  COLUMN ROLES
# ---------------------------------------------------------------------------
TARGET = "meanlifeexpectency"
GROUP_COL = "fips"
YEAR_COL = "year"
ID_COLS = ["fips", "year", "statefp", "countyfp", "name", "geoid", "is_post_2015"]

# ---------------------------------------------------------------------------
#  LEAKAGE BLOCKLIST
#  Columns that are functions of mortality. Including any of them lets the
#  model predict life expectancy from life expectancy. Non-negotiable.
# ---------------------------------------------------------------------------
LEAKAGE_EXACT = {
    "sdoh_chr_avg_life_expec",          # this IS the target, renamed
    "sdoh_chr_premat_death_rate",
    "sdoh_chr_infant_death_rate",
    "sdoh_chr_child_death_rate",
    "sdoh_chr_firearm_death_rate",
    "sdoh_chr_pct_alcohol_driv_death",
    "health_lung_cancer_mortality_rate",
}
LEAKAGE_REGEX = re.compile(
    r"(life_expec|lifeexpec|meanlife|_death_|death_rate|mortality|"
    r"yrs_life_lost|years_of_life|premature_death|infant_death|suicide_rate|"
    r"overdose|drug_death)", re.I)


def is_leakage(col: str) -> bool:
    return col in LEAKAGE_EXACT or bool(LEAKAGE_REGEX.search(col))


# ---------------------------------------------------------------------------
#  MODALITY REGISTRY  (prefix -> human label).  Longest prefix wins.
# ---------------------------------------------------------------------------
MODALITY_PREFIX = {
    "cams_": "CAMS atmospheric composition",
    "era5_": "ERA5 reanalysis",
    "prism_": "PRISM climate",
    "terra_": "TerraClimate",
    "lst_": "MODIS land-surface temperature",
    "ndvi_": "MODIS vegetation",
    "landsat_": "Landsat optical",
    "ls57_": "Landsat optical",
    "dem_": "Elevation and terrain",
    "ghm_": "Human modification",
    "jrc_": "Surface water",
    "meta_": "Canopy structure",
    "ntl_harmonized_": "Night-time lights",
    "ntl_": "Night-time lights (raw)",
    "pm25_": "Fine particulate matter",
    "pm1": "Particulate matter (other)",
    "air_": "Sentinel-5P air quality",
    "soil_": "Soil profile",
    "usda_": "Cropland and agriculture",
    "live_": "Livestock",
    "s1_": "Sentinel-1 radar",
    "s2_": "Sentinel-2 optical",
    "econ_": "Macro economics",
    "health_": "Health outcomes",
    "sdoh_": "Socioeconomic (SDOH)",
    "eng_": "Engineered (legacy)",
    "eng2_": "Engineered (v2)",
}
_PREFIXES_SORTED = sorted(MODALITY_PREFIX, key=len, reverse=True)


def modality_of(col: str) -> str:
    for p in _PREFIXES_SORTED:
        if col.startswith(p):
            return MODALITY_PREFIX[p]
    return "Unclassified"


# Modalities that are socioeconomic rather than environmental. Study A excludes
# these by design (its claim is "no census input"); Study B includes them.
SDOH_MODALITIES = {"Socioeconomic (SDOH)", "Macro economics", "Health outcomes"}


def is_sdoh(col: str) -> bool:
    return modality_of(col) in SDOH_MODALITIES


# ---------------------------------------------------------------------------
#  PROTECTED FEATURES
#  Physiologically motivated variables that must survive variance and
#  collinearity pruning, so that the question "does wet-bulb matter?" can
#  actually be asked rather than silently filtered away.
# ---------------------------------------------------------------------------
PROTECT_PATTERNS = [
    r"wet_?bulb", r"wbgt", r"heat_?index", r"humidex", r"discomfort",
    r"apparent_?temp", r"wind_?chill", r"tdmean", r"dew_?point", r"dewpoint",
    r"relative_humidity", r"specific_humidity", r"absolute_humidity",
    r"prism_tmin", r"tmin_min", r"night_cool", r"nightcool", r"vpd",
    r"diurnal", r"hot_?night", r"days_tmin_over", r"days_tmax_over",
    r"formaldehyde", r"nitrogen_dioxide", r"ozone", r"pm25_", r"pm2_5",
    r"total_column_water", r"lst_night",
]
_PROTECT_RE = re.compile("|".join(PROTECT_PATTERNS), re.I)


def is_protected(col: str) -> bool:
    return bool(_PROTECT_RE.search(col))


# ---------------------------------------------------------------------------
#  LOADING AND CLEANING
# ---------------------------------------------------------------------------
def load_master(csv_path: str, add_engineered: bool = True, verbose: bool = True):
    """Load the superset CSV, normalise column names, drop degenerate and
    duplicated columns, and (optionally) add the v2 engineered layer."""
    t0 = time.time()
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if verbose:
        print(f"[load] {csv_path}: {df.shape[0]:,} rows x {df.shape[1]:,} cols "
              f"({time.time()-t0:.1f}s)")

    # -- duplicate column names: keep the copy with the most observed values --
    dups = df.columns[df.columns.duplicated()].unique().tolist()
    if dups:
        keep_idx = []
        for name in df.columns.unique():
            block = df.loc[:, df.columns == name]
            if block.shape[1] == 1:
                keep_idx.append(block)
            else:
                best = block.iloc[:, int(np.argmax(block.notna().sum().values))]
                keep_idx.append(best.to_frame(name))
        df = pd.concat(keep_idx, axis=1)
        if verbose:
            print(f"[load] resolved {len(dups)} duplicated column names "
                  f"(kept the copy with the most observed values)")

    # -- contiguous US only, if the identifier allows it --
    if "statefp" in df.columns:
        before = len(df)
        df = df[~df["statefp"].isin([2, 15, 60, 66, 69, 72, 78])]
        if verbose and before != len(df):
            print(f"[load] dropped {before-len(df):,} non-contiguous rows")

    if add_engineered:
        from engineered_features_v2 import add_engineered_features
        df, cov = add_engineered_features(df, verbose=verbose)
    return df


def degenerate_columns(df: pd.DataFrame, min_unique: int = 3) -> list[str]:
    """Columns carrying fewer than `min_unique` distinct observed values.
    In this dataset that catches the 20 broken meta_canopy_cover_* columns."""
    out = []
    for c in df.columns:
        if c in ID_COLS or c == TARGET:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) < min_unique:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
#  TEMPORAL-COVERAGE GATE  (computed from the data, not from metadata)
# ---------------------------------------------------------------------------
def coverage_gate(df, window, min_coverage=0.95, allow_sdoh=False,
                  drop_raw_ntl=True, extra_exclude=(), verbose=True):
    """Return (kept_features, report). A feature is admitted only if it is
    actually observed for at least `min_coverage` of the county-years inside
    the study window. Every decision is recorded with its reason."""
    y0, y1 = window
    win = df[(df[YEAR_COL] >= y0) & (df[YEAR_COL] <= y1)]
    if len(win) == 0:
        raise ValueError(f"no rows in window {y0}-{y1}")

    degen = set(degenerate_columns(win))
    rows, kept = [], []
    for c in df.columns:
        if c in ID_COLS or c == TARGET:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            reason, ok = "non-numeric", False
        elif is_leakage(c):
            reason, ok = "LEAKAGE-outcome-derived", False
        elif c in extra_exclude:
            reason, ok = "explicitly-excluded", False
        elif c in degen:
            reason, ok = "degenerate-<3-unique-values", False
        elif (not allow_sdoh) and is_sdoh(c):
            reason, ok = "socioeconomic-not-in-scope", False
        elif drop_raw_ntl and c.startswith("ntl_") and not c.startswith("ntl_harmonized_"):
            reason, ok = "raw-NTL-superseded-by-harmonized", False
        else:
            cov = float(win[c].notna().mean())
            if cov >= min_coverage:
                reason, ok = f"admitted ({cov:.0%})", True
            else:
                reason, ok = f"coverage-below-threshold ({cov:.0%})", False
        cov = float(win[c].notna().mean()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
        rows.append(dict(feature=c, modality=modality_of(c), window_coverage=round(cov, 4),
                         admitted=ok, reason=reason, protected=is_protected(c)))
        if ok:
            kept.append(c)

    report = pd.DataFrame(rows).sort_values(
        ["admitted", "modality", "window_coverage"], ascending=[False, True, False])
    if verbose:
        adm = report[report.admitted]
        print(f"[gate] window {y0}-{y1}: admitted {len(adm)} of {len(report)} features "
              f"across {adm.modality.nunique()} modalities")
        for r, n in report[~report.admitted].reason.str.replace(
                r"\s*\(.*\)", "", regex=True).value_counts().items():
            print(f"        {n:>5} excluded: {r}")
        print(f"        {int(adm.protected.sum())} protected features retained")
    return kept, report


# ---------------------------------------------------------------------------
#  UNSUPERVISED REDUNDANCY FILTER  (never sees the target)
# ---------------------------------------------------------------------------
def redundancy_filter(X: pd.DataFrame, corr_threshold=0.98, var_threshold=1e-8,
                      verbose=True):
    protected = {c for c in X.columns if is_protected(c)}

    var = X.var(numeric_only=True, skipna=True)
    keep = [c for c in X.columns if c in protected or var.get(c, 0) > var_threshold]
    n_var = len(X.columns) - len(keep)

    sub = X[keep]
    corr = sub.corr().abs()
    cols = list(sub.columns)
    dropped = set()
    for i, a in enumerate(cols):
        if a in dropped:
            continue
        col_a = corr.loc[a]
        for b in cols[i + 1:]:
            if b in dropped:
                continue
            if col_a.get(b, 0) >= corr_threshold:
                if a in protected and b in protected:
                    continue
                victim = b if b not in protected else a
                if victim not in protected:
                    dropped.add(victim)
    keep = [c for c in cols if c not in dropped]

    lost = [c for c in protected if c not in keep]
    assert not lost, f"PROTECTED FEATURE DROPPED: {lost}"
    if verbose:
        print(f"[filter] variance-dropped {n_var}, collinearity-dropped {len(dropped)}, "
              f"kept {len(keep)} ({len(protected & set(keep))} protected)")
    return keep


# ---------------------------------------------------------------------------
#  MODEL ZOO
# ---------------------------------------------------------------------------
def build_models(use_gpu=False, seed=0, fast=False):
    """Return {name: (constructor, supports_treeshap)}. Constructors are
    callables so each fold gets a fresh, unfitted estimator."""
    n_est = 400 if fast else 1200
    zoo = {}

    try:
        import lightgbm as lgb
        def _lgb():
            kw = dict(n_estimators=n_est, learning_rate=0.03, num_leaves=127,
                      min_child_samples=40, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1,
                      random_state=seed, verbosity=-1)
            if use_gpu:
                kw.update(device_type="gpu")
            return lgb.LGBMRegressor(**kw)
        zoo["LightGBM"] = (_lgb, True)
    except ImportError:
        pass

    try:
        import xgboost as xgb
        def _xgb():
            kw = dict(n_estimators=n_est, learning_rate=0.03, max_depth=8,
                      min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
                      reg_lambda=1.0, n_jobs=-1, random_state=seed,
                      tree_method="hist", enable_categorical=False)
            if use_gpu:
                kw.update(device="cuda")
            return xgb.XGBRegressor(**kw)
        zoo["XGBoost"] = (_xgb, True)
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor
        def _cat():
            kw = dict(iterations=n_est, learning_rate=0.03, depth=8,
                      l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                      allow_writing_files=False)
            if use_gpu:
                kw.update(task_type="GPU")
            return CatBoostRegressor(**kw)
        zoo["CatBoost"] = (_cat, True)
    except ImportError:
        pass

    if not zoo:
        from sklearn.ensemble import HistGradientBoostingRegressor
        zoo["HistGB"] = (lambda: HistGradientBoostingRegressor(
            max_iter=n_est, learning_rate=0.03, random_state=seed), False)
    return zoo


# ---------------------------------------------------------------------------
#  CROSS-VALIDATION  (county-grouped, in-fold imputation only)
# ---------------------------------------------------------------------------
def as_float(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to float64. Without this, integer-typed columns can
    trigger categorical splits in some boosters, which TreeSHAP cannot read."""
    return X.astype("float64")


def cv_evaluate(df, features, model_ctor, n_folds=8, seed=0, return_models=False):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    data = df.dropna(subset=[TARGET]).reset_index(drop=True)
    X = as_float(data[features]); y = data[TARGET].values; g = data[GROUP_COL].values
    oof = np.full(len(data), np.nan)
    fold_rows, models = [], []

    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, g), 1):
        med = X.iloc[tr].median(numeric_only=True)      # TRAIN FOLD ONLY
        Xtr, Xte = X.iloc[tr].fillna(med), X.iloc[te].fillna(med)
        m = model_ctor()
        m.fit(Xtr, y[tr])
        pred = m.predict(Xte)
        oof[te] = pred
        fold_rows.append(dict(fold=k, n_train=len(tr), n_test=len(te),
                              r2=r2_score(y[te], pred),
                              mae=mean_absolute_error(y[te], pred),
                              rmse=float(mean_squared_error(y[te], pred) ** 0.5)))
        if return_models:
            models.append((m, med))

    metrics = dict(r2=float(r2_score(y, oof)),
                   mae=float(mean_absolute_error(y, oof)),
                   rmse=float(mean_squared_error(y, oof) ** 0.5),
                   pearson_r=float(np.corrcoef(y, oof)[0, 1]),
                   within_1yr=float(np.mean(np.abs(y - oof) <= 1) * 100),
                   within_2yr=float(np.mean(np.abs(y - oof) <= 2) * 100),
                   n_obs=int(len(data)), n_features=len(features))
    folds = pd.DataFrame(fold_rows)
    metrics["r2_fold_mean"] = float(folds.r2.mean())
    metrics["r2_fold_std"] = float(folds.r2.std())
    out = dict(data=data, oof=oof, metrics=metrics, folds=folds, features=features)
    if return_models:
        out["models"] = models
    return out


def bootstrap_r2_ci(y, pred, n_boot=1000, seed=0):
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = [r2_score(y[b], pred[b])
            for b in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_tournament(df, features, n_folds=8, use_gpu=False, seed=0, fast=False,
                   verbose=True):
    """Fit every available learner under the identical protocol and return the
    per-learner table plus the name of the winner by out-of-fold R^2."""
    zoo = build_models(use_gpu=use_gpu, seed=seed, fast=fast)
    rows, results = [], {}
    
    for name, (ctor, shap_ok) in zoo.items():
        t0 = time.time()
        if verbose:
            print(f"  [tournament] evaluating {name:<10} ...", end="\r", flush=True)
            
        try:
            res = cv_evaluate(df, features, ctor, n_folds=n_folds, seed=seed)
            res["shap_ok"] = shap_ok
            results[name] = res
            m = res["metrics"]
            rows.append(dict(model=name, r2=m["r2"], mae=m["mae"], rmse=m["rmse"],
                             r2_fold_mean=m["r2_fold_mean"], r2_fold_std=m["r2_fold_std"],
                             seconds=round(time.time() - t0, 1), treeshap=shap_ok))
            if verbose:
                # Overwrite the evaluating line with the success metrics
                print(f"  [tournament] {name:<10} R2={m['r2']:.4f} "
                      f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                      f"({time.time()-t0:.0f}s)")
                
        except Exception as e:
            if verbose:
                # Extract just the first line of the error so it doesn't flood the logs
                err_msg = str(e).splitlines()[0] if str(e) else "Unknown Error"
                print(f"  [tournament] {name:<10} FAILED: {err_msg} (Skipping)")
            continue # Gracefully skip to the next model
            
    if not rows:
        raise RuntimeError("All models in the tournament failed!")
        
    table = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    best = table.iloc[0]["model"]
    if verbose:
        print(f"  [tournament] winner: {best} (R2={table.iloc[0]['r2']:.4f})")
    return table, results, best


# ---------------------------------------------------------------------------
#  ATTRIBUTION
# ---------------------------------------------------------------------------
def fit_full_model(df, features, model_ctor):
    """Refit on all rows with a global median fill, for downstream attribution
    and dependence analysis. Never used to report skill."""
    data = df.dropna(subset=[TARGET]).reset_index(drop=True)
    X = as_float(data[features])
    med = X.median(numeric_only=True)
    Xf = X.fillna(med)
    m = model_ctor()
    m.fit(Xf, data[TARGET].values)
    return m, Xf, data, med


def interventional_shap(df, features, model_ctor, n_folds=8, bg_size=1000,
                        sample_size=4000, seed=0, verbose=True):
    """Out-of-fold interventional TreeSHAP. Background and explainer are built
    from the TRAINING fold only, so attribution inherits the leakage control."""
    import shap
    from sklearn.model_selection import GroupKFold

    data = df.dropna(subset=[TARGET]).reset_index(drop=True)
    X = as_float(data[features]); y = data[TARGET].values; g = data[GROUP_COL].values
    acc = np.zeros(len(features))
    n_used = 0
    per_obs_idx, per_obs_val = [], []

    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, g), 1):
        med = X.iloc[tr].median(numeric_only=True)
        Xtr, Xte = X.iloc[tr].fillna(med), X.iloc[te].fillna(med)
        m = model_ctor()
        m.fit(Xtr, y[tr])
        bg = Xtr.sample(min(bg_size, len(Xtr)), random_state=seed)
        ex = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional")
        samp = Xte.sample(min(sample_size, len(Xte)), random_state=seed)
        sv = ex.shap_values(samp, check_additivity=False)
        acc += np.abs(sv).mean(axis=0)
        n_used += 1
        per_obs_idx.append(samp.index.values)
        per_obs_val.append(sv)
        if verbose:
            print(f"  [shap] fold {k}/{n_folds}")

    ranking = (pd.DataFrame({"feature": features, "mean_abs_shap": acc / max(n_used, 1)})
               .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    ranking["rank"] = ranking.index + 1
    ranking["modality"] = ranking.feature.map(modality_of)
    ranking["protected"] = ranking.feature.map(is_protected)

    idx = np.concatenate(per_obs_idx)
    vals = np.vstack(per_obs_val)
    per_obs = pd.DataFrame(vals, columns=features, index=idx)
    return ranking, per_obs, data


# ---------------------------------------------------------------------------
#  ARTEFACTS
# ---------------------------------------------------------------------------
class Artifacts:
    """Every run writes the same set of files, so downstream plotting and
    analysis code never has to re-fit anything."""

    def __init__(self, outdir: str, tag: str):
        self.dir = Path(outdir); self.dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.manifest = {}

    def _p(self, name): return self.dir / f"{self.tag}_{name}"

    def table(self, name, df, index=False):
        p = self._p(name if name.endswith(".csv") else name + ".csv")
        df.to_csv(p, index=index)
        self.manifest[name] = str(p)
        print(f"    [saved] {p.name}  ({len(df):,} rows)")
        return p

    def frame(self, name, df):
        p = self._p(name if name.endswith(".parquet") else name + ".parquet")
        df.to_parquet(p, index=False)
        self.manifest[name] = str(p)
        print(f"    [saved] {p.name}  ({df.shape[0]:,} x {df.shape[1]:,})")
        return p

    def model(self, name, obj):
        import joblib
        p = self._p(name if name.endswith(".joblib") else name + ".joblib")
        joblib.dump(obj, p, compress=3)
        self.manifest[name] = str(p)
        print(f"    [saved] {p.name}")
        return p

    def json(self, name, obj):
        p = self._p(name if name.endswith(".json") else name + ".json")
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        self.manifest[name] = str(p)
        print(f"    [saved] {p.name}")
        return p

    def close(self):
        self.json("manifest", self.manifest)
