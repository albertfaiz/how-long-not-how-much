"""
03_baselines_five_reference_models.py
The floor the environment-only model has to clear, computed from scratch.

WHY THIS EXISTS
---------------
An R-squared means nothing on its own. Figure 2D reports five reference
predictors scored on the identical county-grouped folds, and the one that
matters is the fourth: an interpolator that is handed the measured life
expectancy of the four nearest TRAINING counties, which the model never sees.
If the model cannot beat that, it is a slow way of drawing a map.

THE FIVE
  1  training-fold mean                 no features at all
  2  training-fold mean for that year   no features, but allowed the trend
  3  night-time lights only             the usual single-product stand-in
  4  latitude, longitude and year       position, measuring nothing
  5  four-nearest-neighbour             the outcome itself, next door

Every one uses the same GroupKFold(n_splits=8) partition on FIPS as the main
fit, so the comparison is paired rather than approximate.

COUNTY CENTROIDS
  Baselines 4 and 5 need a latitude and longitude per county. The script looks
  for a cached copy first, and only then tries the Census gazetteer over the
  network. If it has no cache and no network it says so and skips those two
  rather than failing the whole run; run it again once you are online and the
  cache will be written for good.

READS   the reference frame, the fitted model
WRITES  derived/baselines.json
        derived/baselines.csv
        derived/county_centroids.csv   (the cache, written once)
TIME    15 to 40 minutes on a laptop. Baselines 3 and 4 fit a model each; the
        rest are arithmetic. Pass --only to run one at a time.

    python3 03_baselines_five_reference_models.py
    python3 03_baselines_five_reference_models.py --only knn,coords
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
KNN_NEIGHBOURS = 4        # how many training counties the interpolator sees
KNN_SAME_YEAR = True      # borrow only from the same year, as the paper says
NTL_PREFIX = "ntl_"       # how night-time-lights columns are named
CENTROID_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
                "2020_Gazetteer/2020_Gaz_counties_national.zip")
# ===========================================================================

import argparse
import io
import json
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import _config as C

ORDER = ["naive_mean", "naive_year_mean", "ntl_only", "coords_only",
         "knn4_spatial"]
LABEL = {
    "naive_mean": ("Training-fold mean", "no features"),
    "naive_year_mean": ("Training-fold mean for that year", "no features"),
    "ntl_only": ("Night-time lights only", "{n} features"),
    "coords_only": ("Latitude, longitude, year", "3 features, not observations"),
    "knn4_spatial": ("Four-nearest-neighbour interpolation",
                     "needs the outcome next door"),
}


def folds(X, y, g):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=C.N_FOLDS).split(X, y, g))


def cv_fit_score(X, y, g, tag):
    """Fit the production learner on the production folds and score OOF."""
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score
    oof = np.full(len(y), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(folds(X, y, g), 1):
        m = xgb.XGBRegressor(n_jobs=C.threads(), **C.XGB_PARAMS)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
        print(f"    [{tag}] fold {k}/{C.N_FOLDS}  {time.time()-t0:6.0f}s",
              flush=True)
    return dict(r2=float(r2_score(y, oof)),
                mae=float(mean_absolute_error(y, oof)),
                n_features=int(X.shape[1]), seconds=round(time.time() - t0))


def centroids():
    """Cached first, network second, honest failure third."""
    cache = C.DERIVED / "county_centroids.csv"
    if cache.exists():
        print(f"  centroids from cache: {cache.name}")
        return pd.read_csv(cache)
    print("  no centroid cache; fetching the Census gazetteer ...")
    try:
        with urllib.request.urlopen(CENTROID_URL, timeout=60) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = [n for n in z.namelist() if n.endswith(".txt")][0]
            raw = pd.read_csv(z.open(name), sep="\t", dtype={"GEOID": str})
    except Exception as e:
        print(f"  could not fetch centroids: {type(e).__name__}: {e}")
        return None
    raw.columns = [c.strip() for c in raw.columns]
    df = pd.DataFrame({"fips": raw["GEOID"].astype(int),
                       "lat": raw["INTPTLAT"].astype(float),
                       "lon": raw["INTPTLONG"].astype(float)})
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  centroids cached to {cache.name}; this will not download again")
    return df


# ---------------------------------------------------------------- baselines
def b_naive(p, R):
    from sklearn.metrics import mean_absolute_error, r2_score
    X, y, g, df = p["X"], p["y"], p["groups"], p["df"]
    oof = np.full(len(y), np.nan)
    for tr, te in folds(X, y, g):
        oof[te] = y[tr].mean()
    R["naive_mean"] = dict(r2=float(r2_score(y, oof)),
                           mae=float(mean_absolute_error(y, oof)),
                           n_features=0)
    oof2 = np.full(len(y), np.nan)
    yr = df[C.YEAR].to_numpy()
    for tr, te in folds(X, y, g):
        mu = pd.Series(y[tr]).groupby(yr[tr]).mean()
        oof2[te] = [mu.get(v, y[tr].mean()) for v in yr[te]]
    R["naive_year_mean"] = dict(r2=float(r2_score(y, oof2)),
                                mae=float(mean_absolute_error(y, oof2)),
                                n_features=0)


def b_ntl(p, R):
    cols = [c for c in p["features"] if c.startswith(NTL_PREFIX)]
    if not cols:
        print(f"  no columns start with {NTL_PREFIX!r}; skipping")
        return
    print(f"  night-time lights: {len(cols)} columns")
    R["ntl_only"] = cv_fit_score(p["X"][cols], p["y"], p["groups"], "ntl")


def b_coords(p, R):
    cen = centroids()
    if cen is None:
        return
    d = p["df"][[C.GROUP, C.YEAR, C.TARGET]].merge(cen, on="fips", how="left")
    ok = d.lat.notna().to_numpy()
    X = d.loc[ok, ["lat", "lon", C.YEAR]].astype("float32").reset_index(drop=True)
    R["coords_only"] = cv_fit_score(X, d.loc[ok, C.TARGET].to_numpy(),
                                    d.loc[ok, C.GROUP].to_numpy(), "coords")


def b_knn(p, R):
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.neighbors import NearestNeighbors
    cen = centroids()
    if cen is None:
        return
    d = p["df"][[C.GROUP, C.YEAR, C.TARGET]].merge(cen, on="fips", how="left")
    d = d[d.lat.notna()].reset_index(drop=True)
    y, g = d[C.TARGET].to_numpy(), d[C.GROUP].to_numpy()
    yr = d[C.YEAR].to_numpy()
    P = np.radians(d[["lat", "lon"]].to_numpy())
    oof = np.full(len(d), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(folds(P, y, g), 1):
        years = np.unique(yr[te]) if KNN_SAME_YEAR else [None]
        for v in years:
            itr = tr[yr[tr] == v] if KNN_SAME_YEAR else tr
            ite = te[yr[te] == v] if KNN_SAME_YEAR else te
            if len(itr) < KNN_NEIGHBOURS + 1 or not len(ite):
                continue
            nn = NearestNeighbors(n_neighbors=KNN_NEIGHBOURS,
                                  metric="haversine").fit(P[itr])
            dist, idx = nn.kneighbors(P[ite])
            w = 1.0 / np.maximum(dist, 1e-6)
            oof[ite] = (y[itr][idx] * w).sum(1) / w.sum(1)
        print(f"    [knn] fold {k}/{C.N_FOLDS}  {time.time()-t0:5.0f}s",
              flush=True)
    m = np.isfinite(oof)
    R["knn4_spatial"] = dict(r2=float(r2_score(y[m], oof[m])),
                             mae=float(mean_absolute_error(y[m], oof[m])),
                             n=int(m.sum()), n_features=0)


RUNNERS = {"naive": b_naive, "ntl": b_ntl, "coords": b_coords, "knn": b_knn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma-separated: naive,ntl,coords,knn")
    a = ap.parse_args()

    C.banner("03  five reference baselines")
    C.seed_everything()
    p = C.load_panel()
    print(f"panel: {len(p['df']):,} county-years, {p['X'].shape[1]} features\n")

    out = C.DERIVED / "baselines.json"
    R = json.load(open(out)) if out.exists() else {}
    if R:
        print(f"resuming; already have: {', '.join(sorted(R))}\n")

    picked = list(RUNNERS) if a.only == "all" else [s.strip() for s in
                                                    a.only.split(",")]
    for name in picked:
        print(f"[{name}]")
        RUNNERS[name](p, R)
        C.DERIVED.mkdir(parents=True, exist_ok=True)
        json.dump(R, open(out, "w"), indent=1)
        print()

    # the model itself, for the same table
    try:
        met = json.load(open(C.SUMMARY_A))
        R["satellite_model"] = dict(r2=float(met.get("r2")),
                                    mae=float(met.get("mae")),
                                    n_features=int(met.get("n_features", 0)))
    except Exception:
        pass

    rows = []
    for k in ORDER + ["satellite_model"]:
        if k not in R:
            continue
        nm, note = LABEL.get(k, ("Satellite model", "{n} features"))
        rows.append(dict(key=k, label=nm,
                         note=note.format(n=R[k].get("n_features", 0)),
                         r2=R[k]["r2"], mae=R[k]["mae"]))
    tab = pd.DataFrame(rows)
    print("=" * 74)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 74)
    if "knn4_spatial" in R and "satellite_model" in R:
        d = R["satellite_model"]["r2"] - R["knn4_spatial"]["r2"]
        print(f"  model minus interpolator: {d:+.4f} in R-squared")
        print("  The interpolator is given the outcome in neighbouring")
        print("  counties. The model is given no outcome anywhere.")

    csv = C.DERIVED / "baselines.csv"
    tab.to_csv(csv, index=False)
    json.dump(R, open(out, "w"), indent=1)
    print(f"\n[saved] {out.name}, {csv.name}  in derived/")
    C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [out, csv],
                     settings=dict(KNN_NEIGHBOURS=KNN_NEIGHBOURS,
                                   KNN_SAME_YEAR=KNN_SAME_YEAR),
                     notes=f"ran: {picked}")


if __name__ == "__main__":
    main()
