"""
13_spatial_holdout_and_attribution_structure.py
The four checks a methodological referee asks for, each written to a file.

WHY THIS SCRIPT EXISTS
----------------------
County-grouped cross-validation stops a county predicting itself. It does not
stop a county being predicted by its neighbours, and a referee who works on
spatial data will say so. These four checks answer that objection and the three
that usually follow it.

1.  STATE-LEVEL HOLDOUT.  Folds are grouped by state rather than by county, so
    every test county is in a state the model never saw. This is a harder test
    than county grouping and the drop from it is the honest measure of how much
    of the skill is spatial interpolation.

2.  RESIDUAL SPATIAL STRUCTURE.  Global Moran's I on the out-of-fold residuals
    over a four-nearest-neighbour graph of county centroids, with a permutation
    reference distribution so the value carries a p-value rather than standing
    alone.

3.  MULTICOLLINEARITY.  Maximum pairwise correlation and variance inflation
    among the leading variables. Interventional attribution is defined under
    correlation, but a referee will still want the number.

4.  INTERACTION STRUCTURE.  TreeSHAP interaction values among the leading
    variables, which say whether those variables act additively or whether the
    effect of one depends on the level of another. A turning point reported for
    an additive variable means something simpler than one reported for a
    variable in a strong interaction.

Every number this script prints is also written to a file. Nothing is typed
into the source: an earlier version of this script carried a previous run's
result in a print statement and skipped the computation, which is how a number
reaches a manuscript without anything having produced it.

READS   the modelling panel
        conus_counties_5070.gpkg   (county centroids, checks 2 only)
WRITES  derived/spatial_holdout.json
        derived/residual_spatial_structure.json
        derived/multicollinearity.json
        derived/attribution_interactions.json
TIME    state holdout 5 to 15 min, Moran 5 to 15 min, the other two under 5

    python3 13_spatial_holdout_and_attribution_structure.py
    python3 13_spatial_holdout_and_attribution_structure.py --only holdout
    python3 13_spatial_holdout_and_attribution_structure.py --only vif,interact
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
N_TOP = 15                 # leading variables used by the collinearity check
N_TOP_INTERACT = 6         # leading variables used by the interaction check
INTERACT_ROWS = 800        # rows for the interaction values, which are O(p^2)
# The variables whose turning points the paper reports. They are included in
# the interaction check whether or not they are among the leading few, because
# the question the check answers is about those turning points.
INTERACT_ALWAYS = ["prism_tdmean_min", "eng2_wetbulb_night",
                   "eng_night_cooling_eff"]
MORAN_NEIGHBOURS = 4       # k in the k-nearest-neighbour spatial weights
MORAN_PERMUTATIONS = 999   # reference distribution for the pseudo p-value
VIF_WARN = 10.0            # the conventional line for "severe"
# ===========================================================================

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as C

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
#  Fold-level checkpoints
# ---------------------------------------------------------------------------
#  The two cross-validated checks each fit the production model eight times,
#  which takes ten to thirty minutes. Each fold's out-of-fold predictions are
#  written as they are produced, so a run interrupted by a wall clock or a
#  suspended machine resumes at the first fold that is missing.

def fold_store(tag):
    d = C.DERIVED / "checkpoints_stress" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def cv_oof(X, y, split_groups, tag):
    """Out-of-fold predictions, through the shared checkpointed helper.

    The folds already written under the old per-script store are honoured, so
    a run started before this moved into _config does not refit them.
    """
    old = fold_store(tag)
    new = C.DERIVED / "checkpoints_cv" / tag
    new.mkdir(parents=True, exist_ok=True)
    for f in sorted(old.glob("fold*.npy")):
        dest = new / f.name
        if not dest.exists():
            dest.write_bytes(f.read_bytes())
    return C.cv_oof(X, y, split_groups, tag)


def fit_full(X, y):
    import xgboost as xgb
    m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
#  1. State-level holdout
# ---------------------------------------------------------------------------
def spatial_holdout(panel):
    from sklearn.metrics import r2_score, mean_absolute_error

    print("\n" + "=" * 74)
    print("  1. STATE-LEVEL HOLDOUT")
    print("=" * 74)
    X, y, groups = panel["X"], panel["y"], panel["groups"]
    fips = pd.to_numeric(pd.Series(groups), errors="coerce")
    if fips.isna().any():
        raise SystemExit("Some county codes are not numeric; cannot derive "
                         "a state code from them.")
    states = (fips // 1000).astype(int).to_numpy()
    n_states = len(np.unique(states))
    print(f"  {len(np.unique(fips)):,} counties in {n_states} states, "
          f"{C.N_FOLDS} folds grouped by state")

    t0 = time.time()
    oof = cv_oof(X, y, states, "state_holdout")

    out = dict(r2=float(r2_score(y, oof)),
               mae=float(mean_absolute_error(y, oof)),
               n_states=int(n_states), n_folds=int(C.N_FOLDS),
               n_obs=int(len(y)), seconds=round(time.time() - t0, 1))
    print(f"\n  state-grouped out-of-fold R2  {out['r2']:.4f}")
    print(f"  state-grouped out-of-fold MAE {out['mae']:.4f} years")
    print("  Compare against the county-grouped figure in the performance")
    print("  table. The difference between them is the part of the skill that")
    print("  county grouping alone does not control for.")
    p = C.DERIVED / "spatial_holdout.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"[saved] {p}")
    return [p]


# ---------------------------------------------------------------------------
#  2. Residual spatial structure
# ---------------------------------------------------------------------------
def centroids():
    import geopandas as gpd
    gpkg = os.path.join(HERE, "conus_counties_5070.gpkg")
    if not os.path.exists(gpkg):
        raise SystemExit(f"County boundaries missing: {gpkg}")
    g = gpd.read_file(gpkg, layer="county")
    if "fips" not in g.columns:
        g["fips"] = pd.to_numeric(g["STATE"] + g["COUNTY"], errors="coerce")
    c = g.geometry.centroid
    return pd.DataFrame({"fips": pd.to_numeric(g["fips"], errors="coerce"),
                         "x": c.x.to_numpy(), "y": c.y.to_numpy()}
                        ).dropna().astype({"fips": int})


def residual_structure(panel):
    from scipy.spatial import cKDTree

    print("\n" + "=" * 74)
    print("  2. RESIDUAL SPATIAL STRUCTURE")
    print("=" * 74)
    X, y, groups = panel["X"], panel["y"], panel["groups"]

    print("  out-of-fold predictions, county-grouped")
    oof = cv_oof(X, y, groups, "county_oof")

    # one residual per county, averaged over its years, because the spatial
    # weights are between counties and not between county-years
    res = (pd.DataFrame({"fips": pd.to_numeric(pd.Series(groups)),
                         "r": y - oof})
           .groupby("fips", as_index=False)["r"].mean())
    xy = centroids().merge(res, on="fips")
    print(f"  {len(xy):,} counties placed")

    coords = xy[["x", "y"]].to_numpy()
    tree = cKDTree(coords)
    _, nb = tree.query(coords, k=MORAN_NEIGHBOURS + 1)
    nb = nb[:, 1:]

    z = xy["r"].to_numpy() - xy["r"].mean()
    denom = float(np.sum(z ** 2))

    def moran(vals):
        lag = vals[nb].mean(axis=1)
        return float(np.sum(vals * lag) / denom)

    obs = moran(z)
    rng = C.seed_everything()
    null = np.array([moran(rng.permutation(z))
                     for _ in range(MORAN_PERMUTATIONS)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1)
              / (MORAN_PERMUTATIONS + 1))

    out = dict(morans_i=obs, p_value=p, n_counties=int(len(xy)),
               neighbours=MORAN_NEIGHBOURS,
               permutations=MORAN_PERMUTATIONS,
               null_mean=float(null.mean()), null_sd=float(null.std()))
    print(f"\n  Moran's I on the county mean residual   {obs:.4f}")
    print(f"  permutation pseudo p-value              {p:.4f}")
    print(f"  null distribution                       mean {null.mean():+.4f}, "
          f"sd {null.std():.4f}")
    pth = C.DERIVED / "residual_spatial_structure.json"
    json.dump(out, open(pth, "w"), indent=1)
    print(f"[saved] {pth}")
    return [pth]


# ---------------------------------------------------------------------------
#  3. Multicollinearity among the leading variables
# ---------------------------------------------------------------------------
def multicollinearity(panel):
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 74)
    print("  3. MULTICOLLINEARITY AMONG THE LEADING VARIABLES")
    print("=" * 74)
    X, y = panel["X"], panel["y"]
    rank = C.load_table("shap_ranking").sort_values("rank")
    top = [c for c in rank.feature.head(N_TOP) if c in X.columns]
    print(f"  the {len(top)} leading variables by attribution")

    Z = StandardScaler().fit_transform(X[top].fillna(X[top].median()))
    corr = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)

    # The variance inflation factor is 1/(1-R2) of each column on the others.
    # Solving that by ordinary least squares overflows when two columns are
    # nearly identical, which is exactly the case this check exists to measure.
    # The correlation matrix inverse gives the same quantity and is stable: the
    # kth diagonal element of the inverse correlation matrix IS the kth VIF.
    # A pseudo-inverse keeps it finite when the matrix is singular.
    full = np.corrcoef(Z, rowvar=False)
    inv = np.linalg.pinv(full, rcond=1e-12)
    vifs = {col: float(abs(inv[k, k])) for k, col in enumerate(top)}

    vals = np.array([v for v in vifs.values() if np.isfinite(v)])
    out = dict(n_variables=len(top),
               max_abs_correlation=float(abs(corr[i, j])),
               max_correlation_pair=[top[i], top[j]],
               median_vif=float(np.median(vals)),
               max_vif=float(vals.max()),
               max_vif_variable=max(vifs, key=lambda c: vifs[c]),
               n_over_warn=int((vals > VIF_WARN).sum()),
               warn_level=VIF_WARN, vif=vifs)
    print(f"  largest absolute pairwise correlation   {out['max_abs_correlation']:.3f}")
    print(f"      between {top[i]} and {top[j]}")
    print(f"  median variance inflation factor        {out['median_vif']:.2f}")
    print(f"  largest                                 {out['max_vif']:.2f} "
          f"({out['max_vif_variable']})")
    print(f"  variables above the conventional {VIF_WARN:.0f}        "
          f"{out['n_over_warn']} of {len(top)}")
    p = C.DERIVED / "multicollinearity.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"[saved] {p}")
    return [p]


# ---------------------------------------------------------------------------
#  4. Interaction structure
# ---------------------------------------------------------------------------
def interactions(panel):
    # XGBoost's own pred_interactions is used rather than the shap package's
    # TreeExplainer. They compute the same quantity, and the package's XGBoost
    # loader fails against boosters whose base_score is serialised as a vector,
    # which is how recent XGBoost writes it. Calling the booster directly has
    # no version coupling between two libraries to go wrong.
    import xgboost as xgb

    print("\n" + "=" * 74)
    print("  4. INTERACTION STRUCTURE AMONG THE LEADING VARIABLES")
    print("=" * 74)
    X, y = panel["X"], panel["y"]
    rank = C.load_table("shap_ranking").sort_values("rank")
    top = [c for c in rank.feature.head(N_TOP_INTERACT) if c in X.columns]
    for c in INTERACT_ALWAYS:
        if c in X.columns and c not in top:
            top.append(c)
    print(f"  {len(top)} variables, {INTERACT_ROWS} county-years")
    print("  Interaction values are quadratic in the number of columns, so the")
    print("  model is refitted on these columns alone. Interaction strength is")
    print("  therefore reported relative to this reduced model, which is what")
    print("  the question is about: do these variables act on each other.")

    rng = C.seed_everything()
    idx = np.sort(rng.choice(len(y), size=min(INTERACT_ROWS, len(y)),
                             replace=False))
    Xi = X[top].iloc[idx]
    m = fit_full(X[top], y)
    t0 = time.time()
    dm = xgb.DMatrix(Xi, feature_names=list(Xi.columns))
    iv = m.get_booster().predict(dm, pred_interactions=True)
    # the last row and column are the bias term, which is not a variable
    iv = np.asarray(iv)[:, :len(top), :len(top)]

    main = np.abs(np.einsum("nii->ni", iv)).mean(axis=0)
    pairs = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            pairs.append(dict(a=top[a], b=top[b],
                              strength=float(np.abs(iv[:, a, b]).mean() * 2)))
    pairs.sort(key=lambda d: -d["strength"])
    total_main = float(main.sum())
    total_inter = float(sum(p["strength"] for p in pairs))
    share = total_inter / (total_main + total_inter) if total_main else np.nan

    out = dict(variables=top, n_rows=int(len(idx)),
               reduced_model=True, n_columns_in_model=len(top),
               main_effect_total=total_main,
               interaction_total=total_inter,
               interaction_share=float(share),
               strongest_pair=pairs[0], pairs=pairs[:10],
               main_effects={c: float(v) for c, v in zip(top, main)},
               seconds=round(time.time() - t0, 1))
    print(f"\n  strongest pair   {pairs[0]['a']}")
    print(f"                   {pairs[0]['b']}")
    print(f"                   {pairs[0]['strength']:.4f} years")
    print(f"  interaction share of total attribution   {100 * share:.1f} per cent")
    print("  This is the share within this reduced model. Interactions are")
    print("  larger in a model given only these columns than in the full one,")
    print("  because structure the full model attributes to other variables")
    print("  has nowhere else to go. Read it as an upper bound on how far")
    print("  these variables act on one another, not as an estimate of the")
    print("  share in the production model.")
    p = C.DERIVED / "attribution_interactions.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"[saved] {p}")
    return [p]


CHECKS = {"holdout": spatial_holdout, "moran": residual_structure,
          "vif": multicollinearity, "interact": interactions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: " + ", ".join(CHECKS))
    a = ap.parse_args()
    want = list(CHECKS) if a.only == "all" else [s.strip()
                                                 for s in a.only.split(",")]
    bad = [w for w in want if w not in CHECKS]
    if bad:
        raise SystemExit(f"unknown check(s): {', '.join(bad)}\n"
                         f"    known: {', '.join(CHECKS)}")

    C.banner("13  spatial holdout and attribution structure")
    C.seed_everything()
    panel = C.load_panel()
    out = []
    for w in want:
        out += CHECKS[w](panel)
    C.write_manifest(__file__, [], out, notes=f"checks={','.join(want)}")
    print("\n" + "=" * 74)
    print(f"  {len(out)} result file(s) written to {C.DERIVED.name}/")
    print("=" * 74)


if __name__ == "__main__":
    main()
