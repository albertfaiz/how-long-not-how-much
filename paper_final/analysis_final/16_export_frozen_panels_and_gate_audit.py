"""
16_export_frozen_panels_and_gate_audit.py
Two frozen modelling panels, and a measurement of what the admission gate is
worth.

WHY THIS SCRIPT EXISTS
----------------------
Every script in this folder takes its column list from the fitted model
artefact, so every one of them inherits the coverage gate and the variance and
correlation filters. That is true and it is not obvious, and an analysis that
started from the master CSV instead would silently include the 776 columns the
gate removed. Scores from such a run are higher and mean nothing, because the
removed columns include ones available in only part of the window and ones that
are near-duplicates of a column already present.

This script makes that impossible to get wrong twice. It writes the admitted
panel out as a flat CSV with the key columns beside it, so an analysis can start
from a file that already has the gate applied, and it writes a manifest naming
every column with the checksum of the file it came from.

It then measures the thing the frozen panel protects against: the same learner,
the same folds and the same seed, fitted on the gated set and on the ungated
candidate set, so the difference is a number rather than an intuition.

READS   the reference frame, the fitted model, the admission record
WRITES  derived/panel_all_admitted.csv          fips, year, outcome + admitted
        derived/panel_environment_only.csv      the same minus any non-physical
        derived/panel_manifest.json             columns, checksums, provenance
        derived/gate_audit.json                 what the gate costs in R2
TIME    export under two minutes; the audit adds 20 to 50

    python3 16_export_frozen_panels_and_gate_audit.py
    python3 16_export_frozen_panels_and_gate_audit.py --skip-audit
    python3 16_export_frozen_panels_and_gate_audit.py --audit-only
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
# Three panels are written, each a strict subset of the one before it.
#
#   all_admitted      every column the fitted model uses
#   environment_only  minus anything socioeconomic, clinical or economic
#   satellite_only    minus the interpolated and reanalysis products too, so
#                     what is left is measured from orbit
#
# The second is a guard rather than a filter: no socioeconomic column clears
# the coverage gate in this run, so it currently equals the first. If a future
# gate admits one, the panel still will not contain it.
NON_PHYSICAL_MODALITIES = {
    "Socioeconomic (SDOH)", "Health outcomes", "Macro economics",
}
NON_PHYSICAL_PATTERNS = [
    r"^sdoh_", r"^acs_", r"^census_", r"^svi_", r"mortality", r"_death",
    r"lifeexpect", r"^econ_", r"poverty", r"income", r"uninsured",
]

# Modalities that are measured from orbit. PRISM, gridMET and TerraClimate are
# interpolations of surface station records and ERA5 and CAMS are reanalyses
# that assimilate observations into a model, so none of the four is a satellite
# measurement however much satellite data informs it. The engineered layers are
# excluded from this panel because each is built from more than one source and
# a single provenance cannot be claimed for them.
SATELLITE_MODALITIES = {
    "MODIS land-surface temperature", "MODIS vegetation", "Landsat optical",
    "Night-time lights", "Canopy structure", "Surface water",
    "Cropland and agriculture", "Fine particulate matter",
    "Human modification", "Sentinel-1 radar", "Sentinel-2 optical",
    "Sentinel-5P air quality",
}

# The audit's arms. "gated" is the production set. "ungated" is every numeric
# candidate column present in the frame, which is what an analysis that skipped
# the gate would see.
AUDIT_ARMS = ["gated", "ungated", "gated_no_protected"]
AUDIT_FOLDS = None            # None uses C.N_FOLDS
# ===========================================================================

import argparse
import gc
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as C

NON_PHYSICAL_RE = re.compile("|".join(NON_PHYSICAL_PATTERNS), re.I)


def mod_of(gate, col):
    hit = gate.loc[gate.feature == col, "modality"]
    return str(hit.iloc[0]) if len(hit) else "unknown"


def gate_record():
    """The admission record, with the modality of every candidate column."""
    g = C.load_table("feature_gate")
    need = {"feature", "admitted"}
    if not need <= set(g.columns):
        raise SystemExit(
            f"The admission record has columns {list(g.columns)}; this script "
            f"needs at least {sorted(need)}.")
    return g


def split_columns(model_features, gate):
    """The three nested column lists, in the model's own order."""
    mod = dict(zip(gate.feature, gate.get("modality", pd.Series(dtype=str))))
    env, sat = [], []
    for c in model_features:
        if mod.get(c) in NON_PHYSICAL_MODALITIES or NON_PHYSICAL_RE.search(c):
            continue
        env.append(c)
        if mod.get(c) in SATELLITE_MODALITIES:
            sat.append(c)
    return list(model_features), env, sat


def export(panel, model_features, gate, args):
    C.banner("16  frozen panels")
    df = panel["df"]
    X = panel["X"]
    keys = [C.GROUP, C.YEAR, C.TARGET]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise SystemExit(f"The frame has no {missing}; cannot key the panel.")

    admitted, env, sat = split_columns(model_features, gate)
    print(f"  {len(admitted)} admitted columns")
    print(f"  {len(env)} after removing socioeconomic, clinical and economic")
    print(f"  {len(sat)} measured from orbit")
    if len(env) == len(admitted):
        print("  The first two are identical: no socioeconomic column clears")
        print("  the coverage gate in this run, which is the environment-only")
        print("  claim checked at the level of the columns themselves.")
    by_mod = {}
    for c in sat:
        by_mod[mod_of(gate, c)] = by_mod.get(mod_of(gate, c), 0) + 1
    for k, v in sorted(by_mod.items(), key=lambda kv: -kv[1]):
        print(f"      {v:>4}  {k}")

    out = []
    for name, cols in (("panel_all_admitted", admitted),
                       ("panel_environment_only", env),
                       ("panel_satellite_only", sat)):
        frame = pd.concat([df[keys].reset_index(drop=True),
                           X[cols].reset_index(drop=True)], axis=1)
        frame = frame.rename(columns={C.GROUP: "fips", C.YEAR: "year",
                                      C.TARGET: "life_expectancy"})
        p = C.DERIVED / f"{name}.csv"
        frame.to_csv(p, index=False)
        print(f"  [saved] {p.name}  {frame.shape[0]:,} rows x "
              f"{frame.shape[1]} columns")
        out.append(p)

    man = dict(
        source_frame=str(C.FRAME), source_frame_sha256=C.sha256(C.FRAME),
        source_model=str(C.MODEL_A), source_model_sha256=C.sha256(C.MODEL_A),
        n_candidates=int(len(gate)),
        n_gate_admitted=int(gate.admitted.sum()),
        n_modelled=len(admitted),
        n_environment_only=len(env),
        n_satellite_only=len(sat),
        window=[int(panel["window"][0]), int(panel["window"][1])],
        n_rows=int(len(df)), n_counties=int(df[C.GROUP].nunique()),
        key_columns=["fips", "year", "life_expectancy"],
        columns_all_admitted=admitted,
        columns_environment_only=env,
        columns_satellite_only=sat,
        satellite_modalities=sorted(SATELLITE_MODALITIES),
        note=("These columns are the fitted model's own feature list. They have "
              "already passed the coverage gate and the variance and "
              "correlation filters of the main run. An analysis starting from "
              "the master CSV instead would see every candidate column and is "
              "not comparable with anything in this paper."),
    )
    p = C.DERIVED / "panel_manifest.json"
    json.dump(man, open(p, "w"), indent=1)
    print(f"  [saved] {p.name}")
    out.append(p)
    return out


def audit(panel, model_features, gate, args):
    """What the gate is worth, measured on the same folds with the same seed."""
    import pyarrow.parquet as pq

    C.banner("16  what the admission gate is worth")
    y, groups = panel["y"], panel["groups"]
    n_folds = AUDIT_FOLDS or C.N_FOLDS

    # the ungated arm: every numeric candidate column in the frame
    sch = pq.read_schema(C.FRAME)
    drop = {C.GROUP, C.YEAR, C.TARGET}
    cand = [n for n in sch.names if n not in drop]
    print(f"  the frame carries {len(cand)} candidate columns")
    print(f"  the gate and the filters leave {len(model_features)}")

    protected = set()
    if "protected" in gate.columns:
        protected = set(gate.loc[gate.protected.astype(bool), "feature"])
    print(f"  {len(protected & set(model_features))} of the modelled columns "
          f"are protected")

    def fit_arm(cols, label, frame=None):
        # Per-fold checkpoints live in _config.cv_score, so an arm interrupted
        # part way resumes at the fold it reached rather than at fold one.
        ck = C.DERIVED / "checkpoints_gate_audit" / f"{label}.json"
        ck.parent.mkdir(parents=True, exist_ok=True)
        if ck.exists():
            done = json.load(open(ck))
            print(f"    {label:<22} already done: R2 {done['r2']:.4f}")
            return done
        Xa = frame if frame is not None else panel["X"][cols]
        r = C.cv_score(Xa, y, groups, f"gate_{label}", n_folds=n_folds)
        r["label"] = label
        json.dump(r, open(ck, "w"), indent=1)
        print(f"    {label:<22} R2 {r['r2']:.4f}  [checkpointed]", flush=True)
        return r

    arms = {}
    if "gated" in AUDIT_ARMS:
        arms["gated"] = fit_arm(model_features, "gated")
    if "ungated" in AUDIT_ARMS:
        lo, hi = panel["window"]
        raw = pq.read_table(C.FRAME,
                            columns=sorted(set(cand + list(drop)))).to_pandas()
        raw = raw[(raw[C.YEAR] >= lo) & (raw[C.YEAR] <= hi)
                  & raw[C.TARGET].notna()].reset_index(drop=True)
        Xu = raw[cand].apply(pd.to_numeric, errors="coerce")
        Xu = Xu.loc[:, Xu.notna().any()]
        Xu = Xu.fillna(Xu.median()).astype("float32")
        print(f"  ungated arm: {Xu.shape[1]} numeric columns after coercion")
        arms["ungated"] = fit_arm(None, "ungated", frame=Xu)
        del raw, Xu
        gc.collect()
    if "gated_no_protected" in AUDIT_ARMS and protected:
        keep = [c for c in model_features if c not in protected]
        arms["gated_no_protected"] = fit_arm(keep, "gated_no_protected")

    print("\n  out of fold, identical county-grouped folds:")
    for k, r in arms.items():
        print(f"      {k:<22} R2 {r['r2']:.4f}   MAE {r['mae']:.4f}   "
              f"{r['n_features']} columns")
    if "gated" in arms and "ungated" in arms:
        d = arms["ungated"]["r2"] - arms["gated"]["r2"]
        print(f"\n  skipping the gate changes out-of-fold R2 by {d:+.4f}, on "
              f"{arms['ungated']['n_features'] - arms['gated']['n_features']:+d} "
              f"columns.")
        print("  A higher score there is not a better model. The removed")
        print("  columns include ones present for only part of the window and")
        print("  ones that duplicate a column already in the set, so the extra")
        print("  skill is partly the model learning which years and which")
        print("  duplicates it is looking at.")
    if "gated" in arms and "gated_no_protected" in arms:
        d = arms["gated"]["r2"] - arms["gated_no_protected"]["r2"]
        print(f"\n  the protected columns are worth {d:+.4f} R2 between them.")

    p = C.DERIVED / "gate_audit.json"
    json.dump(dict(arms=arms, n_candidates_in_frame=len(cand),
                   n_modelled=len(model_features),
                   n_protected_modelled=len(protected & set(model_features)),
                   n_folds=n_folds), open(p, "w"), indent=1)
    print(f"[saved] {p}")
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args()

    C.seed_everything()
    gate = gate_record()
    panel = C.load_panel()
    feats = panel["features"]
    out = []
    if not a.audit_only:
        out += export(panel, feats, gate, a)
    if not a.skip_audit:
        out += audit(panel, feats, gate, a)
    C.write_manifest(__file__, [], out)
    print("\n" + "=" * 74)
    print("  The two panels carry the gate with them. An analysis that starts")
    print("  from panel_all_admitted.csv cannot skip it by accident.")
    print("=" * 74)


if __name__ == "__main__":
    main()
