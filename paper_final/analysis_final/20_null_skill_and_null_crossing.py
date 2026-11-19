#!/usr/bin/env python3
"""
20_null_skill_and_null_crossing.py -- what this machinery returns when there
is nothing to find.

    --only skill
    Permute the outcome three ways (across all rows, within each county,
    between whole counties) and rerun the production protocol unchanged. Each
    permutation destroys something different, so the three numbers together
    say where the explained variance lives (Section 3.1, Figure 2D).

    --only crossing
    Run the whole turning-point estimator, bootstrap and stability rule
    included, on attribution permuted against its own predictor, and on curves
    raised until no crossing exists. Report how often a crossing is found and
    how often it passes the rule (Section 3.5, Figure 7B).

Nothing in this script invents a value. A permutation reorders real numbers,
and no permuted column is written to any exported panel.

This is the final-paper copy of 20_null_calibration_and_multiplicity.py. Its
functions are unchanged; the census null and the multiplicity family, which
the final paper no longer reports, are not included.

READS   the modelling panel; the attribution surface of script 05
WRITES  ../derived*/null_skill.json
        ../derived*/null_crossing.json

    python3 20_null_skill_and_null_crossing.py
    python3 20_null_skill_and_null_crossing.py --only crossing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

SKILL_JSON = C.DERIVED / "null_skill.json"
CROSS_JSON = C.DERIVED / "null_crossing.json"

# ===========================================================================
#  SETTINGS
# ===========================================================================
NULL_CROSS_DRAWS = 40       # permutations of the attribution per variable.
NULL_CROSS_BOOT = 120       # bootstrap draws inside each permutation; a null
                            # needs only enough to decide whether the rule
                            # would pass. Checkpointed per variable.


def canonical_crossing():
    """Script 06's estimator, imported rather than reimplemented.

    A second implementation would give crossings differing in the second
    decimal from the published table and no reader could tell whether the
    difference was the null or the code.
    """
    path = os.path.join(HERE, "06_bootstrap_turning_points_and_gate.py")
    if not os.path.exists(path):
        raise SystemExit(
            "06_bootstrap_turning_points_and_gate.py is not beside this file.\n"
            "    This script runs that script's estimator on permuted\n"
            "    attribution, so the null is comparable with the published\n"
            "    table. Copy the whole analysis folder, not single scripts.")
    spec = importlib.util.spec_from_file_location("_s06", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
#  1. What the protocol returns when the outcome carries no signal
# ---------------------------------------------------------------------------
def null_skill(p, args):
    print("\n" + "=" * 74)
    print("  1. NULL CALIBRATION OF THE SCORE")
    print("=" * 74)
    print("  The production protocol, unchanged, on a permuted outcome. Three")
    print("  permutations, each destroying something different:")
    print()
    print("    full          every row's outcome reassigned at random. Nothing")
    print("                  survives. This is the true null.")
    print("    within county the years of each county reordered among")
    print("                  themselves. The county mean is preserved exactly,")
    print("                  the year-to-year pattern is destroyed.")
    print("    between county whole counties' outcome series swapped between")
    print("                  counties. The within-county pattern is preserved,")
    print("                  the link between a county's predictors and its")
    print("                  own level is destroyed.")
    print()
    print("  The three answers locate the explained variance. A high score")
    print("  under the within-county permutation and a null score under the")
    print("  between-county one would say the model is predicting the county's")
    print("  level and not its year-to-year movement, which is worth knowing")
    print("  and is not a flaw as long as the paper says so.")
    print()
    X, y, groups = p["X"], p["y"], p["groups"]
    rng = C.seed_everything(extra=20)
    arms = {}

    # full
    perm = rng.permutation(len(y))
    arms["full"] = y[perm]

    # within county: reorder each county's own values
    wy = y.copy()
    starts = {}
    for i, g in enumerate(groups):
        starts.setdefault(g, []).append(i)
    for g, idx in starts.items():
        idx = np.asarray(idx)
        wy[idx] = y[rng.permutation(idx)]
    arms["within_county"] = wy

    # between county: give each county another county's series, aligned by
    # position so a county with fewer years is not padded
    by = y.copy()
    keys = list(starts.keys())
    shuffled = list(rng.permutation(len(keys)))
    for a_i, b_i in enumerate(shuffled):
        ia, ib = np.asarray(starts[keys[a_i]]), np.asarray(starts[keys[b_i]])
        n = min(len(ia), len(ib))
        by[ia[:n]] = y[ib[:n]]
    arms["between_county"] = by

    out = {}
    for name, yy in arms.items():
        if args.only_arm and name != args.only_arm:
            continue
        tag = f"null_skill_{name}_seed{C.SEED + 20}"
        s = C.cv_score(X, np.asarray(yy, float), groups, tag,
                       n_folds=args.null_folds or None)
        out[name] = s
        print(f"    {name:<16} R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}   "
              f"({s['n_features']} columns)")
    out["note"] = ("Permuted outcomes. These are null distributions and are "
                   "not results about life expectancy.")
    out["n_folds"] = int(args.null_folds or C.N_FOLDS)
    out["seed_offset"] = 20
    # the published number, for the comparison the referee wants in one place
    try:
        pub = C.load_result(C.DERIVED / "baselines.json", ("satellite_model",))
        out["published_for_comparison"] = dict(
            satellite_model=pub.get("satellite_model"),
            naive_mean=pub.get("naive_mean"))
    except SystemExit:
        print("    (baselines.json absent; run script 03 for the published "
              "score to sit beside these)")
    print("\n    The published county-grouped score sits beside these in the")
    print("    file, so the comparison does not depend on anyone remembering")
    print("    it.")
    return out


# ---------------------------------------------------------------------------
#  2. Does the estimator find thresholds in noise?
# ---------------------------------------------------------------------------
def null_crossing(args):
    print("\n" + "=" * 74)
    print("  2. DOES THE CROSSING ESTIMATOR FIND THRESHOLDS IN NOISE?")
    print("=" * 74)
    print("  Two nulls, both using script 06's own estimator, bootstrap and")
    print("  stability gate:")
    print()
    print("    permuted     the attribution values reordered against the")
    print("                 predictor values. Any relation is destroyed; the")
    print("                 marginal distribution of both is untouched.")
    print("    shifted      the real attribution curve raised until it is")
    print("                 everywhere positive. There is no zero crossing to")
    print("                 find, by construction. An estimator that reports")
    print("                 one here is broken.")
    print()
    s06 = canonical_crossing()
    dep, feats = C.dependence_table()
    print(f"    {len(feats)} candidates in the attribution surface, "
          f"{len(dep):,} rows each")

    published = None
    try:
        published = C.load_table("turning_points").set_index("feature")
    except SystemExit:
        print("    (turning_points.csv absent, so the published verdict is "
              "not shown beside the null)")

    rng = C.seed_everything(extra=21)
    rows, per_var = [], {}
    n_draws_want = args.null_draws or NULL_CROSS_DRAWS
    n_boot_want = args.null_boot or NULL_CROSS_BOOT
    store = C.DERIVED / "checkpoints_null_crossing"
    store.mkdir(parents=True, exist_ok=True)
    stamp = f"d{n_draws_want}_b{n_boot_want}"
    for feat in feats:
        ck = store / f"{feat}__{stamp}.json"
        if ck.exists():
            try:
                rec = json.load(open(ck))
                per_var[str(feat)] = rec
                rows.append(rec)
                print(f"    {str(feat)[:44]:<44} from checkpoint: gate passed "
                      f"in {rec['permuted_gate_passed']}/{rec['n_draws']}")
                continue
            except Exception:
                pass
        x = dep[feat].to_numpy(float)
        v = dep["shap_" + feat].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(v)
        x, v = x[ok], v[ok]
        if len(x) < 2000:
            continue
        found_p, passed_p, points = 0, 0, []
        for d in range(args.null_draws or NULL_CROSS_DRAWS):
            r = s06.analyse(str(feat), x, v[rng.permutation(len(v))],
                            args.null_boot or NULL_CROSS_BOOT, C.N_BINS, rng)
            if r["point_estimate"] is not None and \
                    np.isfinite(r["point_estimate"]):
                found_p += 1
                points.append(float(r["point_estimate"]))
            if s06.passes(r, C.GATE_MIN_FOUND, C.GATE_MAX_WIDTH,
                          C.GATE_INSIDE):
                passed_p += 1
        # shifted: no crossing exists
        shifted = v - v.min() + abs(v).max() * 0.05
        rs = s06.analyse(str(feat), x, shifted,
                         args.null_boot or NULL_CROSS_BOOT, C.N_BINS, rng)
        shift_found = bool(rs["point_estimate"] is not None
                           and np.isfinite(rs["point_estimate"]))
        shift_passed = bool(s06.passes(rs, C.GATE_MIN_FOUND,
                                       C.GATE_MAX_WIDTH, C.GATE_INSIDE))
        n_draws = args.null_draws or NULL_CROSS_DRAWS
        rec = dict(feature=str(feat), n_draws=n_draws,
                   permuted_crossing_found=found_p,
                   permuted_gate_passed=passed_p,
                   permuted_pass_rate=passed_p / n_draws,
                   shifted_crossing_found=shift_found,
                   shifted_gate_passed=shift_passed)
        if points:
            rec["permuted_point_spread"] = float(np.std(points))
        if published is not None and feat in published.index:
            rec["published_stable"] = bool(published.loc[feat, "stable"])
            rec["published_point"] = float(
                published.loc[feat, "point_estimate"])
        per_var[str(feat)] = rec
        rows.append(rec)
        json.dump(rec, open(ck, "w"), indent=1, default=str)
        print(f"    {str(feat)[:44]:<44} crossing located in "
              f"{found_p:>3}/{n_draws}, gate passed in {passed_p:>3}/{n_draws}"
              + ("   shifted curve: FOUND ONE" if shift_passed else ""))

    n_draws = args.null_draws or NULL_CROSS_DRAWS
    total_pass = sum(r["permuted_gate_passed"] for r in rows)
    total_draw = sum(r["n_draws"] for r in rows)
    overall = dict(
        n_variables=len(rows), n_draws_each=n_draws,
        total_permutations=total_draw,
        total_gate_passes=total_pass,
        pass_rate=float(total_pass / total_draw) if total_draw else float("nan"),
        shifted_any_pass=int(sum(1 for r in rows if r["shifted_gate_passed"])),
        n_boot_per_draw=int(args.null_boot or NULL_CROSS_BOOT))
    print(f"\n    across {len(rows)} variables and {total_draw} permutations, "
          f"the gate passed {total_pass} times "
          f"({100 * overall['pass_rate']:.2f} per cent)")
    print(f"    on the shifted curves, where no crossing exists, the gate "
          f"passed for {overall['shifted_any_pass']} of {len(rows)} variables")
    print()
    print("    This is the number to quote when the estimator is challenged.")
    print("    The bootstrap below the gate is run at "
          f"{overall['n_boot_per_draw']} draws rather than")
    print(f"    the paper's {C.N_BOOT}, which is stated in the result file: a")
    print("    null needs only enough resolution to decide pass or fail.")
    return dict(per_variable=per_var, overall=overall,
                note="Permuted and shifted attribution. Null distributions, "
                     "not results."), rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: skill, crossing")
    ap.add_argument("--only-arm", default="",
                    help="with --only skill, one permutation: full, "
                         "within_county, between_county")
    ap.add_argument("--null-folds", type=int, default=0,
                    help="folds for the null refits. Default is the paper's "
                         f"{C.N_FOLDS}; a smaller number is a smoke test and "
                         "is written into the result file.")
    ap.add_argument("--null-draws", type=int, default=0,
                    help="permutations per variable (crossing)")
    ap.add_argument("--null-boot", type=int, default=0,
                    help="bootstrap draws inside each permutation")
    a = ap.parse_args()
    want = (["skill", "crossing"] if a.only == "all"
            else [s.strip() for s in a.only.split(",")])

    C.banner("20  what this machinery returns when there is nothing to find")
    C.seed_everything()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    written = []

    if "skill" in want:
        p = C.load_panel()
        print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")
        out = null_skill(p, a)
        json.dump(out, open(SKILL_JSON, "w"), indent=1, default=str)
        print(f"  [saved] {SKILL_JSON}")
        written.append(SKILL_JSON)

    if "crossing" in want:
        out, rows = null_crossing(a)
        json.dump(out, open(CROSS_JSON, "w"), indent=1, default=str)
        print(f"  [saved] {CROSS_JSON}")
        written.append(CROSS_JSON)

    C.write_manifest("20_null_skill_and_null_crossing.py",
                     inputs=[C.FRAME, C.MODEL_A],
                     outputs=written,
                     settings=dict(arms=want,
                                   NULL_CROSS_DRAWS=a.null_draws or NULL_CROSS_DRAWS,
                                   NULL_CROSS_BOOT=a.null_boot or NULL_CROSS_BOOT,
                                   null_folds=a.null_folds or C.N_FOLDS),
                     notes="Null calibration. Every number here is a null "
                           "distribution, not a result about life expectancy.")


if __name__ == "__main__":
    main()
