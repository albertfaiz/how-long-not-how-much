#!/usr/bin/env python3
"""
20_null_calibration_and_multiplicity.py -- what this machinery returns when
there is nothing to find, and what survives counting the tests.

READS   the modelling panel; the attribution surface of script 05; the census
        of script 12; the information census of script 14; and every result
        file in the run folder that records a p-value
WRITES  ../derived*/null_skill.json
        ../derived*/null_crossing.json
        ../derived*/null_census.json
        ../derived*/multiplicity_control.json
        ../derived*/multiplicity_per_field.csv

FOUR OBJECTIONS, FOUR ARMS
--------------------------
    --only skill
        "With 581 columns, 52,309 rows and only 3,077 distinct counties,
         0.67 may be roughly what your protocol returns on noise."
    Permute the outcome three ways and rerun the production protocol
    unchanged. Each permutation destroys something different, so the three
    numbers together say where the explained variance lives.

    --only crossing
        "A smoothing spline crosses zero somewhere in anything. Your eleven
         turning points are a property of the estimator."
    Run the whole estimator, bootstrap and stability gate included, on
    attribution that has been permuted against the predictor. Report how often
    it returns a crossing at all and how often that crossing passes the gate.

    --only census
        "Thirty-eight of forty-seven sounds impressive. What is the null?"
    Shuffle which of each field's two columns is called central, recount, and
    report where the observed count sits in that distribution.

    --only fdr
        "You ran a great many tests and reported the ones that worked."
    Collect every p-value this paper reports, from the files that produced
    them, into one declared family and control the false discovery rate over
    it.

WHAT IS AND IS NOT SYNTHETIC HERE
---------------------------------
Nothing in this script invents a value. A permutation reorders real numbers;
every number entering every arm is a measured one. No permuted column is
written to any exported panel, and nothing here feeds the fitted model, the
figures, or the manuscript's substantive numbers: these are null
distributions, and they are labelled as such in every file they are written
to.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

SKILL_JSON = C.DERIVED / "null_skill.json"
CROSS_JSON = C.DERIVED / "null_crossing.json"
CENSUS_JSON = C.DERIVED / "null_census.json"
FDR_JSON = C.DERIVED / "multiplicity_control.json"
FDR_CSV = C.DERIVED / "multiplicity_per_field.csv"

# ===========================================================================
#  SETTINGS
# ===========================================================================
NULL_CENSUS_DRAWS = 20000   # label shuffles. Cheap, so many.
NULL_CROSS_DRAWS = 40       # permutations of the attribution per variable.
                            # This number also sets the resolution of the
                            # per-variable p-value, which cannot be smaller
                            # than one over it, so it is the number the
                            # multiplicity arm inherits.
NULL_CROSS_BOOT = 120       # bootstrap draws inside each permutation. The
                            # paper uses 5,000; a null needs only enough to
                            # decide whether the gate WOULD pass, and the
                            # product of these two is five thousand crossings
                            # per variable, which is an hour for eleven
                            # variables and is checkpointed per variable.
FDR_Q = (0.05, 0.10)


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
#  3. The null for the census count
# ---------------------------------------------------------------------------
def null_census(args):
    print("\n" + "=" * 74)
    print("  3. THE NULL FOR THE CENSUS COUNT")
    print("=" * 74)
    print("  In each field, the two columns keep their attribution values and")
    print("  swap their labels at random. Recount. Under that null the count")
    print("  is a fair-coin binomial, so the exercise is partly a check that")
    print("  the sign test in the paper is the right test, and partly a")
    print("  statement of how far the observed count sits from the middle.")
    print()
    census = C.load_table("summary_statistic_census")
    use = census[census["comparable"].astype(str).str.lower().isin(
        ("true", "1", "1.0", "yes"))].copy()
    obs = int(use["noncentral_wins"].astype(str).str.lower()
              .isin(("true", "1", "1.0", "yes")).sum())
    n = int(len(use))
    print(f"    {n} comparable fields, non-central leads in {obs}")

    rng = C.seed_everything(extra=22)
    draws = args.null_draws or NULL_CENSUS_DRAWS
    counts = rng.binomial(n, 0.5, draws)
    q = {f"q{int(100 * v):02d}": float(np.quantile(counts, v))
         for v in (0.025, 0.25, 0.5, 0.75, 0.975)}
    p_emp = float((counts >= obs).mean())
    out = dict(observed=obs, n_comparable=n, n_draws=int(draws),
               null_mean=float(counts.mean()), null_sd=float(counts.std()),
               null_quantiles=q, empirical_p_one_sided=p_emp,
               note="Label-shuffle null for the attribution census count.")

    # the same, for the information census, if it is there
    try:
        it = C.load_table("information_theory_census")
        col = ("noncentral_wins_mi" if "noncentral_wins_mi" in it.columns
               else None)
        if col:
            obs_mi = int(it[col].astype(str).str.lower()
                         .isin(("true", "1", "1.0", "yes")).sum())
            n_mi = int(len(it))
            c2 = rng.binomial(n_mi, 0.5, draws)
            out["information_census"] = dict(
                observed=obs_mi, n_fields=n_mi,
                empirical_p_one_sided_high=float((c2 >= obs_mi).mean()),
                empirical_p_one_sided_low=float((c2 <= obs_mi).mean()),
                null_mean=float(c2.mean()))
            print(f"    information census: non-central leads in {obs_mi} of "
                  f"{n_mi}")
            print(f"      p for a count this high {out['information_census']['empirical_p_one_sided_high']:.4f}"
                  f", this low {out['information_census']['empirical_p_one_sided_low']:.4f}")
    except SystemExit:
        print("    (information_theory_census.csv absent; run script 14)")

    print(f"\n    null mean {out['null_mean']:.1f}, sd {out['null_sd']:.2f}, "
          f"95 per cent of draws between {q['q02']:.0f} and {q['q97']:.0f}")
    print(f"    a count of {obs} or more occurs in {p_emp:.5f} of draws")
    return out


# ---------------------------------------------------------------------------
#  4. Multiplicity over the declared family
# ---------------------------------------------------------------------------
def _bh(ps, q):
    """Benjamini-Hochberg. Returns the boolean rejections, in input order."""
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    thresh = q * (np.arange(1, m + 1) / m)
    passing = ps[order] <= thresh
    k = np.nonzero(passing)[0]
    out = np.zeros(m, bool)
    if len(k):
        cut = ps[order][k.max()]
        out = ps <= cut
    return out


def _harvest():
    """Every p-value this paper reports, from the file that produced it.

    The family has to be declared before it can be controlled, and a family
    assembled by hand is a family somebody can shorten. This reads the files.
    """
    found = []

    def add(label, p, source, family, note=""):
        try:
            p = float(p)
        except Exception:
            return
        if not np.isfinite(p):
            return
        found.append(dict(label=label, p=p, source=source, family=family,
                          note=note))

    def read(name, keys=()):
        path = C.DERIVED / name
        if not path.exists():
            return None
        try:
            blob = json.load(open(path))
        except Exception:
            return None
        if keys and any(k not in blob for k in keys):
            return None
        return blob

    b = read("summary_statistic_census.json", ("sign_test_p",))
    if b:
        add("attribution census, sign test over fields", b["sign_test_p"],
            "summary_statistic_census.json", "headline")
    b = read("information_theory_census.json", ("sign_test_p",))
    if b:
        add("information census, sign test over fields", b["sign_test_p"],
            "information_theory_census.json", "headline")
    b = read("residual_spatial_structure.json", ("p_value",))
    if b:
        add("Moran's I on the residuals", b["p_value"],
            "residual_spatial_structure.json", "diagnostic")
    b = read("lifetime_literature_test.json")
    if b:
        rf = b.get("rank_free_test") or {}
        add("lifetime, rank correlation, permutation p",
            rf.get("permutation_p"), "lifetime_literature_test.json",
            "mechanism")
        base = (b.get("cutoff_sweep") or {}).get("published_cutoff") or {}
        add("lifetime, Mann-Whitney at the published cutoff",
            base.get("mannwhitney_p"), "lifetime_literature_test.json",
            "mechanism")
    b = read("temporal_validation.json")
    if b:
        add("era split, Fisher exact", (b.get("era_split") or {}).get("fisher_p"),
            "temporal_validation.json", "diagnostic",
            "a large p is the desired result here, so this test is reported "
            "but is not a discovery and is marked as such")
    b = read("null_census.json")
    if b:
        # a permutation p cannot be smaller than one over the number of draws,
        # and reporting it as zero would make the correction look stronger
        # than the evidence supports
        floor = 1.0 / max(int(b.get("n_draws") or 1), 1)
        add("attribution census against its label-shuffle null",
            max(float(b.get("empirical_p_one_sided") or 0.0), floor),
            "null_census.json", "headline",
            "one-sided, resolution limited by the number of label shuffles")
    b = read("null_crossing.json")
    if b:
        for feat, rec in (b.get("per_variable") or {}).items():
            # the null pass rate IS the p-value for that variable's threshold:
            # the share of permutations in which the gate would have passed
            add(f"turning point survives the gate: {feat}",
                max(rec.get("permuted_pass_rate", 1.0),
                    1.0 / max(rec.get("n_draws", 1), 1)),
                "null_crossing.json", "thresholds",
                "one-sided, resolution limited by the number of permutations")
    return found


def multiplicity(args):
    print("\n" + "=" * 74)
    print("  4. MULTIPLICITY OVER THE DECLARED FAMILY")
    print("=" * 74)
    print("  Every p-value this paper reports, read out of the file that")
    print("  produced it, placed in one family, and controlled with")
    print("  Benjamini-Hochberg. Three things about this are deliberate:")
    print()
    print("   - The family is assembled by reading result files, not by hand,")
    print("     so it cannot be shortened to make the correction easier.")
    print("   - Tests whose desired answer is a LARGE p, such as the era")
    print("     split, are marked. Controlling the discovery rate over a test")
    print("     you hope fails is meaningless and the mark says so.")
    print("   - The per-variable threshold tests are limited in resolution by")
    print("     the number of permutations run, so their p-values are reported")
    print("     as bounded below by one over that number rather than as zero.")
    print()
    rows = _harvest()
    if not rows:
        raise SystemExit(
            "No p-value was found in any result file. Run scripts 12 and 14 "
            "at least, then this arm again.")
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    # tests whose desired answer is a large p are excluded from the correction
    # and reported separately, which is the only coherent thing to do with them
    want_large = df["note"].str.contains("large p", na=False)
    fam = df[~want_large].copy()
    print(f"    {len(df)} p-values found, {int(want_large.sum())} of them "
          f"tests whose desired answer is a large p")
    print(f"    family under control: {len(fam)}")
    print()
    out_q = {}
    for q in FDR_Q:
        rej = _bh(fam["p"].to_numpy(), q)
        fam[f"survives_q{q:g}"] = rej
        out_q[f"q_{q:g}"] = dict(n_rejected=int(rej.sum()),
                                 n_tests=int(len(fam)))
        print(f"    at q = {q:g}: {int(rej.sum())} of {len(fam)} survive")
    print()
    hdr = f"    {'p':>10}  {'family':<12} {'q=0.05':>7} {'q=0.1':>7}  label"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for _, r in fam.iterrows():
        print(f"    {r['p']:10.3g}  {r['family']:<12} "
              f"{'yes' if r.get('survives_q0.05') else 'no':>7} "
              f"{'yes' if r.get('survives_q0.1') else 'no':>7}  "
              f"{r['label'][:46]}")
    if want_large.any():
        print("\n    reported, not corrected (a large p is the desired answer):")
        for _, r in df[want_large].iterrows():
            print(f"    {r['p']:10.3g}  {r['label'][:60]}")

    missing = []
    for name in ("summary_statistic_census.json", "information_theory_census.json",
                 "residual_spatial_structure.json", "null_crossing.json",
                 "lifetime_literature_test.json", "temporal_validation.json"):
        if not (C.DERIVED / name).exists():
            missing.append(name)
    if missing:
        print("\n    NOT YET IN THE FAMILY, because these have not been run:")
        for n in missing:
            print(f"      {n}")
        print("    The correction above is over the tests that exist. Rerun")
        print("    this arm after the battery and the family will be complete.")

    df.to_csv(FDR_CSV, index=False)
    print(f"\n  [saved] {FDR_CSV}")
    return dict(n_p_values=int(len(df)), n_in_family=int(len(fam)),
                families=sorted(df["family"].unique().tolist()),
                control=out_q,
                not_corrected=df[want_large].to_dict("records"),
                tests=fam.to_dict("records"),
                absent_result_files=missing,
                note="Benjamini-Hochberg over the family assembled by reading "
                     "result files.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: skill, crossing, census, fdr")
    ap.add_argument("--only-arm", default="",
                    help="with --only skill, one permutation: full, "
                         "within_county, between_county")
    ap.add_argument("--null-folds", type=int, default=0,
                    help="folds for the null refits. Default is the paper's "
                         f"{C.N_FOLDS}; a smaller number is a smoke test and "
                         "is written into the result file.")
    ap.add_argument("--null-draws", type=int, default=0,
                    help="permutations per variable (crossing) or label "
                         "shuffles (census)")
    ap.add_argument("--null-boot", type=int, default=0,
                    help="bootstrap draws inside each permutation")
    a = ap.parse_args()
    want = (["skill", "crossing", "census", "fdr"] if a.only == "all"
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

    if "census" in want:
        out = null_census(a)
        json.dump(out, open(CENSUS_JSON, "w"), indent=1, default=str)
        print(f"  [saved] {CENSUS_JSON}")
        written.append(CENSUS_JSON)

    if "fdr" in want:
        out = multiplicity(a)
        json.dump(out, open(FDR_JSON, "w"), indent=1, default=str)
        print(f"  [saved] {FDR_JSON}")
        written += [FDR_JSON, FDR_CSV]

    C.write_manifest("20_null_calibration_and_multiplicity.py",
                     inputs=[C.FRAME, C.MODEL_A],
                     outputs=written,
                     settings=dict(arms=want,
                                   NULL_CENSUS_DRAWS=a.null_draws or NULL_CENSUS_DRAWS,
                                   NULL_CROSS_DRAWS=a.null_draws or NULL_CROSS_DRAWS,
                                   NULL_CROSS_BOOT=a.null_boot or NULL_CROSS_BOOT,
                                   null_folds=a.null_folds or C.N_FOLDS,
                                   FDR_Q=list(FDR_Q)),
                     notes="Null calibration and multiplicity control. Every "
                           "number here is a null distribution or a corrected "
                           "p-value, not a result about life expectancy.")


if __name__ == "__main__":
    main()
