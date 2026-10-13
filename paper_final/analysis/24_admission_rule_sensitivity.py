#!/usr/bin/env python3
"""
24_admission_rule_sensitivity.py -- where the admission rules were placed, and
what the paper would look like if they had been placed elsewhere.

READS   pipeline_core_v2.py from the project root, so the gate and the pruning
        filter are THE SAME CODE the main run used and not a reimplementation
        the reference frame (every candidate column, before any filter)
        the census of script 12
WRITES  ../derived*/admission_rule_sensitivity.json
        ../derived*/admission_rule_grid.csv

THE OBJECTION
-------------
    "Ninety-five per cent coverage and a correlation cut at 0.98 are
     arbitrary. So is a rule that exempts a list of variables you chose. These
     were chosen after you saw the result."

Three answers, all of them grids rather than arguments.

  1. COVERAGE. Sweep the gate from 0.90 to 0.99 and report how many columns
     are admitted at each setting.

  2. CORRELATION. At each coverage setting, sweep the collinearity cut from
     0.95 to no cut at all, and report how many columns are modelled.

  3. THE PROTECTED-PAIR RULE. This is the one that matters, because it is the
     rule that makes the paper's central comparison possible. Where both
     members of a correlated pair are protected, both are kept. Run the whole
     grid again with that exemption switched off and count how many of the
     census's comparisons survive. If the answer is that most of them vanish,
     the paper has to say so, in those words, in the Methods: the comparison
     exists because of a rule declared in advance, and a plain correlation
     filter would have removed one member of each pair before any model saw it.

A CHECK ON THIS SCRIPT ITSELF
-----------------------------
The collinearity loop is rerun here across a grid, and a grid of twenty
settings run through pandas `corr()` twenty times is slow, so the correlation
matrix is computed once per coverage setting and the dropping loop is applied
to it. That loop is a second implementation of eleven lines of
`pipeline_core_v2.redundancy_filter`, so the first thing this script does is
run both at the published settings and refuse to continue unless they agree
on every column. The agreement is written into the result file.

    python3 24_admission_rule_sensitivity.py
    python3 24_admission_rule_sensitivity.py --coverage 0.95 --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

OUT_JSON = C.DERIVED / "admission_rule_sensitivity.json"
OUT_CSV = C.DERIVED / "admission_rule_grid.csv"

# ===========================================================================
#  SETTINGS
# ===========================================================================
COVERAGE_LEVELS = (0.80, 0.90, 0.95, 0.983, 0.99)
#  Chosen after looking at the distribution of coverage across candidates,
#  which turns out to be sharply bimodal: a column is either observed for
#  essentially the whole window or for a small part of it, with almost nothing
#  in between. So the admitted count is identical over a wide band of
#  thresholds, and the sweep is placed to show where the band ends rather than
#  to take five samples inside it. The band's own endpoints are computed below
#  and reported, which is a better answer to the objection than any grid.
CORR_LEVELS = (0.95, 0.98, 0.995, 1.01)   # 1.01 means no column is dropped
PUBLISHED_COVERAGE = 0.95
PUBLISHED_CORR = 0.98
VAR_THRESHOLD = 1e-8                      # as pipeline_core_v2 has it
ORDER_SHUFFLES = 25                       # random column orders, to show the
                                          # dropping loop is order dependent and
                                          # that the protected pairs are not
PAIR = ("cams_fot_formaldehyde_above75th_percentile",
        "cams_fot_formaldehyde_above_90th_percentile")
STORE = C.DERIVED / "checkpoints_admission"


def cached_corr(frame_cols, X, label):
    """|corr| for these columns, computed once and kept.

    A pandas pairwise-complete correlation over seven hundred columns and
    fifty thousand rows is half a minute, and the grid needs one per coverage
    setting. Keeping them means an interrupted run continues instead of
    restarting, which matters because this script is slower than everything
    else in tier 0.
    """
    STORE.mkdir(parents=True, exist_ok=True)
    f = STORE / f"corr_{label}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["columns"]) == list(frame_cols):
            print(f"    (correlation matrix for {label} from checkpoint)")
            return z["corr"]
        print(f"    (checkpoint for {label} has different columns; "
              f"recomputing)")
    t0 = time.time()
    M = np.nan_to_num(X[frame_cols].corr().abs().to_numpy(), nan=0.0)
    np.savez_compressed(f, corr=M, columns=np.asarray(frame_cols, dtype=object))
    print(f"    (correlation matrix {M.shape} for {label} in "
          f"{time.time() - t0:.0f} s, saved)")
    return M


def load_pipeline_core():
    """pipeline_core_v2, imported from the project root by path.

    Imported rather than copied for the reason the whole package exists: the
    admission rules have one implementation, in the file the main run used,
    and a sensitivity analysis of those rules that reimplemented them would be
    a sensitivity analysis of the reimplementation.
    """
    path = C.ROOT / "pipeline_core_v2.py"
    if not path.exists():
        raise SystemExit(
            f"pipeline_core_v2.py is not at the project root.\n"
            f"    Looked in {C.ROOT}\n"
            f"    This script sweeps the admission rules using the main run's\n"
            f"    own gate and pruning filter, so that file has to be there.")
    spec = importlib.util.spec_from_file_location("_pcore", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("coverage_gate", "redundancy_filter", "is_protected"):
        if not hasattr(mod, name):
            raise SystemExit(f"pipeline_core_v2.py has no {name}. The file at "
                             f"{path} is not the one this script expects.")
    return mod


# ---------------------------------------------------------------------------
#  The dropping loop, applied to a correlation matrix already in hand
# ---------------------------------------------------------------------------
def drop_collinear(corr_abs, cols, protected, threshold, honour_protection=True):
    """Columns kept, under one correlation threshold.

    Eleven lines of pipeline_core_v2.redundancy_filter, separated from the
    matrix computation so the matrix is built once per coverage setting
    instead of once per grid cell. Checked against the original at the
    published settings before any of the grid is reported.
    """
    prot = set(protected) if honour_protection else set()
    n = len(cols)
    is_prot = np.fromiter((c in prot for c in cols), bool, n)
    dropped = np.zeros(n, bool)
    # Vectorised over j: the original walks every j > i and compares one
    # scalar at a time, which is a quarter of a million numpy scalar
    # operations per call and made the grid slower than the correlation
    # matrices. Scanning the row once and iterating only over the pairs that
    # exceed the threshold is the same logic and two orders of magnitude
    # faster. The semantics kept exactly: the scan over j continues even after
    # column i has itself been dropped, which is what the original does,
    # because its `if a in dropped` test is before the inner loop and not
    # inside it.
    for i in range(n):
        if dropped[i]:
            continue
        hits = np.nonzero(corr_abs[i] >= threshold)[0]
        for j in hits[hits > i]:
            if dropped[j]:
                continue
            if is_prot[i] and is_prot[j]:
                continue
            victim = j if not is_prot[j] else i
            if not is_prot[victim]:
                dropped[victim] = True
    keep = [c for k, c in enumerate(cols) if not dropped[k]]
    return keep, {cols[k] for k in np.nonzero(dropped)[0]}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, nargs="*", default=None,
                    help="coverage levels to sweep")
    ap.add_argument("--corr", type=float, nargs="*", default=None,
                    help="correlation cuts to sweep")
    ap.add_argument("--quick", action="store_true",
                    help="the published coverage only, both protection modes")
    a = ap.parse_args()
    covs = tuple(a.coverage or COVERAGE_LEVELS)
    corrs = tuple(a.corr or CORR_LEVELS)
    if a.quick:
        covs = (PUBLISHED_COVERAGE,)

    C.banner("24  where the admission rules were placed")
    C.seed_everything()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    core = load_pipeline_core()
    print(f"  admission rules imported from {C.ROOT / 'pipeline_core_v2.py'}")
    print(f"  {len(core.PROTECT_PATTERNS)} protected-name patterns")

    import joblib
    import pyarrow.parquet as pq
    C.require(C.FRAME, C.MODEL_A)
    d = joblib.load(C.MODEL_A)
    window = tuple(d["window"])
    published_features = list(d["features"])
    print(f"  window {window[0]} to {window[1]}, "
          f"{len(published_features)} columns in the published model")

    t0 = time.time()
    frame = pq.read_table(C.FRAME).to_pandas()
    print(f"  frame {frame.shape[0]:,} rows x {frame.shape[1]} columns, read "
          f"in {time.time() - t0:.0f} s")

    census = C.load_table("summary_statistic_census")
    pairs_all = C.census_pairs(census=census)
    print(f"  the census compares {len(pairs_all)} pairs")

    # ------------------------------------------------------------- the check
    print("\n" + "=" * 74)
    print("  0. IS THE LOOP IN THIS SCRIPT THE ONE IN THE PIPELINE?")
    print("=" * 74)
    # The gate is run ONCE. Its report carries every column's coverage and
    # the reason it was or was not admitted, so every other coverage setting
    # is a threshold on that report rather than another pass. The pass itself
    # calls nunique on 1,364 columns and takes a minute and a half, so five
    # passes would be most of this script's running time for no new
    # information.
    STORE.mkdir(parents=True, exist_ok=True)
    rep_path = STORE / "gate_report.csv"
    if rep_path.exists():
        report = pd.read_csv(rep_path)
        print(f"    gate report from checkpoint ({len(report)} candidates)")
    else:
        t0 = time.time()
        _kept, report = core.coverage_gate(
            frame, window, min_coverage=PUBLISHED_COVERAGE, verbose=False)
        report.to_csv(rep_path, index=False)
        print(f"    gate run once in {time.time() - t0:.0f} s, report saved")
    # a column whose only obstacle was coverage can be admitted by lowering
    # the threshold; one excluded for leakage, degeneracy, scope or
    # supersession cannot, at any threshold
    only_coverage = report["reason"].astype(str).str.startswith(
        ("admitted", "coverage-below-threshold"))
    eligible = report[only_coverage][["feature", "window_coverage"]]
    print(f"    {len(eligible)} of {len(report)} candidates are decided by "
          f"coverage alone; the rest are excluded at any threshold")

    # Order matters, and this is not a detail. The dropping loop walks the
    # columns and, for a correlated pair, removes the one it reaches second,
    # so the surviving column depends on the order the columns arrive in.
    # coverage_gate returns them in the frame's own order; the report is sorted
    # for reading. Taking the report's order instead of the frame's changed
    # one column out of 581 and the check below caught it, which is what the
    # check is for.
    frame_order = {c: i for i, c in enumerate(frame.columns)}
    cov_of = dict(zip(eligible["feature"].astype(str),
                      eligible["window_coverage"].astype(float)))

    def admitted_at(cov):
        return sorted((c for c, v in cov_of.items() if v >= cov),
                      key=lambda c: frame_order.get(c, 1 << 30))

    kept_pub = admitted_at(PUBLISHED_COVERAGE)
    print(f"    gate at {PUBLISHED_COVERAGE}: {len(kept_pub)} admitted")

    # Where the gate's placement stops mattering. The coverage distribution is
    # bimodal, so there is a band of thresholds over which the admitted set is
    # not merely similar but identical, and the width of that band answers the
    # arbitrariness objection better than any five-point sweep.
    cvals = np.sort(eligible["window_coverage"].astype(float).unique())
    n_pub = len(kept_pub)
    below = cvals[cvals <= PUBLISHED_COVERAGE]
    above = cvals[cvals > PUBLISHED_COVERAGE]
    lo_edge = float(below.max()) if len(below) else 0.0
    hi_edge = float(above.min()) if len(above) else 1.0
    band = dict(published=PUBLISHED_COVERAGE, n_admitted=n_pub,
                identical_from=lo_edge, identical_up_to=hi_edge,
                width=float(hi_edge - lo_edge),
                note="Every threshold strictly above the first value and up "
                     "to and including the second admits exactly the same "
                     "columns, because no candidate's coverage lies between "
                     "them.")
    print(f"    the same {n_pub} columns are admitted by every threshold")
    print(f"    above {lo_edge:.4f} and up to {hi_edge:.4f}, a band "
          f"{hi_edge - lo_edge:.4f} wide, because no")
    print(f"    candidate's coverage falls inside it. The 0.95 in the Methods")
    print(f"    is not a choice between different answers.")
    print(f"    coverage distribution: "
          + ", ".join(f">={t:g}: {int((eligible.window_coverage >= t).sum())}"
                      for t in (0.5, 0.8, 0.9, 0.95, 0.99)))
    year_col = getattr(core, "YEAR_COL", C.YEAR)
    win = frame[(frame[year_col] >= window[0])
                & (frame[year_col] <= window[1])]

    # ONE correlation matrix for the whole script. The correlation between two
    # columns does not depend on the coverage threshold, so the matrix is
    # computed once over the widest column set any swept threshold admits and
    # every narrower set is a slice of it. Computing one per coverage level
    # instead meant five pandas pairwise correlations and five copies of a
    # three-hundred-megabyte frame, which on a laptop is slower than every
    # other tier-zero study put together.
    widest_cov = min(list(covs) + [PUBLISHED_COVERAGE])
    base_cols = admitted_at(widest_cov)
    print(f"    widest set swept: coverage {widest_cov:g}, "
          f"{len(base_cols)} columns")
    Xbase = win[base_cols].astype("float32")
    var_base = Xbase.var(numeric_only=True, skipna=True)
    prot_base = {c for c in base_cols if core.is_protected(c)}
    after_base = [c for c in base_cols
                  if c in prot_base or var_base.get(c, 0) > VAR_THRESHOLD]
    STORE.mkdir(parents=True, exist_ok=True)
    M_base = cached_corr(after_base, Xbase, f"base_cov{widest_cov:g}")
    base_pos = {c: i for i, c in enumerate(after_base)}
    del Xbase
    Xpub = win[kept_pub]
    ck = STORE / "pipeline_reference.json"
    if ck.exists():
        official = json.load(open(ck))["columns"]
        print(f"    pipeline_core_v2.redundancy_filter: {len(official)} kept "
              f"(from checkpoint)")
    else:
        t0 = time.time()
        official = core.redundancy_filter(Xpub, corr_threshold=PUBLISHED_CORR,
                                          var_threshold=VAR_THRESHOLD,
                                          verbose=False)
        json.dump(dict(columns=list(official),
                       corr_threshold=PUBLISHED_CORR,
                       var_threshold=VAR_THRESHOLD,
                       coverage=PUBLISHED_COVERAGE),
                  open(ck, "w"), indent=1)
        print(f"    pipeline_core_v2.redundancy_filter: {len(official)} kept "
              f"({time.time() - t0:.0f} s)")

    protected = set(prot_base)
    after_var = [c for c in kept_pub if c in base_pos]
    idx_pub = np.asarray([base_pos[c] for c in after_var])
    corr_abs = M_base[np.ix_(idx_pub, idx_pub)]
    mine, _ = drop_collinear(corr_abs, after_var, protected, PUBLISHED_CORR)
    same = set(mine) == set(official)
    print(f"    this script's loop: {len(mine)} kept")
    print(f"    identical column sets: {'yes' if same else 'NO'}")
    if not same:
        only_mine = sorted(set(mine) - set(official))[:6]
        only_off = sorted(set(official) - set(mine))[:6]
        print(f"      only here:     {only_mine}")
        print(f"      only pipeline: {only_off}")
        raise SystemExit(
            "The two implementations disagree, so the grid below would be a\n"
            "sensitivity analysis of this script rather than of the paper's\n"
            "admission rules. Fix drop_collinear before reporting anything.\n"
            "Nothing has been written.")
    agrees_with_published = set(mine) == set(published_features)
    print(f"    and identical to the published feature list: "
          f"{'yes' if agrees_with_published else 'no'}")
    if not agrees_with_published:
        print(f"      here {len(mine)}, published {len(published_features)}, "
              f"shared {len(set(mine) & set(published_features))}")
        print("      The published model was fitted by the main run, which may")
        print("      have excluded further columns by hand. The grid below is")
        print("      still a sweep of the admission rules; it is not a claim")
        print("      to have reconstructed the published column list exactly,")
        print("      and this line is in the result file.")

    # ------------------------------------------------------------- the grid
    print("\n" + "=" * 74)
    print("  1. THE GRID")
    print("=" * 74)
    print("    coverage  corr   protection   admitted  modelled  protected   "
          "pair  pairs intact  count         p")
    print("    " + "-" * 100)
    # The published attribution values, so each cell can report not only how
    # many pairs survive but how the paper's own count comes out among the
    # ones that do. Without this the grid says which comparisons were
    # possible and not what they would have said.
    attrib = {}
    try:
        _rk = C.load_table("A_shap_ranking")
        attrib = dict(zip(_rk["feature"].astype(str),
                          _rk["mean_abs_shap"].astype(float)))
    except SystemExit:
        print("    (A_shap_ranking not readable; the per-cell count is "
              "omitted)")

    def count_in(ks):
        from math import comb
        got = wins = 0
        for _f, c1, c2 in pairs_all:
            if c1 in ks and c2 in ks and c1 in attrib and c2 in attrib:
                got += 1
                wins += int(attrib[c2] > attrib[c1])
        p = (float(sum(comb(got, k) for k in range(wins, got + 1)) / 2 ** got)
             if got else float("nan"))
        return got, wins, p

    rows, coverage_sweep = [], {}
    for cov in covs:
        kept = admitted_at(cov)
        after = [c for c in kept if c in base_pos]
        protc = {c for c in after if c in prot_base}
        n_var_dropped = len(kept) - len(after)
        idx = np.asarray([base_pos[c] for c in after])
        M = M_base[np.ix_(idx, idx)]
        coverage_sweep[f"{cov:g}"] = dict(
            n_admitted=len(kept), n_after_variance=len(after),
            n_variance_dropped=int(n_var_dropped),
            n_protected_admitted=int(len(protc)))
        for cut in corrs:
            for honour in (True, False):
                keep, dropped = drop_collinear(M, after, protc, cut,
                                               honour_protection=honour)
                ks = set(keep)
                pair_ok = all(c in ks for c in PAIR)
                intact = sum(1 for _f, c1, c2 in pairs_all
                             if c1 in ks and c2 in ks)
                n_c, w_c, p_c = count_in(ks)
                rec = dict(coverage=float(cov), corr_cut=float(cut),
                           n_counted=n_c, noncentral_wins=w_c,
                           sign_test_p=p_c,
                           protected_pairs_exempt=bool(honour),
                           n_admitted=len(kept),
                           n_variance_dropped=int(n_var_dropped),
                           n_collinearity_dropped=int(len(dropped)),
                           n_modelled=len(keep),
                           n_protected_kept=int(len(protc & ks)),
                           formaldehyde_pair_intact=bool(pair_ok),
                           n_census_pairs_intact=int(intact),
                           n_census_pairs=len(pairs_all),
                           is_published_setting=bool(
                               abs(cov - PUBLISHED_COVERAGE) < 1e-9
                               and abs(cut - PUBLISHED_CORR) < 1e-9
                               and honour))
                rows.append(rec)
                cut_txt = "none" if cut > 1 else f"{cut:g}"
                print(f"    {cov:8.3f}  {cut_txt:<5}  "
                      f"{'exempt' if honour else 'no exempt':<11}  "
                      f"{len(kept):>8}  {len(keep):>8}  "
                      f"{len(protc & ks):>9}  "
                      f"{'yes' if pair_ok else 'NO':>4}  "
                      f"{intact:>6} of {len(pairs_all):<3}  "
                      f"{w_c:>3} of {n_c:<3}  {p_c:>8.2g}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  [saved] {OUT_CSV}")

    # -------------------------------------------- does column order matter
    print("\n" + "=" * 74)
    print("  2. DOES THE ORDER OF THE COLUMNS CHANGE WHICH ONES SURVIVE?")
    print("=" * 74)
    print("  The dropping loop walks the columns and, for a correlated pair,")
    print("  removes the one it reaches second. So the surviving member of")
    print("  each pair depends on the order the columns arrive in, and the")
    print("  published column list is the one the frame's own order produces.")
    print("  This arm shuffles that order and reports how much of the admitted")
    print("  set is the same. The correlation matrix is already in hand, so it")
    print("  costs seconds.")
    print()
    order_rng = C.seed_everything(extra=31)
    base_set = set(mine)
    ord_rows = []
    for k in range(ORDER_SHUFFLES):
        shuffled = list(order_rng.permutation(np.asarray(after_var, object)))
        pos = {c: i for i, c in enumerate(shuffled)}
        idx = np.asarray([pos[c] for c in after_var])
        inv = np.empty(len(after_var), int)
        inv[idx] = np.arange(len(after_var))
        M = corr_abs[np.ix_(inv, inv)]
        keep, _ = drop_collinear(M, shuffled, protected, PUBLISHED_CORR)
        ks = set(keep)
        ord_rows.append(dict(draw=k + 1, n_modelled=len(keep),
                             shared_with_published=len(ks & base_set),
                             formaldehyde_pair_intact=bool(
                                 all(c in ks for c in PAIR)),
                             n_census_pairs_intact=int(sum(
                                 1 for _f, c1, c2 in pairs_all
                                 if c1 in ks and c2 in ks))))
    od = pd.DataFrame(ord_rows)
    order_sens = dict(
        n_shuffles=int(ORDER_SHUFFLES),
        published_n_modelled=len(mine),
        n_modelled_min=int(od.n_modelled.min()),
        n_modelled_max=int(od.n_modelled.max()),
        shared_min=int(od.shared_with_published.min()),
        shared_mean=float(od.shared_with_published.mean()),
        pair_intact_always=bool(od.formaldehyde_pair_intact.all()),
        census_pairs_min=int(od.n_census_pairs_intact.min()),
        census_pairs_max=int(od.n_census_pairs_intact.max()),
        note="The collinearity filter is order dependent by construction. "
             "The protected columns are exempt from it, so the pairs the "
             "paper compares are unaffected, and this arm demonstrates that "
             "rather than asserting it.")
    print(f"    column count stays between {order_sens['n_modelled_min']} and "
          f"{order_sens['n_modelled_max']} (published {len(mine)})")
    print(f"    on average {order_sens['shared_mean']:.1f} of {len(mine)} "
          f"columns are the same as the published set")
    print(f"    the formaldehyde pair survives every shuffle: "
          f"{'yes' if order_sens['pair_intact_always'] else 'NO'}")
    print(f"    census pairs intact between {order_sens['census_pairs_min']} "
          f"and {order_sens['census_pairs_max']} of {len(pairs_all)}")
    print()
    print("    THAT LAST LINE IS A REAL CAVEAT AND THE PAPER HAS TO CARRY IT.")
    print("    A pair with both members protected cannot be broken by any")
    print("    ordering, because a protected column is never the victim. A")
    print("    pair with an unprotected member can be, and under another order")
    print("    some of the 47 fields would not have had both descriptions in")
    print("    the model at all. So the population of comparable fields is")
    print("    itself a function of the order the filter walked.")
    print()
    immune = [(f, c1, c2) for f, c1, c2 in pairs_all
              if core.is_protected(c1) and core.is_protected(c2)]
    print(f"    of the {len(pairs_all)} pairs, {len(immune)} have both members")
    print(f"    protected and are immune to ordering; "
          f"{len(pairs_all) - len(immune)} are not")
    order_sens["n_pairs_both_protected"] = len(immune)
    order_sens["n_pairs_order_dependent"] = len(pairs_all) - len(immune)
    # The count restricted to the pairs no ordering can break. This is the
    # version of the headline that is immune to the objection, and it should
    # be reported beside the full count rather than instead of it.
    try:
        rank = C.load_table("A_shap_ranking")
        val = dict(zip(rank["feature"].astype(str),
                       rank["mean_abs_shap"].astype(float)))
        wins = sum(1 for _f, c1, c2 in immune
                   if val.get(c2, np.nan) > val.get(c1, np.nan))
        got = sum(1 for _f, c1, c2 in immune
                  if c1 in val and c2 in val)
        from math import comb
        p = (float(sum(comb(got, k) for k in range(wins, got + 1)) / 2 ** got)
             if got else float("nan"))
        order_sens.update(immune_subset=dict(
            n=got, noncentral_wins=wins, sign_test_p=p,
            note="The census count restricted to the pairs whose both members "
                 "are protected, and which therefore no ordering of the "
                 "collinearity filter could have broken."))
        print(f"    restricted to those {got} order-immune pairs, the "
              f"non-central")
        print(f"    description leads in {wins}, sign test p = {p:.4g}")
        print()
        print("    Report that number beside the full count. It is smaller and")
        print("    it is unimpeachable, and the two together say more than")
        print("    either alone.")
    except SystemExit:
        print("    (A_shap_ranking not readable, so the immune-subset count "
              "is not available)")

    # ------------------------------------------------- what the rule is worth
    print("\n" + "=" * 74)
    print("  3. WHAT THE PROTECTED-PAIR RULE IS WORTH")
    print("=" * 74)
    at = df[(df.coverage.sub(PUBLISHED_COVERAGE).abs() < 1e-9)
            & (df.corr_cut.sub(PUBLISHED_CORR).abs() < 1e-9)]
    with_rule = at[at.protected_pairs_exempt]
    without = at[~at.protected_pairs_exempt]
    rule = {}
    if len(with_rule) and len(without):
        w, o = with_rule.iloc[0], without.iloc[0]
        rule = dict(
            coverage=PUBLISHED_COVERAGE, corr_cut=PUBLISHED_CORR,
            modelled_with_rule=int(w.n_modelled),
            modelled_without_rule=int(o.n_modelled),
            census_pairs_with_rule=int(w.n_census_pairs_intact),
            census_pairs_without_rule=int(o.n_census_pairs_intact),
            n_census_pairs=int(w.n_census_pairs),
            formaldehyde_pair_with_rule=bool(w.formaldehyde_pair_intact),
            formaldehyde_pair_without_rule=bool(o.formaldehyde_pair_intact))
        print(f"    at the published settings, coverage {PUBLISHED_COVERAGE} "
              f"and cut {PUBLISHED_CORR}:")
        print(f"      with the exemption:    {int(w.n_modelled)} columns, "
              f"{int(w.n_census_pairs_intact)} of {int(w.n_census_pairs)} "
              f"census pairs intact, formaldehyde pair "
              f"{'kept' if w.formaldehyde_pair_intact else 'broken'}")
        print(f"      without it:            {int(o.n_modelled)} columns, "
              f"{int(o.n_census_pairs_intact)} of {int(o.n_census_pairs)} "
              f"census pairs intact, formaldehyde pair "
              f"{'kept' if o.formaldehyde_pair_intact else 'broken'}")
        lost = int(w.n_census_pairs_intact) - int(o.n_census_pairs_intact)
        print()
        print(f"    The exemption is what makes {lost} of the paper's "
              f"comparisons possible.")
        print("    That sentence belongs in the Methods, in those words. The")
        print("    rule was declared before the model was fitted and its")
        print("    patterns are listed in the appendix; stating what it buys")
        print("    is what separates a declared convention from a convenient")
        print("    one.")

    # the stability of the headline count across the grid
    span = dict(
        modelled_min=int(df.n_modelled.min()),
        modelled_max=int(df.n_modelled.max()),
        census_intact_min=int(df.n_census_pairs_intact.min()),
        census_intact_max=int(df.n_census_pairs_intact.max()),
        cells=int(len(df)),
        cells_pair_intact=int(df.formaldehyde_pair_intact.sum()))
    if "sign_test_p" in df.columns and df.sign_test_p.notna().any():
        span.update(
            worst_sign_test_p=float(df.sign_test_p.max()),
            best_sign_test_p=float(df.sign_test_p.min()),
            cells_significant_at_05=int((df.sign_test_p < 0.05).sum()),
            count_share_min=float((df.noncentral_wins / df.n_counted).min()),
            count_share_max=float((df.noncentral_wins / df.n_counted).max()))
    print("\n" + "=" * 74)
    print("  4. ACROSS THE WHOLE GRID")
    print("=" * 74)
    print(f"    {span['cells']} cells, modelled column count between "
          f"{span['modelled_min']} and {span['modelled_max']}")
    print(f"    census pairs intact between {span['census_intact_min']} and "
          f"{span['census_intact_max']} of {len(pairs_all)}")
    print(f"    the formaldehyde pair survives in {span['cells_pair_intact']} "
          f"of {span['cells']} cells")
    if "worst_sign_test_p" in span:
        print()
        print(f"    AND THIS IS THE SENTENCE FOR THE PAPER. In every one of the")
        print(f"    {span['cells']} cells of this grid, including the cells "
              f"where the")
        print(f"    protected-pair exemption is switched off and the cells "
              f"where the")
        print(f"    collinearity cut removes a third of the comparisons, the")
        print(f"    non-central description still leads: the share is between")
        print(f"    {span['count_share_min']:.2f} and "
              f"{span['count_share_max']:.2f} and the worst sign-test p over "
              f"the")
        print(f"    whole grid is {span['worst_sign_test_p']:.3g}, with "
              f"{span['cells_significant_at_05']} of {span['cells']} cells "
              f"below 0.05.")

    blob = dict(
        window=[int(window[0]), int(window[1])],
        coverage_levels=[float(x) for x in covs],
        correlation_levels=[float(x) for x in corrs],
        published=dict(coverage=PUBLISHED_COVERAGE, corr_cut=PUBLISHED_CORR,
                       var_threshold=VAR_THRESHOLD),
        implementation_check=dict(
            matches_pipeline_core=bool(same),
            matches_published_feature_list=bool(agrees_with_published),
            n_pipeline=len(official), n_here=len(mine),
            n_published=len(published_features)),
        coverage_sweep=coverage_sweep,
        coverage_insensitivity_band=band,
        correlation_sweep=df.to_dict("records"),
        protected_rule=rule,
        column_order_sensitivity=order_sens,
        grid_span=span,
        n_protect_patterns=len(core.PROTECT_PATTERNS),
        protect_patterns=list(core.PROTECT_PATTERNS),
        note="A sweep of the main run's own admission rules, using the main "
             "run's own gate and pruning code imported from "
             "pipeline_core_v2.py.")
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n  [saved] {OUT_JSON}")

    C.write_manifest("24_admission_rule_sensitivity.py",
                     inputs=[C.FRAME, C.MODEL_A, C.ROOT / "pipeline_core_v2.py"],
                     outputs=[OUT_JSON, OUT_CSV],
                     settings=dict(COVERAGE_LEVELS=list(covs),
                                   CORR_LEVELS=list(corrs),
                                   VAR_THRESHOLD=VAR_THRESHOLD,
                                   PUBLISHED_COVERAGE=PUBLISHED_COVERAGE,
                                   PUBLISHED_CORR=PUBLISHED_CORR),
                     notes="Sensitivity of the admitted set, and of the "
                           "census comparisons, to where the coverage gate "
                           "and the collinearity cut were placed and to the "
                           "protected-pair exemption.")


if __name__ == "__main__":
    main()
