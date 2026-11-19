"""
06_bootstrap_turning_points_and_gate.py
Where each response turns, how far that value moves under resampling, and
whether the answer depends on where we drew the line.

WHAT A TURNING POINT IS HERE
----------------------------
The value of a variable at which its contribution to the prediction changes
sign. Observations go into equal-count quantile bins of the variable, the mean
interventional attribution is taken in each bin, a smoothing spline is fitted
to those means, and the located value is where the spline crosses zero.

THE GATE, AND WHO DECIDED IT
----------------------------
Nobody outside this project. The three conditions

    a crossing is found in at least GATE_MIN_FOUND of the resamples
    the interval is narrower than GATE_MAX_WIDTH of the observed range
    the point estimate lies inside its own interval

are our convention, fixed before any estimate was seen. The percentile
bootstrap underneath them is standard; the cut values are not taken from any
published standard, and a paper that leans on them should not pretend
otherwise.

What makes the result defensible is therefore not the cuts but their
irrelevance, and this script measures that directly. It sweeps both cuts over
a wide grid and reports how many candidates pass at every combination. If the
verdict is the same across most of the grid, the convention is not doing the
work. If it flips near the values we chose, the paper has a problem and should
say so. Either way you will see it rather than guess.

ON THE NUMBER OF RESAMPLES
--------------------------
The first submission used 400. A 2.5th percentile estimated from 400 draws is
the tenth order statistic, which moves noticeably from one run to the next.
The default here is 2,000, and the script reports the Monte Carlo standard
error of each interval endpoint so you can judge whether to go higher. The
rule of thumb it prints is based on the spread of the draws themselves, not on
a table.

READS   derived/shap_dependence_candidates.csv   (from script 05)
WRITES  derived/turning_points.json
        derived/turning_points.csv
        derived/gate_sensitivity.csv
TIME    N_BOOT x variables spline fits. At 2,000 draws and eleven variables,
        roughly 10 to 25 minutes. Scales linearly, and --boot lets you try a
        small number first.

    python3 06_bootstrap_turning_points_and_gate.py
    python3 06_bootstrap_turning_points_and_gate.py --boot 200      # quick look
    python3 06_bootstrap_turning_points_and_gate.py --boot 5000     # publish
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS  (N_BOOT, N_BINS and the gate live in _config.py)
# ===========================================================================
# The grid the gate is swept over. The paper quotes the pass count at the
# declared cuts; this shows what it would have been elsewhere.
SWEEP_FOUND = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SWEEP_WIDTH = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

TRIM_PCT = (1, 99)   # ignore the extreme percentiles of a variable when
                     # fitting its curve, so one outlying county cannot place
                     # a crossing
MIN_DRAWS_FOR_CI = 30

# -- the two estimator choices that actually move the verdict ---------------
#
# A smoothed curve can cross zero more than once, and something has to decide
# which crossing is "the" turning point. That choice is not cosmetic: it
# changes how many candidates the gate passes, and this script reports the
# verdict under both rules so the dependence is visible rather than buried.
#
#   "steepest"        the crossing where the curve is steepest.
#                     This is what the published analysis used. It is the
#                     defensible choice, because a crossing on a steep section
#                     is a real change of sign and one on a flat section is
#                     noise; and it is the LESS forgiving choice, because on a
#                     genuinely flat curve the steepest crossing jumps from
#                     resample to resample, which is exactly the instability
#                     the gate is there to detect.
#
#   "nearest_median"  the crossing closest to the middle of the data.
#                     More stable by construction, and therefore a weaker
#                     test: it will pass curves that "steepest" rejects.
#
CROSSING_RULE = "steepest"      # "steepest" | "nearest_median"

#   "median"  the median attribution within each bin (the published choice,
#             robust to a few extreme SHAP values)
#   "mean"    the mean within each bin
BIN_STATISTIC = "median"        # "median" | "mean"

# Print the whole table a second time under the other rule, so the paper can
# state how far its headline depends on the estimator.
REPORT_BOTH_RULES = True
# ===========================================================================

import argparse
import json
import warnings

import numpy as np
import pandas as pd

import _config as C

warnings.filterwarnings("ignore")

NICE = {
    "prism_tdmean_min": "Annual minimum dew point",
    "eng2_night_cooling_efficiency": "Night cooling efficiency (mean form)",
    "eng_night_cooling_eff": "Night cooling efficiency (tail form)",
    "lst_night_1km_mean": "Mean night land-surface temperature",
    "eng2_wetbulb_night": "Nocturnal wet-bulb temperature",
    "eng2_wetbulb_hot": "Warm-hour wet-bulb temperature",
    "eng2_hcho_no2_ratio": "Formaldehyde to nitrogen dioxide ratio",
    "pm25_mean": "Annual mean fine particulate matter",
    "prism_tmin_min": "Annual minimum daily minimum temperature",
    "cams_fot_formaldehyde_above75th_percentile":
        "Formaldehyde hours above the 75th percentile",
    "usda_cropland_usda_corn_pct": "Cropland in corn",
}
UNIT = {
    "prism_tdmean_min": "degC", "eng2_wetbulb_night": "degC",
    "eng2_wetbulb_hot": "degC", "prism_tmin_min": "degC",
    "lst_night_1km_mean": "degC", "pm25_mean": "ug/m3",
    "cams_fot_formaldehyde_above75th_percentile": "%",
}


def crossing(x, y, nbins, lo, hi, rule=None, binstat=None):
    """
    Locate the zero crossing of the smoothed attribution curve, or None.

    Binned statistic, then a smoothing spline whose smoothing factor comes
    from the bin-to-bin scatter rather than from a chosen constant, then the
    crossing picked by `rule`. This reproduces the estimator used in the
    published analysis when rule="steepest" and binstat="median".
    """
    from scipy.interpolate import UnivariateSpline
    rule = rule or CROSSING_RULE
    binstat = binstat or BIN_STATISTIC
    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if m.sum() < nbins * 5:
        return None
    x, y = x[m], y[m]
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs[-1] += 1e-9
    bx, by = [], []
    for i in range(nbins):
        s = (x >= qs[i]) & (x < qs[i + 1])
        if s.sum() > 2:
            bx.append(np.median(x[s]))
            by.append(np.median(y[s]) if binstat == "median" else y[s].mean())
    bx, by = np.asarray(bx), np.asarray(by)
    keep = np.concatenate([[True], np.diff(bx) > 1e-9])
    bx, by = bx[keep], by[keep]
    if len(bx) < 8:
        return None
    r = np.diff(by)
    sc = np.median(np.abs(r - np.median(r))) * 1.4826
    try:
        spl = UnivariateSpline(bx, by, s=max(len(bx) * sc ** 2 * 1.2, 1e-12),
                               k=4)
    except Exception:
        return None
    gx = np.linspace(bx.min(), bx.max(), 600)
    gy = spl(gx)
    sign = np.sign(gy)
    idx = np.where(np.diff(sign) != 0)[0]
    if not len(idx):
        return None

    def at(i):
        x0, x1, y0, y1 = gx[i], gx[i + 1], gy[i], gy[i + 1]
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    if rule == "steepest":
        d1 = spl.derivative(1)
        best, best_slope = None, -np.inf
        for i in idx:
            xi = at(i)
            sl = abs(float(d1(xi)))
            if sl > best_slope:
                best, best_slope = xi, sl
        return best
    if rule == "nearest_median":
        med = np.median(x)
        return at(idx[np.argmin(np.abs(gx[idx] - med))])
    raise ValueError(f"unknown CROSSING_RULE {rule!r}")


def analyse(name, x, y, nboot, nbins, rng, rule=None, binstat=None):
    lo, hi = np.percentile(x[np.isfinite(x)], TRIM_PCT)
    span = float(hi - lo)
    point = crossing(x, y, nbins, lo, hi, rule, binstat)

    draws = []
    n = len(x)
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        v = crossing(x[idx], y[idx], nbins, lo, hi, rule, binstat)
        if v is not None and np.isfinite(v):
            draws.append(v)
    v = np.asarray(draws, float)
    out = dict(feature=name, label=NICE.get(name, name),
               unit=UNIT.get(name, ""), n=int(n), n_boot_attempted=int(nboot),
               n_boot_found=int(len(v)), x_lo=float(lo), x_hi=float(hi),
               span=span, point_estimate=point)

    if len(v) >= MIN_DRAWS_FOR_CI:
        ci_lo, ci_hi = np.percentile(v, [2.5, 97.5])
        out["ci_lo"], out["ci_hi"] = float(ci_lo), float(ci_hi)
        out["width_frac"] = float((ci_hi - ci_lo) / span) if span else np.nan
        # Monte Carlo error on the endpoints: bootstrap the bootstrap draws
        mc = [np.percentile(rng.choice(v, len(v)), [2.5, 97.5])
              for _ in range(200)]
        mc = np.asarray(mc)
        out["mc_se_lo"] = float(mc[:, 0].std())
        out["mc_se_hi"] = float(mc[:, 1].std())
        out["boot_sd"] = float(v.std())
    else:
        for k in ("ci_lo", "ci_hi", "width_frac", "mc_se_lo", "mc_se_hi",
                  "boot_sd"):
            out[k] = np.nan
    out["found_frac"] = len(v) / nboot if nboot else np.nan
    if point is not None and np.isfinite(out.get("ci_lo", np.nan)):
        out["pct_above"] = float(100 * (x > point).mean())
    else:
        out["pct_above"] = np.nan
    out["_draws"] = v
    return out


def passes(r, min_found, max_width, need_inside=True):
    if not np.isfinite(r.get("point_estimate") or np.nan):
        return False
    if r["found_frac"] < min_found:
        return False
    if not np.isfinite(r["width_frac"]) or r["width_frac"] >= max_width:
        return False
    if need_inside and not (r["ci_lo"] - 1e-9 <= r["point_estimate"]
                            <= r["ci_hi"] + 1e-9):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=C.N_BOOT)
    ap.add_argument("--bins", type=int, default=C.N_BINS)
    a = ap.parse_args()

    C.banner("06  bootstrap turning points and the stability gate")
    src = C.DERIVED / "shap_dependence_candidates.csv"
    C.require(src)
    d = pd.read_csv(src)
    feats = [c for c in d.columns if "shap_" + c in d.columns]
    print(f"  {len(feats)} candidate variables, {len(d):,} rows")
    print(f"  {a.boot:,} resamples, {a.bins} bins")
    print(f"  declared gate: found >= {C.GATE_MIN_FOUND:.0%}, "
          f"width < {C.GATE_MAX_WIDTH:.0%} of range, estimate inside interval")
    print()

    rng = C.seed_everything()
    rows = []
    for f in feats:
        x = d[f].to_numpy(float)
        y = d["shap_" + f].to_numpy(float)
        r = analyse(f, x, y, a.boot, a.bins, rng)
        r["stable"] = passes(r, C.GATE_MIN_FOUND, C.GATE_MAX_WIDTH,
                             C.GATE_INSIDE)
        rows.append(r)
        pt = r["point_estimate"]
        print(f"  {'PASS' if r['stable'] else 'fail'}  {r['label'][:44]:<46}"
              f"{'' if pt is None else f'{pt:8.3f}'}"
              f"  found {r['found_frac']:5.0%}"
              f"  width {r['width_frac']*100 if np.isfinite(r['width_frac']) else float('nan'):5.1f}%")

    tab = pd.DataFrame([{k: v for k, v in r.items() if k != "_draws"}
                        for r in rows])
    n_pass = int(tab.stable.sum())
    print(f"\n  {len(tab)} candidates, {n_pass} stable, "
          f"{len(tab) - n_pass} not")

    # -- is the bootstrap big enough? -----------------------------------
    print("\nMONTE CARLO ERROR ON THE INTERVAL ENDPOINTS")
    print("  If this is small next to the interval width, the resample count")
    print("  is sufficient. If it is comparable, raise --boot.")
    ok = tab[np.isfinite(tab.ci_lo)]
    for _, r in ok.iterrows():
        w = r.ci_hi - r.ci_lo
        rel = max(r.mc_se_lo, r.mc_se_hi) / w if w else np.nan
        flag = "ok   " if rel < 0.10 else "RAISE"
        print(f"  {flag} {r.label[:44]:<46}interval {w:8.3f}   "
              f"MC s.e. {max(r.mc_se_lo, r.mc_se_hi):7.4f}  ({rel:5.1%})")
    worst = (ok[["mc_se_lo", "mc_se_hi"]].max(axis=1)
             / (ok.ci_hi - ok.ci_lo)).max()
    if np.isfinite(worst):
        need = int(np.ceil(a.boot * (worst / 0.10) ** 2 / 100.0) * 100)
        print(f"\n  worst endpoint error is {worst:.1%} of its interval.")
        if worst >= 0.10:
            print(f"  To bring every endpoint under 10 %, use about "
                  f"--boot {need:,}.")
        else:
            print(f"  Every endpoint is under 10 % of its interval at "
                  f"{a.boot:,} resamples.")

    # -- does the verdict depend on where we drew the line? -------------
    print("\nGATE SENSITIVITY  (number of candidates passing)")
    grid = []
    for mf in SWEEP_FOUND:
        row = []
        for mw in SWEEP_WIDTH:
            k = sum(passes(r, mf, mw, C.GATE_INSIDE) for r in rows)
            grid.append(dict(min_found=mf, max_width=mw, n_pass=k))
            row.append(k)
        print(f"  found >= {mf:.0%}   " +
              "".join(f"{v:4d}" for v in row))
    print("        width <     " +
          "".join(f"{w*100:3.0f}%" for w in SWEEP_WIDTH))

    sw = pd.DataFrame(grid)
    at_declared = int(sw[(sw.min_found == C.GATE_MIN_FOUND)
                         & (sw.max_width == C.GATE_MAX_WIDTH)].n_pass.iloc[0])
    same = float((sw.n_pass == at_declared).mean())
    print(f"\n  At the declared cuts: {at_declared} pass.")
    print(f"  The same verdict holds over {same:.0%} of the swept grid "
          f"({len(sw)} combinations).")
    if same >= 0.6:
        print("  The convention is not driving the answer. That is the")
        print("  sentence the paper should make, with this table beside it.")
    else:
        print("  The verdict moves across the grid. Say so in the paper and")
        print("  report the range rather than a single count.")

    # -- and how far does it depend on the estimator itself? ------------
    alt = None
    if REPORT_BOTH_RULES:
        other = ("nearest_median" if CROSSING_RULE == "steepest"
                 else "steepest")
        print(f"\nUNDER THE OTHER CROSSING RULE  ({other})")
        print("  The gate asks whether a located value survives resampling.")
        print("  Which of several crossings is located is a separate choice,")
        print("  and it moves the answer, so both are reported.")
        rng2 = C.seed_everything()
        arows = []
        for f in feats:
            r = analyse(f, d[f].to_numpy(float),
                        d["shap_" + f].to_numpy(float),
                        max(200, a.boot // 4), a.bins, rng2, rule=other)
            r["stable"] = passes(r, C.GATE_MIN_FOUND, C.GATE_MAX_WIDTH,
                                 C.GATE_INSIDE)
            arows.append(r)
        n_alt = sum(r["stable"] for r in arows)
        print(f"  {CROSSING_RULE:<15} {n_pass} of {len(tab)} stable")
        print(f"  {other:<15} {n_alt} of {len(arows)} stable")
        flipped = [r["label"] for r, b in zip(arows, rows)
                   if r["stable"] != b["stable"]]
        if flipped:
            print("  verdict changes for:")
            for f in flipped:
                print(f"    {f}")
            print("\n  This is a real fork and the paper has to pick one rule")
            print("  and name it. 'steepest' is the published choice and the")
            print("  stricter one: on a flat curve the steepest crossing moves")
            print("  between resamples, which is the instability the gate is")
            print("  built to catch. 'nearest_median' is stabilised by")
            print("  construction and will pass curves that carry no signal.")
        else:
            print("  The verdict is the same under both rules.")
        alt = pd.DataFrame([{k: v for k, v in r.items() if k != "_draws"}
                            for r in arows])
        alt["rule"] = other

    C.DERIVED.mkdir(parents=True, exist_ok=True)
    p1 = C.DERIVED / "turning_points.csv"
    tab["rule"] = CROSSING_RULE
    tab.to_csv(p1, index=False)
    if alt is not None:
        alt.to_csv(C.DERIVED / "turning_points_alt_rule.csv", index=False)
    p2 = C.DERIVED / "gate_sensitivity.csv"
    sw.to_csv(p2, index=False)
    p3 = C.DERIVED / "turning_points.json"
    json.dump(dict(
        settings=dict(n_boot=a.boot, n_bins=a.bins, trim=list(TRIM_PCT),
                      gate_min_found=C.GATE_MIN_FOUND,
                      gate_max_width=C.GATE_MAX_WIDTH,
                      gate_inside=C.GATE_INSIDE, seed=C.SEED,
                      crossing_rule=CROSSING_RULE,
                      bin_statistic=BIN_STATISTIC),
        n_candidates=len(tab), n_stable=n_pass,
        verdict_unchanged_fraction_of_grid=same,
        worst_mc_error_fraction=float(worst) if np.isfinite(worst) else None,
        results=tab.to_dict("records")), open(p3, "w"), indent=1, default=str)
    print(f"\n[saved] derived/{p1.name}, derived/{p2.name}, derived/{p3.name}")
    C.write_manifest(__file__, [src], [p1, p2, p3],
                     settings=dict(n_boot=a.boot, n_bins=a.bins),
                     notes=f"{n_pass}/{len(tab)} stable; verdict unchanged "
                           f"over {same:.0%} of the gate grid")


if __name__ == "__main__":
    main()
