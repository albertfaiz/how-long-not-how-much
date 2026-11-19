"""
02_outcome_distribution_and_error_brackets.py
Where the outcome sits, and where the model's error sits inside it.

WHY THIS EXISTS
---------------
Figure 2C reports a mean absolute error of about 6.6 years in the lowest
life-expectancy bracket, on n = 153 county-years, and 153 looks too small to
be right. This script recomputes the brackets from the out-of-fold predictions
so the number can be checked rather than believed, and prints the whole
outcome distribution around it.

The short answer, which the output below will show you: the bracket is
labelled in YEARS OF LIFE EXPECTANCY AT BIRTH, not in age. The lowest observed
county-year in the panel is about 65 years, so the interval [60, 70) contains
only the small number of county-years whose life expectancy is under seventy.
There are 22 such counties. They are the reservation and central-Appalachian
counties that also dominate the residual map, which is a useful consistency
check rather than a coincidence.

The script also warns when a bracket is too thin to support a stable mean, so
that the bracket edges can be reconsidered rather than defended.

READS   studyA_oof_predictions.csv
WRITES  derived/outcome_distribution.csv
        derived/bracket_error.csv        (the figure's input, recomputed)
        derived/bracket_error_check.json
TIME    seconds

    python3 02_outcome_distribution_and_error_brackets.py
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
# The bracket edges the manuscript uses. Left-closed, right-open.
BRACKET_EDGES = [60, 70, 74, 76.5, 79, 95]

# A bracket holding fewer than this many county-years gets a warning printed,
# because a mean absolute error on a handful of rows is not a stable quantity.
THIN_BRACKET = 500

# Also report equal-count brackets, as an alternative the paper could use.
ALSO_REPORT_QUINTILES = True
# ===========================================================================

import json

import numpy as np
import pandas as pd

import _config as C

FIPS_NAMES = {
    46102: "Oglala Lakota, SD", 46121: "Todd, SD", 46017: "Buffalo, SD",
    38085: "Sioux, ND", 38079: "Rolette, ND", 30085: "Roosevelt, MT",
    30035: "Glacier, MT", 54047: "McDowell, WV", 21189: "Owsley, KY",
    12125: "Union, FL", 51690: "Martinsville, VA", 51730: "Petersburg, VA",
}


def main():
    C.banner("02  outcome distribution and error brackets")
    C.require(C.OOF_A)
    d = pd.read_csv(C.OOF_A)
    y = d[C.TARGET].to_numpy(float)
    err = np.abs(d["resid"].to_numpy(float))

    print(f"county-years            {len(d):,}")
    print(f"counties                {d[C.GROUP].nunique():,}")
    print(f"years                   {d[C.YEAR].min()}-{d[C.YEAR].max()}")
    print()
    print("LIFE EXPECTANCY AT BIRTH, YEARS  (this is the outcome, not an age)")
    q = np.percentile(y, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    for lab, v in zip(["min", "p1", "p5", "p25", "median", "p75", "p95",
                       "p99", "max"], q):
        print(f"  {lab:<8}{v:8.2f}")
    print()
    print(f"  county-years below 70 years : {int((y < 70).sum()):>6,}"
          f"   ({100*(y<70).mean():.2f} % of the panel)")
    print(f"  county-years below 72 years : {int((y < 72).sum()):>6,}")
    print(f"  county-years below 74 years : {int((y < 74).sum()):>6,}")
    print(f"  distinct counties below 70  : "
          f"{d.loc[y < 70, C.GROUP].nunique():>6,}")

    sub = d[y < 70]
    if len(sub):
        print("\n  the counties that make up the lowest bracket:")
        cnt = sub.groupby(C.GROUP).size().sort_values(ascending=False)
        for fips, n in cnt.items():
            nm = FIPS_NAMES.get(int(fips), "")
            print(f"    {int(fips):>5}  {n:>3} year(s)   {nm}")

    # -- the brackets the figure uses, recomputed -----------------------
    print("\nERROR BY BRACKET  (recomputed from the out-of-fold predictions)")
    lab = []
    for lo, hi in zip(BRACKET_EDGES[:-1], BRACKET_EDGES[1:]):
        lab.append(f"{lo:g}-{hi:g}")
    cut = pd.cut(y, BRACKET_EDGES, right=False, labels=lab)
    g = pd.DataFrame({"bracket": cut, "err": err, "y": y,
                      "resid": d["resid"]}).groupby("bracket", observed=False)
    br = pd.DataFrame({
        "bracket": lab,
        "n": g.size().reindex(lab).to_numpy(),
        "mae": g.err.mean().reindex(lab).to_numpy(),
        "mean_resid": g.resid.mean().reindex(lab).to_numpy(),
    })
    # a standard error on the bracket mean, so a thin bracket declares itself
    br["mae_se"] = (g.err.std().reindex(lab).to_numpy()
                    / np.sqrt(br.n.to_numpy()))
    print(br.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    thin = br[br.n < THIN_BRACKET]
    warnings = []
    if len(thin):
        for _, r in thin.iterrows():
            msg = (f"bracket {r.bracket} holds {int(r.n)} county-years; its "
                   f"MAE of {r.mae:.2f} carries a standard error of "
                   f"{r.mae_se:.2f}")
            warnings.append(msg)
            print(f"\n  NOTE  {msg}")
        print("        The value is correct, and it rests on few observations.")
        print("        Figure 2C prints each bracket's count for this reason.")

    # -- compare against the stored file the figure actually reads ------
    stored_path = C.STUDY_A / "studyA_bracket_error.csv"
    agree = None
    if stored_path.exists():
        st = pd.read_csv(stored_path)
        merged = br.merge(st, on="bracket", suffixes=("_new", "_stored"),
                          how="outer")
        dn = np.abs(merged.n_new - merged.n_stored).max()
        dm = np.abs(merged.mae_new - merged.mae_stored).max()
        agree = bool(dn == 0 and dm < 1e-6)
        print(f"\nAGAINST THE STORED FILE ({stored_path.name})")
        print(f"  max difference in count  {dn:.0f}")
        print(f"  max difference in MAE    {dm:.2e}")
        print(f"  {'reproduces exactly' if agree else 'DOES NOT REPRODUCE'}")

    # -- an equal-count alternative, for comparison ---------------------
    if ALSO_REPORT_QUINTILES:
        print("\nEQUAL-COUNT ALTERNATIVE  (five brackets of equal size)")
        qc = pd.qcut(y, 5)
        qg = pd.DataFrame({"b": qc, "err": err}).groupby("b", observed=True)
        alt = pd.DataFrame({"bracket": [str(i) for i in qg.size().index],
                            "n": qg.size().to_numpy(),
                            "mae": qg.err.mean().to_numpy()})
        print(alt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("  Equal-count brackets remove the thin-bracket problem but hide")
        print("  the extreme tail, which is the point Figure 2C is making.")

    C.DERIVED.mkdir(parents=True, exist_ok=True)
    out1 = C.DERIVED / "outcome_distribution.csv"
    pd.DataFrame({"percentile": [0, 1, 5, 25, 50, 75, 95, 99, 100],
                  "life_expectancy_years": q}).to_csv(out1, index=False)
    out2 = C.DERIVED / "bracket_error.csv"
    br.to_csv(out2, index=False)
    out3 = C.DERIVED / "bracket_error_check.json"
    json.dump(dict(n_rows=int(len(d)),
                   min_life_expectancy=float(y.min()),
                   max_life_expectancy=float(y.max()),
                   n_below_70=int((y < 70).sum()),
                   counties_below_70=int(d.loc[y < 70, C.GROUP].nunique()),
                   brackets=br.to_dict("records"),
                   reproduces_stored_file=agree,
                   thin_bracket_warnings=warnings),
              open(out3, "w"), indent=1)
    print(f"\n[saved] {out1.name}, {out2.name}, {out3.name}  in derived/")
    C.write_manifest(__file__, [C.OOF_A], [out1, out2, out3],
                     settings=dict(BRACKET_EDGES=BRACKET_EDGES),
                     notes="; ".join(warnings))


if __name__ == "__main__":
    main()
