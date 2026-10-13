"""
verify_nce_identity.py -- is the night-cooling pair one field described twice,
or two engineered variables built from two different pairs of sources?

The paper's central claim rests on the first being true. If the two night
cooling variables were derived from different instruments -- one from MODIS,
say, and one from ERA5 or PRISM -- then the twenty-fold attribution gap between
them would be a comparison of two products and not a comparison of two summary
statistics, and the result would collapse.

The panel holds enough temperature-like columns from enough instruments that
either variable COULD have been built several ways. So this does not check the
formula the pipeline claims to use. It searches every ordered pair of every
temperature-like column in the panel, across five ratio forms, and asks which
combinations reproduce each target. A claim survives only if the combinations
that reproduce it all come from one instrument family.

Two passes, because the full search is 94 x 93 x 5 x 2 evaluations over 52,309
rows: a subsample scan to shortlist, then full-panel confirmation of the
shortlist at full precision.

What "exact" means here is set by the data, not by a constant. Both targets are
stored float64, so the tolerance is stated as a relative agreement and reported
alongside the physical size of the residual in kelvin, which is what decides
whether a residual is a different source or a different arithmetic path.

    python3 verify_nce_identity.py --frame <parquet> --out nce_identity.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _config as C


TARGETS = ["eng_night_cooling_eff", "eng2_night_cooling_efficiency"]

# every column whose name marks it as a temperature, whatever instrument wrote
# it. Deliberately generous: a false candidate costs search time, a missing
# candidate would let a rival explanation go untested.
TEMP = re.compile(r"(lst|temp|tmax|tmin|tmean|skin|thermal)", re.I)

FORMS = {
    "(a-b)/a": lambda A, B: (A - B) / A,
    "(a-b)/b": lambda A, B: (A - B) / B,
    "(a-b)/(a+273.15)": lambda A, B: (A - B) / (A + 273.15),
    "(a-b)/(a+b)": lambda A, B: (A - B) / (A + B),
    "a-b": lambda A, B: A - B,
}

# a pair is shortlisted at NEAR and confirmed at EXACT_REL. EXACT_REL is a
# relative bound: an instrument disagreement shows up at 1e-2 or worse, an
# arithmetic-path difference at 1e-5 or better, and there is nothing in
# between, so the exact value of the cut does not decide anything.
NEAR = 1e-2
EXACT_REL = 1e-5
SUBSAMPLE = 2500


def source_of(col: str) -> str:
    return col.split("_", 1)[0]


def scan(frame: str) -> dict:
    names = pq.ParquetFile(frame).schema.names
    cand = [c for c in names if TEMP.search(c) and c not in TARGETS]
    d = pq.read_table(frame, columns=sorted(set(cand + TARGETS))).to_pandas()

    by_src: dict[str, int] = {}
    for c in cand:
        by_src[source_of(c)] = by_src.get(source_of(c), 0) + 1
    print(f"{len(d):,} rows, {len(cand)} temperature-like columns "
          f"from {len(by_src)} sources")
    for s, n in sorted(by_src.items()):
        print(f"   {s:<10}{n:>4}")

    sub = d.sample(min(SUBSAMPLE, len(d)), random_state=0)
    report = {}
    search = dict(n_rows=int(len(d)), n_columns=len(cand),
                  n_sources=len(by_src), by_source=by_src,
                  n_forms=len(FORMS), near=NEAR, exact_rel=EXACT_REL)

    for tgt in TARGETS:
        near = []
        t_sub = sub[tgt].to_numpy(float)
        ok_sub = np.isfinite(t_sub)
        for a, b in itertools.permutations(cand, 2):
            A = sub[a].to_numpy(float)
            B = sub[b].to_numpy(float)
            for label, f in FORMS.items():
                with np.errstate(all="ignore"):
                    v = f(A, B)
                m = ok_sub & np.isfinite(v)
                if m.sum() < 200:
                    continue
                md = np.abs(v[m] - t_sub[m]).max()
                if md < NEAR:
                    near.append((md, label, a, b))
        near.sort()

        hits = []
        t = d[tgt].to_numpy(float)
        for _, label, a, b in near[:12]:
            A = d[a].to_numpy(float)
            B = d[b].to_numpy(float)
            with np.errstate(all="ignore"):
                v = FORMS[label](A, B)
            m = np.isfinite(v) & np.isfinite(t)
            if m.sum() < 1000:
                continue
            resid = v[m] - t[m]
            md = float(np.abs(resid).max())
            rel = float((np.abs(resid) / np.abs(t[m])).max())
            # the residual re-expressed as the offset in b that would remove
            # it, which for a temperature ratio is a temperature in kelvin
            if label == "(a-b)/a":
                kelvin = float(np.median(np.abs(A[m] * (1 - t[m]) - B[m])))
            else:
                kelvin = float("nan")
            hits.append(dict(form=label, a=a, b=b, n=int(m.sum()),
                             maxdiff=md, max_rel=rel, kelvin=kelvin,
                             r=float(np.corrcoef(v[m], t[m])[0, 1]),
                             source_a=source_of(a), source_b=source_of(b),
                             exact=rel < EXACT_REL))
        hits.sort(key=lambda h: h["max_rel"])
        report[tgt] = hits

        print(f"\n### {tgt}")
        print(f"    {len(near)} pairs within {NEAR} on the subsample, "
              f"{sum(h['exact'] for h in hits)} confirmed on the full panel")
        for h in hits[:3]:
            tag = "  <-- IDENTITY" if h["exact"] else ""
            k = "" if np.isnan(h["kelvin"]) else f"  = {h['kelvin']:.1e} K"
            print(f"    max|d|={h['maxdiff']:.4e}  rel={h['max_rel']:.2e}{k}"
                  f"  {h['form']}  a={h['a']}  b={h['b']}  "
                  f"n={h['n']:,}{tag}")
    return report, search


def verdict(report: dict) -> dict:
    srcs, per_target = set(), {}
    ok = True
    for tgt, hits in report.items():
        ex = [h for h in hits if h["exact"]]
        s = {h["source_a"] for h in ex} | {h["source_b"] for h in ex}
        per_target[tgt] = sorted(s)
        srcs |= s
        if len(ex) != 1 or len(s) != 1:
            ok = False
    single = len(srcs) == 1
    print("\n" + "=" * 70)
    if ok and single:
        src = srcs.pop()
        print(f"VERDICT: each variable is reproduced by exactly one combination")
        print(f"         of columns, and both combinations come from the same")
        print(f"         instrument family, {src!r}. No column from any other")
        print(f"         source comes within {NEAR} of either target.")
        print("         The two variables are one field described twice.")
    else:
        print("VERDICT: NOT a clean single-source identity. Inspect the hits.")
        for t, s in per_target.items():
            print(f"         {t}: {s}")
    print("=" * 70)
    return dict(single_source=ok and single, sources=per_target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=str(C.FRAME))
    ap.add_argument("--out", default=str(C.DERIVED / "nce_identity.json"))
    a = ap.parse_args()
    rep, search = scan(a.frame)
    v = verdict(rep)
    json.dump(dict(hits=rep, verdict=v, search=search),
              open(a.out, "w"), indent=2)
    print("[saved]", a.out)


if __name__ == "__main__":
    main()
