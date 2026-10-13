#!/usr/bin/env python3
"""
29_census_exchangeable_null.py -- what census count would arise if a field's
summaries were interchangeable?

READS   the production attribution ranking (_config.RANKING_A)
        the census definitions of 12_summary_statistic_census_and_matched_refit.py
WRITES  ../derived*/census_exchangeable_null.json

WHY THIS EXISTS
---------------
The census compares, in each field, the most attributed NON-CENTRAL summary
with the most attributed CENTRAL one. A field usually carries more non-central
summaries (minimum, maximum, percentiles, spread) than central ones (mean,
median), so the best of several non-central summaries beats the best of one
or two central ones more often than half the time even when the statistic
does not matter at all. The fair-coin sign test ignores that.

The right null is exchangeability: if every summary of a field were equally
likely to be the most attributed, the non-central side would win field i with
probability n_noncentral_i / n_summaries_i. The count over fields is then a
sum of independent Bernoulli draws with those probabilities (a
Poisson-binomial distribution), computed here exactly by convolution.

    python3 29_census_exchangeable_null.py
"""
from __future__ import annotations

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

OUT_JSON = C.DERIVED / "census_exchangeable_null.json"
CENSUS_SCRIPT = os.path.join(HERE, "12_summary_statistic_census_and_matched_refit.py")


def load_census_rules():
    """The name parser and the minimum-summaries rule of the census itself,
    imported rather than copied so the two can never drift apart."""
    spec = importlib.util.spec_from_file_location("census12", CENSUS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.split_name, mod.MIN_STATS_PER_FIELD


def poisson_binomial(ps):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for q in ps:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return dist


def main():
    C.banner("29  the census count under exchangeable summaries")
    split_name, min_stats = load_census_rules()
    rk = pd.read_csv(C.RANKING_A)
    rows = []
    for _, r in rk.iterrows():
        s = split_name(r["feature"])
        if s:
            rows.append(dict(field=s[0], cls=s[1], stat=s[2],
                             shap=float(r["mean_abs_shap"])))
    p = pd.DataFrame(rows)

    fields = []
    for field, g in p.groupby("field"):
        if len(g) < min_stats:
            continue
        cen, non = g[g.cls == "central"], g[g.cls != "central"]
        if len(cen) and len(non):
            fields.append(dict(field=field, n_central=int(len(cen)),
                               n_noncentral=int(len(non)),
                               p_noncentral=len(non) / len(g),
                               noncentral_wins=bool(non.shap.max() > cen.shap.max())))
    f = pd.DataFrame(fields)
    obs = int(f["noncentral_wins"].sum())
    dist = poisson_binomial(f["p_noncentral"].to_numpy())
    cum = np.cumsum(dist)
    lo, hi = int(np.searchsorted(cum, 0.025)), int(np.searchsorted(cum, 0.975))
    expected = float(f["p_noncentral"].sum())
    p_upper = float(dist[obs:].sum())
    p_lower = float(dist[:obs + 1].sum())
    blob = dict(n_fields=int(len(f)), observed=obs, expected=expected,
                range95_lo=lo, range95_hi=hi,
                p_at_least_observed=p_upper, p_at_most_observed=p_lower,
                p_two_sided=float(min(1.0, 2 * min(p_upper, p_lower))),
                mean_share_noncentral=float(f["p_noncentral"].mean()),
                share_counts=f["p_noncentral"].round(3).value_counts()
                .sort_index().to_dict(),
                fields=fields,
                note="Null in which every summary of a field is equally likely "
                     "to be the most attributed. The fair-coin sign test is the "
                     "special case of one central and one non-central summary.")
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"  {len(f)} comparable fields, non-central leads in {obs}")
    print(f"  expected under exchangeable summaries {expected:.1f} "
          f"(95% range {lo} to {hi}), P(count >= {obs}) = {p_upper:.3f}")
    print(f"  on average {100 * f['p_noncentral'].mean():.0f}% of a field's "
          f"summaries are non-central")
    print(f"\n  [saved] {OUT_JSON}")
    C.write_manifest("29_census_exchangeable_null.py", inputs=[C.RANKING_A],
                     outputs=[OUT_JSON], settings={})


if __name__ == "__main__":
    main()
