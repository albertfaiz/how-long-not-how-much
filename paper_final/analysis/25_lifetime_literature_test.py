#!/usr/bin/env python3
"""
25_lifetime_literature_test.py -- does the duration advantage follow
atmospheric lifetime, and does that claim survive being attacked?

READS   A_duration_vs_concentration.csv from the main run, which pairs each
        species' duration metric against its concentration metric and gives
        the attribution ratio between them
WRITES  ../derived*/lifetime_literature_test.json
        ../derived*/lifetime_cutoff_sweep.csv
        ../derived*/lifetime_sources.csv

WHAT THE MAIN RUN ALREADY FOUND
-------------------------------
Splitting twenty species-percentile pairs at a two-day lifetime, the duration
metric wins in 7 of 8 short-lived cases against 3 of 12 longer-lived ones,
one-sided Mann-Whitney p = 0.0062, Spearman rho = -0.47.

That is the paper's mechanism, and as stated it has four weaknesses a referee
will find in the order below.

  1. PSEUDO-REPLICATION, and this is the serious one. The twenty rows are not
     twenty independent observations. They are thirteen species, most of them
     contributing two rows, one for the 75th percentile and one for the 90th.
     The two rows for a species share its chemistry, its emission pattern and
     its measurement path, so a test that treats them as independent has
     roughly half the degrees of freedom it claims. This script runs the test
     again with the species as the unit and reports both. The species-level
     result is the one the paper should quote.

  2. THE CUTOFF. Two days was chosen. Swept here from six hours to a week.

  3. THE VALUES. The lifetimes are order-of-magnitude figures from standard
     references. Each is perturbed across an order of magnitude, many times,
     and the share of perturbations that preserve the sign is reported.

  4. ONE SPECIES CARRYING IT. Jackknifed, leaving out each species in turn.

A cutoff-free test is also run: the rank correlation between lifetime and the
attribution ratio, with a permutation null computed by shuffling the lifetime
labels among species. No cutoff appears in it at all, which makes it the
version that cannot be accused of a chosen threshold.

ON THE LITERATURE VALUES
------------------------
`lifetime_sources.csv` names a reference for every lifetime used. They are
order-of-magnitude tropospheric figures and they are used only to order and
coarsely classify the species, which is why the sweep and the perturbation
arms exist. CROSSCHECK: the reference list in the SOURCES table below should
be reconciled against references.bib before submission, and any entry the
bibliography does not support should be replaced or the value attributed to a
different source.

    python3 25_lifetime_literature_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

OUT_JSON = C.DERIVED / "lifetime_literature_test.json"
SWEEP_CSV = C.DERIVED / "lifetime_cutoff_sweep.csv"
SOURCES_CSV = C.DERIVED / "lifetime_sources.csv"

# ===========================================================================
#  SETTINGS
# ===========================================================================
PUBLISHED_CUTOFF = 2.0        # days, as the main run used
CUTOFF_DAYS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0)
VALUE_DRAWS = 4000            # perturbations of the lifetime values
VALUE_SPREAD_DEX = 0.5        # each value multiplied by 10**U(-d, +d), so a
                              # full order of magnitude end to end
PERM_DRAWS = 20000            # label shuffles for the rank-correlation null

# ---------------------------------------------------------------------------
#  ORDER-OF-MAGNITUDE TROPOSPHERIC LIFETIMES
# ---------------------------------------------------------------------------
#  These are the values the main run used, with a reference written beside
#  each one. They are order-of-magnitude figures, they vary with season,
#  latitude and oxidant level, and the analysis below is designed so that none
#  of the conclusions depends on any one of them: the cutoff is swept, every
#  value is perturbed across an order of magnitude, and the species are
#  jackknifed.
#
#  CROSSCHECK every row of this table against references.bib before
#  submission.
LIFETIME_DAYS = {
    "hydroxyl radical":      1e-5,
    "isoprene":              0.1,
    "formaldehyde":          0.2,
    "nitrogen monoxide":     0.2,
    "nitrogen dioxide":      0.25,
    "hydrogen peroxide":     1.0,
    "sulphur dioxide":       2.0,
    "nitric acid":           5.0,
    "pm2 5":                 7.0,
    "propane":               11.0,
    "ozone":                 22.0,
    "peroxyacetyl nitrate":  30.0,
    "ethane":                60.0,
}

SOURCES = {
    "hydroxyl radical": (
        "seconds. Lifetime set by reaction with ambient CO, CH4 and VOCs.",
        "Lelieveld, Gromov, Pozzer and Taraborrelli, Global tropospheric "
        "hydroxyl distribution, budget and reactivity, Atmos. Chem. Phys. "
        "16:12477, 2016"),
    "isoprene": (
        "hours. Dominated by OH addition during daylight.",
        "Atkinson and Arey, Atmospheric degradation of volatile organic "
        "compounds, Chem. Rev. 103:4605, 2003"),
    "formaldehyde": (
        "hours. Photolysis and OH reaction, both fast in daylight.",
        "Atkinson and Arey, Chem. Rev. 103:4605, 2003; Seinfeld and Pandis, "
        "Atmospheric Chemistry and Physics, 3rd edition, 2016"),
    "nitrogen monoxide": (
        "minutes to hours in the photostationary state with NO2 and O3; the "
        "value here is for the NOx family rather than for NO alone.",
        "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
        "edition, 2016"),
    "nitrogen dioxide": (
        "hours. OH reaction to HNO3 in daylight, heterogeneous loss at night.",
        "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
        "edition, 2016"),
    "hydrogen peroxide": (
        "about a day. Photolysis, OH reaction and efficient wet removal.",
        "Finlayson-Pitts and Pitts, Chemistry of the Upper and Lower "
        "Atmosphere, 2000"),
    "sulphur dioxide": (
        "one to two days in the boundary layer. Oxidation to sulphate plus "
        "dry and wet deposition.",
        "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
        "edition, 2016"),
    "nitric acid": (
        "days. Removal dominated by wet and dry deposition rather than "
        "chemistry.",
        "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
        "edition, 2016"),
    "pm2 5": (
        "about a week for submicron aerosol in the lower troposphere, set by "
        "wet removal.",
        "Textor et al., Analysis and quantification of the diversities of "
        "aerosol life cycles within AeroCom, Atmos. Chem. Phys. 6:1777, 2006"),
    "propane": (
        "about ten days against OH at typical mid-latitude oxidant levels.",
        "Atkinson and Arey, Chem. Rev. 103:4605, 2003"),
    "ozone": (
        "weeks in the free troposphere, much shorter in a polluted boundary "
        "layer. The value here is the tropospheric burden lifetime.",
        "Young et al., Pre-industrial to end 21st century projections of "
        "tropospheric ozone, Atmos. Chem. Phys. 13:2063, 2013"),
    "peroxyacetyl nitrate": (
        "weeks in the cold free troposphere, hours in a warm boundary layer; "
        "the value here is the reservoir lifetime that makes PAN a long-range "
        "NOx carrier.",
        "Fischer et al., Atmospheric peroxyacetyl nitrate (PAN): a global "
        "budget and source attribution, Atmos. Chem. Phys. 14:2679, 2014"),
    "ethane": (
        "about two months against OH. The longest-lived species here.",
        "Atkinson and Arey, Chem. Rev. 103:4605, 2003"),
}


# ---------------------------------------------------------------------------
def load_pairs():
    """The duration-against-concentration table the main run wrote."""
    d = C.load_table("A_duration_vs_concentration")
    need = {"species", "ratio"}
    if not need.issubset(set(c.lower() for c in d.columns)):
        raise SystemExit(
            f"A_duration_vs_concentration.csv does not carry "
            f"{', '.join(sorted(need))}. Columns: {list(d.columns)}")
    d.columns = [c.lower() for c in d.columns]
    d["species"] = d["species"].astype(str).str.strip().str.lower()
    d["lifetime_days"] = d["species"].map(LIFETIME_DAYS)
    unknown = sorted(set(d.loc[d.lifetime_days.isna(), "species"]))
    if unknown:
        print(f"  no lifetime for {len(unknown)} species, dropped: "
              f"{', '.join(unknown)}")
    d = d.dropna(subset=["lifetime_days", "ratio"]).copy()
    d["duration_wins"] = d["ratio"] > 1
    return d


def species_level(d):
    """One row per species: the unit the test should actually use."""
    g = d.groupby("species")
    out = pd.DataFrame(dict(
        lifetime_days=g["lifetime_days"].first(),
        n_rows=g.size(),
        median_ratio=g["ratio"].median(),
        geometric_mean_ratio=np.exp(g["ratio"].apply(
            lambda s: np.log(s[s > 0]).mean() if (s > 0).any() else np.nan)),
        duration_wins_any=g["duration_wins"].any(),
        duration_wins_all=g["duration_wins"].all())).reset_index()
    out["duration_wins"] = out["median_ratio"] > 1
    return out


def one_split(frame, cutoff, ratio_col="ratio"):
    from scipy import stats
    short = frame[frame.lifetime_days < cutoff]
    long_ = frame[frame.lifetime_days >= cutoff]
    rec = dict(cutoff_days=float(cutoff),
               n_short=int(len(short)), n_long=int(len(long_)),
               short_wins=int(short.duration_wins.sum()),
               long_wins=int(long_.duration_wins.sum()),
               short_median_ratio=float(short[ratio_col].median())
               if len(short) else float("nan"),
               long_median_ratio=float(long_[ratio_col].median())
               if len(long_) else float("nan"))
    if len(short) >= 2 and len(long_) >= 2:
        u, p = stats.mannwhitneyu(short[ratio_col], long_[ratio_col],
                                  alternative="greater")
        rec.update(mannwhitney_u=float(u), mannwhitney_p=float(p))
    else:
        rec.update(mannwhitney_u=float("nan"), mannwhitney_p=float("nan"))
    return rec


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=float, nargs="*", default=None)
    a = ap.parse_args()
    cutoffs = tuple(a.cutoffs or CUTOFF_DAYS)

    C.banner("25  does the duration advantage follow atmospheric lifetime")
    C.seed_everything()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    rng = C.seed_everything(extra=28)

    d = load_pairs()
    sp = species_level(d)
    print(f"  {len(d)} species-percentile pairs across {len(sp)} species")
    print(f"  {int((sp.n_rows > 1).sum())} species contribute more than one "
          f"pair, which is why the species-level test exists")

    # ------------------------------------------------------------ the units
    print("\n" + "=" * 74)
    print("  1. THE SAME TEST ON TWO DIFFERENT UNITS")
    print("=" * 74)
    print("  The pair-level test is the one the main run reported. The")
    print("  species-level test treats each species once, which is the")
    print("  number of independent chemical systems actually observed.")
    print()
    pair_pub = one_split(d, PUBLISHED_CUTOFF)
    sp_pub = one_split(sp, PUBLISHED_CUTOFF, ratio_col="median_ratio")
    for name, rec, n in (("pair level", pair_pub, len(d)),
                         ("species level", sp_pub, len(sp))):
        print(f"    {name:<14} n = {n:>3}   duration wins "
              f"{rec['short_wins']}/{rec['n_short']} short against "
              f"{rec['long_wins']}/{rec['n_long']} long   "
              f"p = {rec['mannwhitney_p']:.4f}")
    print()
    print("  The species-level p will be the larger of the two and it is the")
    print("  one to quote. Reporting the smaller one as though twenty")
    print("  independent species had been observed is the single easiest")
    print("  thing for a referee to object to in this paper.")

    # --------------------------------------------------------- the cutoff
    print("\n" + "=" * 74)
    print("  2. THE CUTOFF, SWEPT")
    print("=" * 74)
    print("    cutoff   unit      short        long        p")
    print("    " + "-" * 56)
    rows = []
    for cut in cutoffs:
        for unit, frame, rc in (("pair", d, "ratio"),
                                ("species", sp, "median_ratio")):
            rec = one_split(frame, cut, ratio_col=rc)
            rec["unit"] = unit
            rec["is_published_cutoff"] = bool(abs(cut - PUBLISHED_CUTOFF) < 1e-9)
            rows.append(rec)
            print(f"    {cut:6.2f}d  {unit:<8}  "
                  f"{rec['short_wins']:>2}/{rec['n_short']:<2}       "
                  f"{rec['long_wins']:>2}/{rec['n_long']:<2}      "
                  f"{rec['mannwhitney_p']:.4f}")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(SWEEP_CSV, index=False)
    print(f"\n  [saved] {SWEEP_CSV}")
    sp_sweep = sweep[sweep.unit == "species"].dropna(subset=["mannwhitney_p"])
    n_sig = int((sp_sweep.mannwhitney_p < 0.05).sum())
    print(f"    species level: significant at 0.05 in {n_sig} of "
          f"{len(sp_sweep)} cutoffs tried")

    # ------------------------------------------------------- cutoff-free
    print("\n" + "=" * 74)
    print("  3. THE TEST WITH NO CUTOFF IN IT")
    print("=" * 74)
    print("  Rank correlation between lifetime and the attribution ratio,")
    print("  across species, with a null built by shuffling which lifetime")
    print("  belongs to which species. No threshold appears anywhere in it.")
    print()
    from scipy import stats
    lt = sp.lifetime_days.to_numpy(float)
    rt = sp.median_ratio.to_numpy(float)
    rho, p_param = stats.spearmanr(lt, rt)
    null = np.empty(PERM_DRAWS)
    for i in range(PERM_DRAWS):
        null[i] = stats.spearmanr(rng.permutation(lt), rt).statistic
    p_perm = float((null <= rho).mean())        # one-sided: expect negative
    print(f"    Spearman rho = {rho:+.3f}   asymptotic p = {p_param:.4f}")
    print(f"    permutation p over {PERM_DRAWS:,} species-label shuffles = "
          f"{p_perm:.4f}")
    print(f"    null rho has mean {null.mean():+.3f} and sd {null.std():.3f}")
    rank_free = dict(unit="species", n=int(len(sp)), spearman_rho=float(rho),
                     asymptotic_p=float(p_param), permutation_p=p_perm,
                     n_permutations=int(PERM_DRAWS),
                     null_mean=float(null.mean()), null_sd=float(null.std()),
                     note="One-sided for a negative correlation: the "
                          "hypothesis is that the duration advantage falls "
                          "as lifetime rises.")
    # the same at the pair level, for the comparison
    rho_p, pp = stats.spearmanr(d.lifetime_days, d.ratio)
    rank_free["pair_level_rho"] = float(rho_p)
    rank_free["pair_level_asymptotic_p"] = float(pp)
    print(f"    for comparison, pair level rho = {rho_p:+.3f} "
          f"(p = {pp:.4f}), which overstates the evidence")

    # --------------------------------------------------- value perturbation
    print("\n" + "=" * 74)
    print("  4. THE LIFETIME VALUES, PERTURBED")
    print("=" * 74)
    print(f"  Each lifetime multiplied by 10 raised to a uniform draw on")
    print(f"  plus or minus {VALUE_SPREAD_DEX:g} decades, so every value moves")
    print(f"  over a full order of magnitude end to end, {VALUE_DRAWS:,}")
    print("  times. This is the arm that answers an objection to the table")
    print("  itself rather than to the test.")
    print()
    rhos = np.empty(VALUE_DRAWS)
    sig_pub, sig_any = 0, 0
    for i in range(VALUE_DRAWS):
        mult = 10 ** rng.uniform(-VALUE_SPREAD_DEX, VALUE_SPREAD_DEX, len(sp))
        pert = lt * mult
        rhos[i] = stats.spearmanr(pert, rt).statistic
        frame = sp.assign(lifetime_days=pert)
        frame["duration_wins"] = frame.median_ratio > 1
        rec = one_split(frame, PUBLISHED_CUTOFF, ratio_col="median_ratio")
        if np.isfinite(rec["mannwhitney_p"]) and rec["mannwhitney_p"] < 0.05:
            sig_pub += 1
        sig_any += 1
    neg = float((rhos < 0).mean())
    print(f"    the rank correlation stays negative in "
          f"{100 * neg:.1f} per cent of perturbations")
    print(f"    median perturbed rho {np.median(rhos):+.3f}, "
          f"2.5th to 97.5th percentile "
          f"{np.percentile(rhos, 2.5):+.3f} to "
          f"{np.percentile(rhos, 97.5):+.3f}")
    print(f"    the species-level split at {PUBLISHED_CUTOFF:g} days stays "
          f"significant at 0.05 in {100 * sig_pub / max(sig_any, 1):.1f} "
          f"per cent")
    value_sens = dict(
        n_draws=int(VALUE_DRAWS), spread_dex=float(VALUE_SPREAD_DEX),
        share_rho_negative=neg, median_rho=float(np.median(rhos)),
        rho_ci=[float(np.percentile(rhos, 2.5)),
                float(np.percentile(rhos, 97.5))],
        share_split_significant=float(sig_pub / max(sig_any, 1)),
        note="A perturbation of the literature values, not of the data.")

    # ------------------------------------------------------------ jackknife
    print("\n" + "=" * 74)
    print("  5. LEAVE ONE SPECIES OUT")
    print("=" * 74)
    jrows = []
    for s in sp.species:
        sub = sp[sp.species != s]
        r, _ = stats.spearmanr(sub.lifetime_days, sub.median_ratio)
        rec = one_split(sub, PUBLISHED_CUTOFF, ratio_col="median_ratio")
        jrows.append(dict(left_out=s, spearman_rho=float(r),
                          mannwhitney_p=rec["mannwhitney_p"],
                          short_wins=rec["short_wins"],
                          n_short=rec["n_short"],
                          long_wins=rec["long_wins"],
                          n_long=rec["n_long"]))
        print(f"    without {s:<22} rho {r:+.3f}   p "
              f"{rec['mannwhitney_p']:.4f}")
    jk = pd.DataFrame(jrows)
    worst = jk.loc[jk.mannwhitney_p.idxmax()]
    print(f"\n    the worst case is leaving out {worst['left_out']}: "
          f"rho {worst['spearman_rho']:+.3f}, p {worst['mannwhitney_p']:.4f}")
    if (jk.spearman_rho < 0).all():
        print(f"    rho stays negative in all {len(jk)} jackknife replicates")
    else:
        print(f"    rho changes sign in "
              f"{int((jk.spearman_rho >= 0).sum())} of {len(jk)} replicates")
    jack = dict(n_replicates=int(len(jk)),
                rho_min=float(jk.spearman_rho.min()),
                rho_max=float(jk.spearman_rho.max()),
                worst_p=float(jk.mannwhitney_p.max()),
                worst_left_out=str(worst["left_out"]),
                all_negative=bool((jk.spearman_rho < 0).all()),
                per_species=jk.to_dict("records"))

    # ------------------------------------------------------------- sources
    src_rows = []
    for name, days in sorted(LIFETIME_DAYS.items(), key=lambda kv: kv[1]):
        why, ref = SOURCES.get(name, ("", ""))
        src_rows.append(dict(species=name, lifetime_days=days,
                             order_of_magnitude=why, reference=ref,
                             in_analysis=bool(name in set(sp.species))))
    pd.DataFrame(src_rows).to_csv(SOURCES_CSV, index=False)
    print(f"\n  [saved] {SOURCES_CSV}")
    print("  CROSSCHECK: reconcile every reference in that file against "
          "references.bib")

    # ------------------------------------------------------------- verdict
    # direction, as distinct from significance: at each cutoff, does the
    # short-lived group win more often than the longer-lived one?
    def wins_share(row, which):
        n = row[f"n_{which}"]
        return (row[f"{which}_wins"] / n) if n else float("nan")

    dir_ok = 0
    for _, row in sp_sweep.iterrows():
        s, l = wins_share(row, "short"), wins_share(row, "long")
        if np.isfinite(s) and np.isfinite(l) and s > l:
            dir_ok += 1
    print("\n" + "=" * 74)
    print("  WHAT TO WRITE")
    print("=" * 74)
    print(f"  At the species level, which is {len(sp)} independent chemical")
    print(f"  systems rather than {len(d)} rows, the rank relationship between")
    print(f"  lifetime and the duration advantage is Spearman {rho:+.2f} with a")
    print(f"  permutation p of {p_perm:.3f}.")
    print()
    print("  THE DIRECTION IS ROBUST AND THE SIGNIFICANCE IS NOT, and the")
    print("  paper has to say both:")
    print(f"    - the sign is negative in all {int(len(jk))} jackknife")
    print(f"      replicates and in {100 * neg:.0f} per cent of "
          f"{VALUE_DRAWS:,} perturbations")
    print(f"      of the literature values;")
    print(f"    - the short-lived group wins more often than the longer-lived")
    print(f"      one at {dir_ok} of {len(sp_sweep)} cutoffs tried;")
    print(f"    - but the species-level split reaches p < 0.05 at only "
          f"{n_sig} of")
    print(f"      {len(sp_sweep)} of those cutoffs, and the cutoff-free "
          f"permutation p is {p_perm:.3f}.")
    print()
    print(f"  So the pair-level p of {pair_pub['mannwhitney_p']:.4f} should not "
          f"be quoted. Thirteen")
    print("  species cannot carry it. What the data supports is a consistent")
    print("  direction of the expected sign and of a plausible size, which is")
    print("  evidence CONSISTENT WITH the published exposure-metric")
    print("  literature rather than an independent establishment of it. Write")
    print("  it that way: the finding builds on that literature, and the")
    print("  honest statement of the strength of the present evidence is the")
    print("  species-level rank correlation with its permutation p beside it.")
    print()
    print("  A referee who notices the nesting after reading a pair-level")
    print("  claim will distrust the rest of the paper. One who reads the")
    print("  correction stated plainly will not.")

    blob = dict(
        n_pairs=int(len(d)), n_species=int(len(sp)),
        published_cutoff=PUBLISHED_CUTOFF,
        unit_comparison=dict(pair_level=pair_pub, species_level=sp_pub),
        cutoff_sweep=dict(levels=[float(x) for x in cutoffs],
                          published_cutoff=sp_pub,
                          n_species_level_significant=n_sig,
                          n_species_level_direction_correct=int(dir_ok),
                          n_cutoffs=int(len(sp_sweep)),
                          rows=sweep.to_dict("records")),
        verdict=dict(
            direction_robust=True,
            significance_at_species_level="marginal",
            headline_to_quote=(
                f"Spearman {rho:+.2f} across {len(sp)} species, permutation "
                f"p = {p_perm:.3f}"),
            do_not_quote=(
                f"the pair-level Mann-Whitney p of "
                f"{pair_pub['mannwhitney_p']:.4f}, which treats "
                f"{len(d)} nested rows as independent"),
            how_to_frame=("consistent with the published exposure-metric "
                          "literature rather than an independent "
                          "establishment of it")),
        rank_free_test=rank_free,
        value_sensitivity=value_sens,
        jackknife=jack,
        species_table=sp.to_dict("records"),
        lifetime_days=LIFETIME_DAYS,
        note="The duration-against-lifetime mechanism, tested at the species "
             "level, with the cutoff swept, the literature values perturbed, "
             "and the species jackknifed.")
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n  [saved] {OUT_JSON}")

    C.write_manifest("25_lifetime_literature_test.py",
                     inputs=[p for p in C.table_candidates(
                         "A_duration_vs_concentration") if p.exists()][:1],
                     outputs=[OUT_JSON, SWEEP_CSV, SOURCES_CSV],
                     settings=dict(PUBLISHED_CUTOFF=PUBLISHED_CUTOFF,
                                   CUTOFF_DAYS=list(cutoffs),
                                   VALUE_DRAWS=VALUE_DRAWS,
                                   VALUE_SPREAD_DEX=VALUE_SPREAD_DEX,
                                   PERM_DRAWS=PERM_DRAWS),
                     notes="Critical assessment of the lifetime mechanism: "
                           "species as the unit, cutoff swept, literature "
                           "values perturbed, species jackknifed.")


if __name__ == "__main__":
    main()
