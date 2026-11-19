#!/usr/bin/env python3
"""
build_paper_constants.py -- every number the final manuscript quotes from the
analysis battery, written as LaTeX macros.

This is the final-paper copy (version 28). It writes only the numbers and
tables the final manuscript prints; the blocks that read the census, the
attribution-instrument comparison, the permutation splits, the admission grid
and the multiplicity family are not included, and the background size of the
turning-point attribution is read from script 05's own record.

READS   the result files in the derived folder (see _config.DERIVED)
        main_run_constants/constants.tex, constants_v10.tex and
        constants_thresholds.tex, the main run's generated constants
        ../main.tex, to know which main-run constants the paper quotes
WRITES  ../constants_results.tex   every number from the analysis battery
        ../constants_base.tex      the main-run constants the paper quotes,
                                   resolved and flattened
        ../tables_generated_slim.tex  appendix Tables 3 and 4 (species,
                                   turning points)
        ../tables_learner.tex      appendix Table A1 (the learner)
        ../tables_data.tex         Table 1 (data sources and the outcome)

main.tex loads constants_base.tex and then constants_results.tex. Every macro
in the results file is defined with \\providecommand followed by
\\renewcommand and that file is loaded last, so where a name exists in both,
the value read from the current result files is the one printed. No number in
this script is typed in: each is read from a result file, and the file it came
from is written as a comment above it.

    python3 build_paper_constants.py
    python3 build_paper_constants.py --out ../constants_results.tex
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

LINES: list[str] = []
SOURCE = ""


def src(name):
    global SOURCE
    SOURCE = name
    LINES.append(f"\n% ---- {name}")


def js(name):
    src(name)
    return json.load(open(C.DERIVED / name))


def tab(name):
    src(name)
    return pd.read_csv(C.DERIVED / name)


def put(macro, value):
    LINES.append(f"\\providecommand{{\\{macro}}}{{}}"
                 f"\\renewcommand{{\\{macro}}}{{{value}}}")


def f(v, nd=3):
    """A plain decimal with a true minus sign."""
    s = f"{float(v):.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def n(v):
    """An integer with a thin thousands separator."""
    return f"{int(round(float(v))):,}".replace(",", "{,}")


def pct(v, nd=0):
    return f"{100 * float(v):.{nd}f}"


def p(v):
    """A p value: three significant figures above 0.001, scientific below."""
    v = float(v)
    if v >= 0.001:
        return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def sign_p(k, nn):
    """Exact two-sided sign test of k successes in nn against a fair coin.
    Every sign-test p value the manuscript prints comes from here, so all are
    two-sided whatever convention the result file that holds k used."""
    k, nn = int(k), int(nn)
    up = sum(comb(nn, i) for i in range(k, nn + 1)) / 2 ** nn
    lo = sum(comb(nn, i) for i in range(0, k + 1)) / 2 ** nn
    return min(1.0, 2 * min(up, lo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(HERE).parent / "constants_results.tex"))
    a = ap.parse_args()

    # ------------------------------------------------------------ baselines
    b = js("baselines.json")
    put("BlKnn", f(b["knn4_spatial"]["r2"]))

    br = tab("bracket_error.csv")
    lo = br.iloc[0]
    put("BrLowMAE", f(lo["mae"], 2))
    mid = br.loc[br["mae"].idxmin()]
    put("BrMidMAE", f(mid["mae"], 2))

    s = js("spatial_holdout.json")
    put("HoldStateRtwo", f(s["r2"]))

    r = js("rank_stability.json")
    lc = r["learning_curve"]
    put("LcTenRtwo", f(lc["010pct"]["r2"]))
    put("LcHalfRtwo", f(lc["050pct"]["r2"]))
    put("LcFullRtwo", f(lc["100pct"]["r2"]))

    ns = js("null_skill.json")
    put("NullSkillWithin", f(ns["within_county"]["r2"]))
    put("NullSkillBetween", f(ns["between_county"]["r2"]))

    t = js("temporal_validation.json")
    put("LoyoMean", f(t["leave_one_year_out"]["r2_mean"]))
    put("FwdMean", f(t["forward_chaining"]["r2_mean"]))

    inf = js("influence_drop.json")
    w5 = inf["arms"]["worst_residual_5pct"]
    put("InflDropN", n(w5["n_counties_dropped"]))
    put("InflDropRtwo", f(w5["r2"]))
    put("InflOverlap", n(inf["topk_overlap"]["shared"]))
    put("InflTopK", n(inf["topk_overlap"]["topk"]))

    mo = js("modality_arms.json")
    put("ArmAllRtwo", f(mo["arms"]["all_admitted"]["r2"]))
    put("ArmAllN", n(mo["arms"]["all_admitted"]["n_features"]))

    # ------------------------------------------- fields with several summaries
    sc = js("summary_statistic_census.json")
    put("CensusMedianSpread", f(sc["median_spread"], 1))

    # background rows of the attribution the turning points are located on
    sdc = js("shap_dependence_check.json")
    put("InvShapBackground", n(sdc["background"]))

    # ------------------------------------------------------- lifetime
    lt = js("lifetime_literature_test.json")
    rf = lt["rank_free_test"]
    put("LtSpecies", n(lt["n_species"]))
    put("LtPairs", n(lt["n_pairs"]))
    put("LtRho", f(rf["spearman_rho"], 2))
    put("LtPermP", p(rf["permutation_p"]))
    put("LtPerms", n(rf["n_permutations"]))
    put("LtCutoff", f"{lt['published_cutoff']:g}")
    u = lt["unit_comparison"]["species_level"]
    put("LtShortWins", n(u["short_wins"]))
    put("LtShortN", n(u["n_short"]))
    put("LtLongWins", n(u["long_wins"]))
    put("LtLongN", n(u["n_long"]))
    put("LtShortRatio", f(u["short_median_ratio"], 2))
    put("LtLongRatio", f(u["long_median_ratio"], 2))
    sw = lt["cutoff_sweep"]
    put("LtSweepN", n(sw["n_cutoffs"]))
    put("LtSweepDirection", n(sw["n_species_level_direction_correct"]))
    put("LtSweepSig", n(sw["n_species_level_significant"]))
    jj = lt["jackknife"]
    put("LtJackLo", f(jj["rho_min"], 2))
    put("LtJackHi", f(jj["rho_max"], 2))
    vs = lt["value_sensitivity"]
    put("LtValueDraws", n(vs["n_draws"]))
    put("LtValueDex", f(vs["spread_dex"], 1))
    put("LtValueNegative", pct(vs["share_rho_negative"]))

    # ------------------------------------------------------- turning points
    tp = tab("turning_points.csv").set_index("feature")
    put("TpBoot", n(tp["n_boot_attempted"].max()))
    put("TpStable", n(tp["stable"].astype(bool).sum()))
    put("TpTotal", n(len(tp)))
    for feat, mac, nd in (("prism_tdmean_min", "TpDew", 2),
                          ("eng2_wetbulb_night", "TpWet", 2),
                          ("eng_night_cooling_eff", "TpNce", 3),
                          ("eng2_night_cooling_efficiency", "TpNceMean", 3),
                          ("lst_night_1km_mean", "TpLst", 2),
                          ("pm25_mean", "TpPm", 2),
                          ("prism_tmin_min", "TpTmin", 2),
                          ("cams_fot_formaldehyde_above75th_percentile", "TpHcho", 1),
                          ("eng2_hcho_no2_ratio", "TpRatio", 3)):
        rr = tp.loc[feat]
        put(mac, f(rr["point_estimate"], nd))
        put(mac + "Lo", f(rr["ci_lo"], nd))
        put(mac + "Hi", f(rr["ci_hi"], nd))
        put(mac + "Width", f(100 * rr["width_frac"], 1))
        put(mac + "Above", f(rr["pct_above"], 1))
    trip = tp.loc[["prism_tdmean_min", "eng2_wetbulb_night",
                   "eng_night_cooling_eff"], "width_frac"]
    put("TpTripletWidthMax", f(100 * trip.max(), 1))

    # --------------------------- composition, dew point and modalities
    # The production ranking (the main run's out-of-fold SHAP ranking) gives
    # the chemical-family shares, the 75th against 90th percentile durations
    # and the seven summaries of dew point; the main run's temperature strata
    # and modality leave-one-out tables give the rest. The family rules are
    # the ones the composition figure draws (paper_figures.FAMILY_RULES).
    LINES.append(f"\n% ---- {C.RANKING_A.name} (production ranking)")
    rk = pd.read_csv(C.RANKING_A)
    rnk = dict(zip(rk["feature"], rk["rank"]))
    val = dict(zip(rk["feature"], rk["mean_abs_shap"]))
    from paper_figures import cams_family                      # noqa: E402
    comp = rk[rk["feature"].str.startswith("cams")].copy()
    comp["family"] = comp["feature"].map(cams_family)
    tot = comp["mean_abs_shap"].sum()
    fam = comp.groupby("family")["mean_abs_shap"].agg(["sum", "size"])
    fam = fam.sort_values("sum", ascending=False)
    put("CompN", n(len(comp)))
    put("FamN", n(len(fam)))
    put("FamOxVOCPct", f(100 * fam.loc["Oxidised VOC", "sum"] / tot, 1))
    put("FamOxVOCn", n(fam.loc["Oxidised VOC", "size"]))
    put("FamSecondName", fam.index[1].lower())
    put("FamSecondPct", f(100 * fam["sum"].iloc[1] / tot, 1))
    hcho = comp[comp["feature"].str.contains("formaldehyde")]
    put("FamHchoPct", f(100 * hcho["mean_abs_shap"].sum() / tot, 1))
    put("FamHchoN", n(len(hcho)))
    p90 = "cams_fot_formaldehyde_above_90th_percentile"
    put("HchoNinetyRank", n(rnk[p90]))
    du = comp[comp["feature"].str.contains("_fot_")
              & comp["feature"].str.contains("percentile")].copy()
    du["pct"] = np.where(du["feature"].str.contains("75th"), 75, 90)
    g = du.groupby("pct")["mean_abs_shap"].agg(["mean", "size"])
    put("PctRatio", f(g.loc[75, "mean"] / g.loc[90, "mean"], 1))
    du["species"] = du["feature"].str.replace(r"_?above_?\d+th_percentile", "",
                                              regex=True)
    pv = du.pivot_table(index="species", columns="pct",
                        values="mean_abs_shap").dropna()
    put("PctSpeciesN", n(len(pv)))
    put("PctSpeciesWins", n((pv[75] > pv[90]).sum()))
    put("PctSpeciesMedRatio", f((pv[75] / pv[90]).median(), 2))
    pm = pv[pv.index.str.contains(r"pm")]
    put("PctPmLo", f((pm[90] / pm[75]).min(), 1))
    put("PctPmHi", f((pm[90] / pm[75]).max(), 1))
    nohcho = du[~du["feature"].str.contains("formaldehyde")]
    g2 = nohcho.groupby("pct")["mean_abs_shap"].mean()
    put("PctRatioNoHcho", f(g2.loc[75] / g2.loc[90], 1))
    put("HchoPctRatio", f(val["cams_fot_formaldehyde_above75th_percentile"] / val[p90], 1))

    # a static field for contrast: soil bulk density, 90th percentile and mean
    put("SoilPninetyRank", n(rnk["soil_bulk_density_30cm_p90"]))
    put("SoilMeanRank", n(rnk["soil_bulk_density_30cm_mean"]))
    put("SoilRatio", f(val["soil_bulk_density_30cm_p90"]
                       / val["soil_bulk_density_30cm_mean"], 1))

    dpp = rk[rk["feature"].str.startswith("prism_tdmean_")].sort_values("rank")
    put("DPSummariesN", n(len(dpp)))
    put("DPMinRank", n(rnk["prism_tdmean_min"]))
    put("DPNextRank", n(dpp["rank"].iloc[1]))
    put("DPMeanRank", n(rnk["prism_tdmean_mean"]))
    put("DPProfileRatio", f(dpp["mean_abs_shap"].iloc[0] / dpp["mean_abs_shap"].iloc[1], 1))

    st = C.load_table("A_temp_strata")
    LINES.append("\n% ---- A_temp_strata.csv (main run)")
    crossing = st[np.isfinite(st["zero_cross"])]
    put("StrataCrossLo", f(crossing["zero_cross"].min(), 2))
    put("StrataCrossHi", f(crossing["zero_cross"].max(), 2))

    loo = C.load_table("A_modality_loo").sort_values("delta")
    LINES.append("\n% ---- A_modality_loo.csv (main run)")
    put("LOOnModalities", n(len(loo)))
    put("LOOcams", f(loo["delta"].iloc[0], 4))
    put("LOOcamsN", n(loo["n_removed"].iloc[0]))

    nx = js("null_crossing.json")["overall"]
    put("NullCrossTotal", n(nx["total_permutations"]))
    put("NullCrossPasses", n(nx["total_gate_passes"]))
    put("NullCrossDraws", n(nx["n_draws_each"]))
    put("NullCrossBoot", n(nx["n_boot_per_draw"]))
    put("NullCrossShifted", n(nx["shifted_any_pass"]))

    pt = tab("perturbation_thresholds.csv")
    agg = pt.groupby("magnitude")["shift_over_interval_width"].agg(
        med=lambda s: s.abs().median(), mx=lambda s: s.abs().max())
    for mag, tag in ((0.01, "One"), (0.02, "Two"), (0.05, "Five"), (0.10, "Ten")):
        put(f"Pert{tag}Med", f(agg.loc[mag, "med"], 2))
        put(f"Pert{tag}Max", f(agg.loc[mag, "mx"], 2))
    put("PertDraws", n(pt["n_found"].max()))

    # ------------------------------------------------------- regional invariance
    for name, tag in (("threshold_regional_invariance_all.json", "Med"),
                      ("threshold_regional_invariance_100W_all.json", "Mer")):
        ri = js(name)
        cut = abs(ri["cut_degrees_east"])
        put(f"Reg{tag}Cut", f"{cut:g}" if cut == int(cut) else f(cut, 1))
        res = ri["results"]
        anyf = next(iter(res.values()))
        put(f"Reg{tag}West", n(anyf["n_west"]))
        put(f"Reg{tag}East", n(anyf["n_east"]))
        put(f"Reg{tag}Matched", n(anyf["n_west"] + anyf["n_east"]))
        for feat, mac in (("prism_tdmean_min", "Dew"),
                          ("eng2_wetbulb_night", "Wet"),
                          ("eng_night_cooling_eff", "Nce"),
                          ("lst_night_1km_mean", "Lst"),
                          ("pm25_mean", "Pm"),
                          ("eng2_hcho_no2_ratio", "Ratio"),
                          ("eng2_wetbulb_hot", "WetHot"),
                          ("prism_tmin_min", "Tmin")):
            d = res[feat].get("delta_pct_of_range")
            put(f"Reg{tag}{mac}", "n/a" if d is None else f(d, 1))
            if feat in ("prism_tdmean_min", "eng2_wetbulb_night",
                        "eng_night_cooling_eff") and d is not None:
                put(f"Reg{tag}{mac}Abs", f(res[feat]["delta"],
                                            3 if feat.startswith("eng_") else 2))
        trip = [res[k].get("delta_pct_of_range") for k in
                ("prism_tdmean_min", "eng2_wetbulb_night", "eng_night_cooling_eff")]
        put(f"Reg{tag}TripletMax", f(max(trip), 1))

    # ------------------------------------------------------- failure geography
    rs = js("residual_social_determinants.json")
    put("ResSdohRtwo", f(rs["oof_r2_residual_from_sdoh"], 2))
    put("ResEnvRtwo", f(rs["oof_r2_residual_from_environment"], 2))
    put("ResSdohN", n(rs["n_sdoh_columns"]))
    put("ResEnvN", n(rs["n_env_columns"]))
    put("ResSD", f(rs["residual_sd"], 2))
    put("ResMissN", n(rs["n_large_miss"]))
    put("ResMissYears", f"{rs['large_miss_threshold']:g}")
    ind = {i["column"]: i for i in rs["indicators"]}
    for col, mac in (("sdoh_saipe_pct_pov", "Pov"),
                     ("sdoh_saipe_median_hh_income", "Inc"),
                     ("sdoh_unemployment_rate", "Unemp"),
                     ("sdoh_acs_pct_uninsured", "Unins"),
                     ("sdoh_bachelor's_degree_or_higher_%", "Bach"),
                     ("sdoh_disability_rate", "Dis")):
        put(f"Res{mac}Rho", f(ind[col]["rho"], 2))
        put(f"Res{mac}Pct", f(rs["large_miss_profile"][col]
                               ["median_percentile_large_miss"], 0))
    put("ResHolmMax", p(max(i["p_holm"] for i in rs["indicators"])))
    put("ResAbsRhoLo", f(min(abs(i["rho"]) for i in rs["indicators"]), 2))
    put("ResAbsRhoHi", f(max(abs(i["rho"]) for i in rs["indicators"]), 2))

    sens_path = C.DERIVED / "residual_audit_sensitivity.json"
    if sens_path.exists():                       # written by script 30
        sv = json.loads(sens_path.read_text())
        by_src = sv["columns_by_source"]
        put("ResSdohYearLo", str(sv["years"][0]))
        put("ResSdohYearHi", str(sv["years"][1]))
        put("ResAcsN", n(by_src.get("American Community Survey", 0)))
        arms = sv["arms"]
        put("ResNoChrN", n(arms["without_county_health_rankings"]["n_columns"]))
        put("ResNoChrRtwo", f(arms["without_county_health_rankings"]["oof_r2"], 2))
        put("ResAcsRtwo", f(arms["american_community_survey_only"]["oof_r2"], 2))

    # Moran's I of the production model's own residual (the residual the
    # audit uses and Figure 8 maps), and of the outcome for comparison
    mi = js("residual_spatial_structure.json")
    put("ResMoranI", f(mi["production"]["morans_i"], 3))
    put("ResMoranP", p(mi["production"]["p_value"]))
    put("ResMoranOutcome", f(mi["outcome"]["morans_i"], 2))

    # refits without the formaldehyde durations (step 17)
    fd = js("formaldehyde_drop.json")
    put("HchoDropFullRtwo", f(fd["r2"]["both"], 4))
    put("HchoDropOneRtwo", f(fd["r2"]["without_p75"], 4))
    put("HchoDropBothRtwo", f(fd["r2"]["without_both"], 4))
    put("HchoDropInheritPct", f(100 * fd["inherit_p90_of_p75"], 0))
    put("HchoDropShapN", n(fd["shap_rows"]))
    put("HchoDropShapBg", n(fd["shap_background"]))
    col = "cams_total_column_formaldehyde_x1e9"
    put("HchoColRankWith", n(fd["rank"][col]["both"]))
    put("HchoColRankWithout", n(fd["rank"][col]["without_both"]))
    put("HchoDropBothCost", f(fd["r2"]["both"] - fd["r2"]["without_both"], 3))
    surf = "cams_formaldehyde_x1e9"
    put("HchoSurfRankWith", n(fd["rank"][surf]["both"]))
    put("HchoSurfRankWithout", n(fd["rank"][surf]["without_both"]))

    # The reference of every SHAP value: the model's average prediction. A
    # turning point is where a variable's contribution crosses this level.
    LINES.append(f"\n% ---- {C.OOF_A.name}")
    oof = pd.read_csv(C.OOF_A)
    put("PredMean", f(oof["pred"].mean(), 1))

    # The reference configuration every refit uses (scripts 04, 12, 22).
    LINES.append("\n% ---- nce_refit.json (refit protocol)")
    proto = json.load(open(C.DERIVED / "nce_refit.json"))["protocol"]
    put("RefitTrees", n(proto["n_estimators"]))
    put("RefitLR", f(proto["learning_rate"], 2))
    put("RefitDepth", n(proto["max_depth"]))
    put("RefitShapN", n(proto["nshap"]))
    put("RefitShapBg", n(proto["nbg"]))
    cond = json.load(open(C.DERIVED / "nce_refit.json"))["conditions"]
    put("RefitFullTailSHAP", f(cond["FULL"]["pct_shap"], 4))

    # Library versions of the final run, and how much the numbers moved
    # against the earlier run under other releases (script 34).
    lv_path = C.DERIVED / "library_version_sensitivity.json"
    if lv_path.exists():
        LINES.append(f"\n% ---- {lv_path.name}")
        lv = json.loads(lv_path.read_text())
        vf, vr = lv["versions_final"], lv["versions_reference"]
        for mac, key in (("VerPython", "python"), ("VerXgb", "xgboost"),
                         ("VerShap", "shap"), ("VerSklearn", "sklearn"),
                         ("VerNumpy", "numpy"), ("VerPandas", "pandas"),
                         ("VerScipy", "scipy"), ("VerCatboost", "catboost")):
            put(mac, str(vf.get(key, "")))

    write_tables(Path(a.out).parent / "tables_generated_slim.tex", tp)
    write_data_table(Path(a.out).parent / "tables_data.tex")
    manuscript = Path(a.out).parent / "main.tex"
    if manuscript.exists():
        names = {re.match(r"\\providecommand\{\\([A-Za-z]+)\}", x).group(1)
                 for x in LINES if x.startswith("\\providecommand")}
        write_base(Path(a.out).parent / "constants_base.tex", manuscript, names)

    head = ("% constants_results.tex -- written by analysis/build_paper_constants.py\n"
            "% Do not edit by hand: rerun the script. Each block names the result\n"
            "% file its values were read from.\n")
    Path(a.out).write_text(head + "\n".join(LINES) + "\n")
    print(f"  [saved] {a.out}  ({sum(1 for x in LINES if x.startswith(chr(92)))} macros)")
    check_text_assumptions(Path(a.out).parent)


# ---------------------------------------------------------------------------
#  Text-assumption checks
# ---------------------------------------------------------------------------
# Numbers in the manuscript are macros, so a rerun updates them everywhere.
# A few sentences, however, also make a qualitative statement about a number
# ("not significant", "weakly clustered", "barely move"). If a rerun moved a
# number across such a line, the macro would update but the sentence would
# become wrong. These checks list every such sentence and test it against the
# freshly written values. A failed check never stops the build: it prints
# WARN, names the section to reread, and is recorded in
# derived/text_assumption_checks.json.
TEXT_ASSUMPTIONS = [
    ("1, 3.3, 4.2, 5: lifetime correlation is inverse",
     lambda v: v["LtRho"] < 0),
    ("3.3, 4.2: the lifetime relation is 'modest' (one-sided p not below 0.05)",
     lambda v: v["LtPermP"] >= 0.05),
    ("3.3, 5: the oxygenated family leads the composition attribution",
     lambda v: v["FamOxVOCPct"] > v["FamSecondPct"]),
    ("3.3: the 75th-percentile advantage is carried by formaldehyde",
     lambda v: v["PctRatioNoHcho"] < 1.5 and v["PctSpeciesWins"] <= v["PctSpeciesN"] / 2 + 1),
    ("3.2, 5: minimum dew point leads the next dew-point summary several-fold",
     lambda v: v["DPProfileRatio"] > 5),
    ("3.1: removing composition costs the most skill",
     lambda v: v["LOOcams"] < 0),
    ("3.1, 4.4, 5: holding out whole states lowers skill",
     lambda v: v["HoldStateRtwo"] < v["Rtwo"]),
    ("3.5: only 10% noise moves a value by more than one interval width",
     lambda v: v["PertTenMax"] > 1 and v["PertFiveMax"] <= 1),
    ("3.5, 5: no permuted curve passes the stability criterion",
     lambda v: v["NullCrossPasses"] == 0),
    ("1, 3.5, 4.3, 5: the three moisture and night-cooling points move little "
     "between regions (<= 5% of range)",
     lambda v: v["RegMerTripletMax"] <= 5 and v["RegMedTripletMax"] <= 5),
    ("1, 3.5, 4.3, 5: at the 100th meridian two of them shift by more than their "
     "own interval widths",
     lambda v: v["RegMerDew"] > v["TpDewWidth"] and v["RegMerNce"] > v["TpNceWidth"]),
    ("1, 3.5, 5: satellite PM2.5 moves more than the three moisture and "
     "night-cooling points",
     lambda v: v["RegMerPm"] > v["RegMerTripletMax"]),
    ("3.5, 4.3: minimum temperature also moves little at the 100th meridian (<= 5%)",
     lambda v: v["RegMerTmin"] <= 5),
    ("3.5, 4.3: the formaldehyde to NO2 ratio moved at the median longitude (> 5%)",
     lambda v: v["RegMedRatio"] > 5),
    ("2.10, 3.4: crossings lie near the middle of each distribution (25 to 75%)",
     lambda v: all(25 <= v[k] <= 75 for k in ("TpDewAbove", "TpWetAbove", "TpNceAbove"))),
    ("3.6, 4.4: residuals 'weakly but significantly' clustered",
     lambda v: v["ResMoranP"] < 0.05 and abs(v["ResMoranI"]) < 0.1),
    ("3.6: the residual is far less clustered than the outcome",
     lambda v: v["ResMoranI"] < 0.2 * v["ResMoranOutcome"]),
    ("4.4, 5: socioeconomic columns explain under half of the residual variance",
     lambda v: v["ResSdohRtwo"] < 0.5),
    ("1, 3.2, 5: removing both formaldehyde durations costs little skill (< 0.01)",
     lambda v: 0 <= v["HchoDropFullRtwo"] - v["HchoDropBothRtwo"] < 0.01),
    ("3.2, 4, 5: the drop-one NCE refit changes skill by less than 0.01",
     lambda v: abs(v["RefitFullRtwo"] - v["RefitDropRtwo"]) < 0.01),
    ("3.1, 5: error is largest in the lowest bracket",
     lambda v: v["BrLowMAE"] > v["BrMidMAE"]),
    ("2.8: refit configuration reproduces production skill (within 0.01)",
     lambda v: abs(v["ArmAllRtwo"] - v["Rtwo"]) < 0.01),
]


def _macro_values(folder):
    """Read every macro written to the constants files as a float where the
    value is a number (signs, thin separators and 10^x forms understood)."""
    vals = {}
    rx = re.compile(r"\\(?:renewcommand|newcommand)\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
    for name in ("constants_base.tex", "constants_results.tex"):
        path = Path(folder) / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            m = rx.search(line)
            if m and m.group(2):
                s = m.group(2).replace("{,}", "").replace("$-$", "-").replace("$", "")
                s = s.replace("<", "").replace("\\,", "")
                mm = re.match(r"^\s*(-?[0-9.]+)\\times10\^\{(-?\d+)\}?\s*$", s)
                try:
                    vals[m.group(1)] = (float(mm.group(1)) * 10 ** int(mm.group(2))
                                        if mm else float(s))
                except ValueError:
                    pass
    return vals


def check_text_assumptions(folder):
    vals = _macro_values(folder)
    out, bad = [], 0
    print("\n  TEXT-ASSUMPTION CHECKS (a sentence that states a quality of a number)")
    for label, test in TEXT_ASSUMPTIONS:
        try:
            ok = bool(test(vals))
            status = "PASS" if ok else "WARN"
        except KeyError as e:
            ok, status = False, f"MISSING {e}"
        bad += not ok
        out.append(dict(check=label, status=status))
        print(f"    {status:<8} {label}")
    if bad:
        print(f"  {bad} sentence(s) to reread before the paper is rebuilt.")
    else:
        print("  every qualitative statement still matches its number.")
    (C.DERIVED / "text_assumption_checks.json").write_text(
        json.dumps(dict(checks=out, n_warn=bad), indent=1))


# ---------------------------------------------------------------------------
#  The base constants: the main run's generated values, flattened
# ---------------------------------------------------------------------------
BASE_SOURCES = ("constants.tex", "constants_v10.tex", "constants_thresholds.tex")
_HEAD = re.compile(r"\\(newcommand|renewcommand|providecommand)\{\\([A-Za-z]+)\}\{")


def resolve_tex_macros(paths):
    """Resolve macro definitions the way TeX would, in file order:
    \\providecommand only when undefined, \\newcommand and \\renewcommand
    always. Returns {name: raw TeX value}. Lines that are comments are
    skipped; a definition whose name is not a valid control word is ignored,
    as is any stray text between definitions."""
    vals = {}
    for path in paths:
        text = "\n".join("" if ln.lstrip().startswith("%") else ln.split("%%")[0]
                         for ln in Path(path).read_text().splitlines())
        pos = 0
        while True:
            m = _HEAD.search(text, pos)
            if not m:
                break
            depth, i = 1, m.end()
            while depth and i < len(text):
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            value, pos = text[m.end():i - 1], i
            if m.group(1) == "providecommand" and m.group(2) in vals:
                continue
            if value == "" and m.group(1) == "providecommand":
                vals.setdefault(m.group(2), value)
                continue
            vals[m.group(2)] = " ".join(value.split())
    return vals


def write_base(out_path, manuscript, results_names):
    """constants_base.tex: every main-run macro the manuscript uses and this
    script does not itself compute, resolved from the main run's generated
    constants files and written flat, one definition each."""
    # In the paper folder the main run's constants files are kept, unchanged,
    # in analysis/main_run_constants/; in the working tree they sit beside
    # main.tex. Whichever exists is used.
    src_dir = Path(HERE) / "main_run_constants"
    if not src_dir.exists():
        src_dir = C.PAPER
    found = [src_dir / n for n in BASE_SOURCES if (src_dir / n).exists()]
    if not found:
        print("  [kept] constants_base.tex: the main run's constants files were "
              "not found, so the existing file is left as it is")
        return
    base = resolve_tex_macros(found)
    used = set(re.findall(r"\\([A-Za-z]+)", Path(manuscript).read_text()))
    # the schematic figures print constants too; keep the ones they name
    schematic = Path(HERE) / "figure01_methodology_and_graphical_abstract.py"
    if schematic.exists():
        used |= set(re.findall(r"K\[['\"]([A-Za-z]+)['\"]\]",
                               schematic.read_text()))
    keep = sorted(k for k in used if k in base and k not in results_names)
    lines = ["% constants_base.tex -- written by analysis/build_paper_constants.py",
             "% The main run's generated values that the manuscript quotes,",
             "% resolved from " + ", ".join(BASE_SOURCES) + " in",
             "% analysis/main_run_constants/ (written by the main run's constants",
             "% builder) and flattened to one definition each. Do not edit by hand.",
             ""]
    lines += [f"\\newcommand{{\\{k}}}{{{base[k]}}}" for k in keep]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  [saved] {out_path}  ({len(keep)} macros)")


UNIT_TEX = {"degC": "$^{\\circ}$C", "%": "\\% of year", "ug/m3": "\\textmu g\\,m$^{-3}$"}
UNIT_OTHER = {"usda_cropland_usda_corn_pct": "fraction of area"}
SHORT_TEX = {
    "prism_tdmean_min": "Minimum dew point",
    "eng2_night_cooling_efficiency": "NCE, mean form",
    "eng_night_cooling_eff": "NCE, tail form",
    "lst_night_1km_mean": "Mean night LST",
    "eng2_wetbulb_night": "Nocturnal wet-bulb temperature",
    "eng2_wetbulb_hot": "Warm-hour wet-bulb temperature",
    "eng2_hcho_no2_ratio": "Formaldehyde to nitrogen dioxide ratio",
    "pm25_mean": "Annual mean PM$_{2.5}$",
    "prism_tmin_min": "Minimum daily minimum temperature",
    "cams_fot_formaldehyde_above75th_percentile": "Formaldehyde, time above 75th percentile",
    "usda_cropland_usda_corn_pct": "Cropland in corn",
}


def write_tables(path, tp):
    """Two appendix tables, generated so no value in them is typed in."""
    out = ["% tables_generated.tex -- written by analysis/build_paper_constants.py",
           "% Do not edit by hand.", ""]

    # Table A4: every located turning point and the three criteria
    found_all = bool((tp["found_frac"] >= 1).all())
    out += [r"\begin{table}[H]", r"\caption{Every candidate turning point. The "
            r"interval is the 2.5th to 97.5th percentile of the located value over "
            r"the bootstrap resamples, and its width is given as a share of the "
            r"variable's observed range. A candidate is stable when a crossing is "
            r"found in at least 60\% of resamples, the width is below 40\% and the "
            r"full-sample value lies inside the interval. PM$_{2.5}$ is given in "
            r"the product's native unit."
            + (" A crossing was found in every resample for every candidate."
               if found_all else "") + r"\label{tab:turning}}",
            r"\centering\footnotesize",
            r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
            r"Variable & Unit & Value & 95\% interval & Width (\%) & Stable \\",
            r"\midrule"]
    for feat, r in tp.sort_values("width_frac").iterrows():
        unit = UNIT_TEX.get(str(r["unit"]), UNIT_OTHER.get(feat, "dimensionless"))
        # PM2.5 is stored at one tenth of the product's native unit; the table
        # prints it in micrograms per cubic metre so no factor is left to read
        k = 10.0 if str(r["unit"]) == "ug/m3" else 1.0
        nd = 3 if abs(k * r["point_estimate"]) < 2 else 2
        out.append(f"{SHORT_TEX.get(feat, feat)} & {unit} & "
                   f"{f(k * r['point_estimate'], nd)} & "
                   f"{f(k * r['ci_lo'], nd)} to {f(k * r['ci_hi'], nd)} & "
                   f"{f(100 * r['width_frac'], 1)} & "
                   f"{'yes' if bool(r['stable']) else 'no'} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table A3: the species of the lifetime test (script 25)
    lt = json.load(open(C.DERIVED / "lifetime_literature_test.json"))
    sp = sorted(lt["species_table"], key=lambda r: r["lifetime_days"])
    names = {"pm2 5": "Fine particulate matter (reanalysis)",
             "nitrogen monoxide": "Nitric oxide"}
    out += [r"\begin{table}[H]", r"\caption{The species of the lifetime test, "
            r"ordered by the atmospheric lifetime taken from the literature (the "
            r"tropospheric burden divided by the removal rate, an e-folding time; "
            r"the half-life is this lifetime times $\ln 2$), "
            r"with the number of duration and concentration pairs admitted for "
            r"each, the median ratio of duration to concentration importance and "
            r"whether duration was credited more. All series are from the "
            r"atmospheric composition reanalysis. For nitric oxide the lifetime "
            r"is that of nitric oxide and nitrogen dioxide together, because the two "
            r"interconvert within minutes in daylight.\label{tab:species}}",
            r"\centering\footnotesize",
            r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
            r"Species & Lifetime (days) & Pairs & Median ratio & Duration favoured \\",
            r"\midrule"]
    for r in sp:
        nm = names.get(r["species"], r["species"].capitalize())
        lt_d = r["lifetime_days"]
        lt_s = f"{lt_d:.0e}".replace("e-0", "$\\times 10^{-") + "}$" if lt_d < 0.01 else f"{lt_d:g}"
        out.append(f"{nm} & {lt_s} & {r['n_rows']} & {f(r['median_ratio'], 2)} & "
                   f"{'yes' if r['duration_wins'] else 'no'} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table A1: the learner and its settings, from the tournament output
    ls = json.load(open(C.DERIVED / "learner_settings.json"))
    xp = ls["params"]["XGB"]
    names = {"n_estimators": "Number of trees", "max_depth": "Maximum depth",
             "learning_rate": "Learning rate", "subsample": "Row subsample",
             "colsample_bytree": "Column subsample per tree",
             "min_child_weight": "Minimum child weight",
             "reg_alpha": "L1 regularisation", "reg_lambda": "L2 regularisation"}
    out += [r"\begin{table}[H]", r"\caption{The fitted learner. Settings of the "
            r"XGBoost model and the out-of-fold skill of each learner compared on "
            r"identical county-grouped folds. Settings not listed are the "
            r"library defaults.\label{tab:learner}}",
            r"\centering\small", r"\begin{tabular}{@{}lr@{}}", r"\toprule",
            r"Setting & Value \\", r"\midrule"]
    for k, lab in names.items():
        if k not in xp:                    # not set: the library default applies
            continue
        v = xp[k]
        out.append(f"{lab} & {v if isinstance(v, int) else f'{v:.3g}'} \\\\")
    out.append(r"\midrule")
    out.append(r"Learner & Out-of-fold $R^{2}$ (mean absolute error, years) \\")
    out.append(r"\midrule")
    for k, v in ls["scores"].items():
        if "r2" in v:
            out.append(f"{k} & {v['r2']:.3f} ({v['mae']:.3f}) \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    # The learner table goes to its own file (Appendix A); the species and
    # turning-point tables, in that order, to the file Appendix B inputs.
    text = "\n".join(out) + "\n"
    head, *tbls = text.split("\\begin{table}[H]")
    blocks = {re.search(r"\\label\{(tab:[a-z]+)\}", b).group(1):
              "\\begin{table}[H]" + b for b in tbls}
    Path(path).write_text(
        head.replace("tables_generated.tex", "tables_generated_slim.tex")
        + "".join(blocks[k] for k in ("tab:species", "tab:turning")))
    lp = Path(path).with_name("tables_learner.tex")
    lp.write_text(head.replace("tables_generated.tex", "tables_learner.tex")
                  + blocks["tab:learner"])
    print(f"  [saved] {path} and {lp}")


# Table 1: every data source with its native resolution, and the number of its
# columns among those the model sees (counted from the model's own ranking).
DATA_SOURCES = [
    # name, variables (examples), spatial, time step, column prefixes
    ("MODIS land-surface temperature (Terra)", "Day and night land-surface temperature", "1 km", "8-day", ("lst_",)),
    ("MODIS vegetation indices", "Normalised difference and enhanced vegetation indices", "500 m", "16-day", ("ndvi_",)),
    ("Landsat 5 and 7", "Surface reflectance, vegetation indices", "30 m", "16-day revisit", ("ls57_",)),
    ("Sentinel-1", "Radar backscatter", "10 m", "6 to 12 days, from 2014", ()),
    ("Sentinel-2", "Optical reflectance", "10 to 60 m", "5 days, from 2015", ()),
    ("Sentinel-5P", "Trace gases, aerosol index", "About 5 km", "Daily, from 2018", ()),
    ("Satellite fine particulate matter", "Ground-level PM$_{2.5}$", "About 1 km", "Annual", ("pm25_",)),
    ("Night-time lights (DMSP, VIIRS)", "Settlement brightness, rescaled within each sensor", "About 1 km; 500 m", "Annual", ("ntl_",)),
    ("Global Surface Water (JRC)", "Water occurrence and class", "30 m", "Annual", ("jrc_",)),
    ("Global Human Modification", "Human pressure on land", "1 km", "Single map", ("ghm_",)),
    ("Canopy height", "Canopy height and cover", "1 m", "Single map", ("meta_",)),
    ("Cropland Data Layer (USDA)", "Share of land in each crop", "30 m", "Annual", ("usda_",)),
    ("CAMS global reanalysis (EAC4)", "Trace gases, aerosol optical depth, particulate matter", "0.75\\textdegree{} (about 80 km)", "3-hourly", ("cams_",)),
    ("ERA5 reanalysis", "Humidity, pressure, wind, snow, terrain height", "0.25\\textdegree{} (about 28 km)", "Hourly", ("era5_",)),
    ("PRISM", "Air temperature, dew point, vapour-pressure deficit, precipitation", "4 km", "Daily", ("prism_",)),
    ("TerraClimate", "Water balance, wind, solar radiation", "About 4 km", "Monthly", ("terra_",)),
    ("OpenLandMap soil maps", "Bulk density, texture, acidity, organic carbon", "250 m", "Single map", ("soil_",)),
    ("Gridded Livestock of the World", "Density of each farm animal", "About 10 km", "Reference years, linear between", ("live_",)),
    ("Engineered from the above", "Wet-bulb and heat indices, night cooling efficiency, species ratios", "Of the inputs", "Annual", ("eng_", "eng2_")),
]


def write_data_table(path):
    feats = pd.read_csv(C.RANKING_A)["feature"].astype(str).tolist()
    counts, seen = [], set()
    for row in DATA_SOURCES:
        hit = [x for x in feats if row[4] and x.startswith(row[4])]
        seen.update(hit)
        counts.append(len(hit))
    missing = sorted(set(feats) - seen)
    if missing or sum(counts) != len(feats):
        raise SystemExit(f"data table: columns not assigned to a source: {missing[:10]}")
    out = ["% tables_data.tex -- written by analysis/build_paper_constants.py",
           "% Column counts are read from the model's ranking; do not edit by hand.",
           r"\begin{table}[H]",
           r"\caption{Data sources and the outcome, with each product's native "
           r"resolution and the number of its columns among the "
           f"{len(feats)} the model sees. "
           r"Single maps enter every year with the same value. The three Sentinel "
           r"missions were acquired but none of their columns met the coverage gate."
           r"\label{tab:data}}",
           r"\centering\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{3.7cm}>{\raggedright\arraybackslash}p{1.8cm}>{\raggedright\arraybackslash}p{1.8cm}r@{}}",
           r"\toprule",
           r"Source & Variables, for example & Spatial resolution & Time step & Columns \\",
           r"\midrule"]
    for (nm, var, sp, ts, _), k in zip(DATA_SOURCES, counts):
        out.append(f"{nm} & {var} & {sp} & {ts} & {k} \\\\")
    out += [r"\midrule",
            r"IHME small-area estimates & Life expectancy at birth, both sexes, all races (the outcome) & County & Annual & -- \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(out) + "\n")
    print(f"  [saved] {path}")


if __name__ == "__main__":
    main()
