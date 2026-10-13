#!/usr/bin/env python3
"""
build_paper_constants.py -- every number the final manuscript quotes from the
analysis battery, written as LaTeX macros.

READS   the result files in the derived folder (see _config.DERIVED)
        main_run_constants/constants.tex, constants_v10.tex and
        constants_thresholds.tex, the main run's generated constants
        ../main.tex, to know which main-run constants the paper quotes
WRITES  ../constants_results.tex   every number from the analysis battery
        ../constants_base.tex      the main-run constants the paper quotes,
                                   resolved and flattened
        ../tables_generated.tex    the three appendix tables

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
    put("BlNaive", f(b["naive_mean"]["r2"], 3).replace("$-$0.000", "0.000"))
    put("BlYear", f(b["naive_year_mean"]["r2"]))
    put("BlNtl", f(b["ntl_only"]["r2"]))
    put("BlCoord", f(b["coords_only"]["r2"]))
    put("BlKnn", f(b["knn4_spatial"]["r2"]))
    put("BlModel", f(b["satellite_model"]["r2"]))
    put("BlModelMAE", f(b["satellite_model"]["mae"]))
    put("BlKnnMAE", f(b["knn4_spatial"]["mae"]))

    br = tab("bracket_error.csv")
    lo = br.iloc[0]
    put("BrLowLabel", str(lo["bracket"]).replace("-", " to "))
    put("BrLowMAE", f(lo["mae"], 2))
    put("BrLowN", n(lo["n"]))
    mid = br.loc[br["mae"].idxmin()]
    put("BrMidLabel", str(mid["bracket"]).replace("-", " to "))
    put("BrMidMAE", f(mid["mae"], 2))
    hi = br.iloc[-1]
    put("BrHighMAE", f(hi["mae"], 2))
    put("BrHighLabel", str(hi["bracket"]).replace("-", " to "))

    s = js("spatial_holdout.json")
    put("HoldStateRtwo", f(s["r2"]))
    put("HoldStateMAE", f(s["mae"], 2))
    put("HoldStateN", n(s["n_states"]))

    r = js("rank_stability.json")
    lc = r["learning_curve"]
    put("LcTenRtwo", f(lc["010pct"]["r2"]))
    put("LcTenCounties", n(lc["010pct"]["n_counties"]))
    put("LcFullRtwo", f(lc["100pct"]["r2"]))
    put("LcLastStep", f(lc["last_step_r2"]))
    put("LcFolds", n(lc["n_folds"]))
    tk = r["topk_overlap"]["topk_overlap"]
    put("RankDraws", n(r["topk_overlap"]["n_draws"]))
    put("RankTopFiveIdentical", pct(tk["top5"]["share_fully_identical"]))
    put("RankTopFifteenMin", n(round(15 * tk["top15"]["min_share"])))
    first = [x for x in r["topk_overlap"]["leading"] if x["point_rank"] == 1][0]
    put("RankFirstShare", pct(first["share_rank_1"]))
    sd = r["reseed"]["seeds"].values()
    put("ReseedTopFifteenLo", n(min(x["top15_shared"] for x in sd)))
    put("ReseedTopFifteenHi", n(max(x["top15_shared"] for x in sd)))
    put("ReseedN", n(len(r["reseed"]["seeds"])))

    ns = js("null_skill.json")
    put("NullSkillFull", f(ns["full"]["r2"]))
    put("NullSkillWithin", f(ns["within_county"]["r2"]))
    put("NullSkillBetween", f(ns["between_county"]["r2"]))

    t = js("temporal_validation.json")
    put("LoyoMean", f(t["leave_one_year_out"]["r2_mean"]))
    put("LoyoMin", f(t["leave_one_year_out"]["r2_min"]))
    put("FwdMean", f(t["forward_chaining"]["r2_mean"]))
    put("FwdFirst", f(t["forward_chaining"]["r2_first"]))
    put("FwdLast", f(t["forward_chaining"]["r2_last"]))
    put("FwdSteps", n(t["forward_chaining"]["n_steps"]))
    put("FwdSlope", f(t["forward_chaining"]["slope_r2_per_train_year"], 4))
    put("EraSplitYear", str(int(t["era_split"]["split_at"])))
    put("EraFisherP", p(t["era_split"]["fisher_p"]))

    inf = js("influence_drop.json")
    w5 = inf["arms"]["worst_residual_5pct"]
    put("InflDropN", n(w5["n_counties_dropped"]))
    put("InflDropRtwo", f(w5["r2"]))
    put("InflExtremeRtwo", f(inf["arms"]["extreme_outcome_5pct"]["r2"]))
    put("InflOverlap", n(inf["topk_overlap"]["shared"]))
    put("InflTopK", n(inf["topk_overlap"]["topk"]))

    g = js("gate_audit.json")
    put("GateUngatedRtwo", f(g["arms"]["ungated"]["r2"]))
    put("GateUngatedN", n(g["arms"]["ungated"]["n_features"]))
    put("GateGatedRtwo", f(g["arms"]["gated"]["r2"]))
    put("GateNoProtRtwo", f(g["arms"]["gated_no_protected"]["r2"]))
    put("GateNoProtN", n(g["arms"]["gated_no_protected"]["n_features"]))
    put("GateProtectedN", n(g["n_protected_modelled"]))

    mo = js("modality_arms.json")
    for key, mac in (("all_admitted", "ArmAll"), ("satellite_only", "ArmSat"),
                     ("non_satellite", "ArmNonSat")):
        put(mac + "Rtwo", f(mo["arms"][key]["r2"]))
        put(mac + "N", n(mo["arms"][key]["n_features"]))
    cs = mo["census_within_satellite"]
    put("ArmSatPairs", n(cs["n_pairs"]))
    put("ArmSatWins", n(cs["noncentral_wins"]))
    put("ArmSatP", p(sign_p(cs["noncentral_wins"], cs["n_pairs"])))
    put("ArmSatAgree", n(cs["agreement_with_full_model"]))

    # ------------------------------------------------------- the census
    sc = js("summary_statistic_census.json")
    ci = json.loads(sc["median_ratio_ci"]) if isinstance(sc["median_ratio_ci"], str) \
        else sc["median_ratio_ci"]
    put("CensusRatioCIlo", f(ci[0], 2))
    put("CensusRatioCIhi", f(ci[1], 2))
    put("CensusMedianSpread", f(sc["median_spread"], 1))

    nc = js("null_census.json")
    put("NullCensusDraws", n(nc["n_draws"]))
    put("NullCensusMean", f(nc["null_mean"], 1))
    put("NullCensusLo", n(nc["null_quantiles"]["q02"]))
    put("NullCensusHi", n(nc["null_quantiles"]["q97"]))
    pe = nc["empirical_p_one_sided"]
    put("NullCensusP", p(pe) if pe > 0 else
        f"$<{1 / nc['n_draws'] * 1e5:.0f}\\times10^{{-5}}$")
    put("NullInfoLowP", p(nc["information_census"]["empirical_p_one_sided_low"]))

    ex = js("census_exchangeable_null.json")
    put("ExchExpected", f(ex["expected"], 1))
    put("ExchLo", n(ex["range95_lo"]))
    put("ExchHi", n(ex["range95_hi"]))
    put("ExchP", p(ex["p_at_least_observed"]))
    put("ExchShare", pct(ex["mean_share_noncentral"]))

    it = js("information_theory_census.json")
    put("MIwins", n(it["n_noncentral_wins_mi"]))
    put("MIcentralWins", n(it["n_fields"] - it["n_noncentral_wins_mi"]))
    put("MIagree", n(it["n_agree"]))
    put("MIratio", f(it["median_mi_ratio"], 2))
    put("MIsignP", p(sign_p(it["n_noncentral_wins_mi"], it["n_fields"])))
    put("MIneighbours", n(it["mi_neighbours"]))

    pa = js("pair_anatomy.json")["pre_model"]
    put("PairCorr", f(pa["correlation"]))
    put("PairMIlow", f(pa["mi_low"]))
    put("PairMIhigh", f(pa["mi_high"]))
    put("PairMIband", f(pa["mi_band"]))
    put("PairROutLow", f(pa["r_outcome_low"], 2))
    put("PairROutHigh", f(pa["r_outcome_high"], 2))

    inv = js("attribution_invariance.json")["estimators"]
    for key, mac in (("interventional_published", "InvIntPub"),
                     ("interventional_recomputed", "InvIntRe"),
                     ("path_dependent", "InvPath"),
                     ("oof_path_dependent", "InvOOF"),
                     ("total_gain", "InvGain")):
        put(mac, n(inv[key]["noncentral_wins"]))
        put(mac + "P", p(sign_p(inv[key]["noncentral_wins"],
                                inv[key]["n_comparable"])))
    shares = [inv[k]["share"] for k in ("interventional_published",
              "interventional_recomputed", "path_dependent",
              "oof_path_dependent", "total_gain")]
    put("InvShareLo", pct(min(shares)))
    put("InvShareHi", pct(max(shares)))
    put("InvShapRows", n(inv["interventional_recomputed"]["n_rows_attributed"]))
    put("InvShapBackground", n(inv["interventional_recomputed"]["background"]))

    ps = js("permutation_across_splits.json")
    sp = pd.DataFrame(ps["splits"])
    for who, mac in (("production", "PermProd"), ("second", "PermSec")):
        ok = sp[sp[f"{who}_valid"]]
        put(mac + "Lo", n(ok[f"{who}_wins"].min()))
        put(mac + "Hi", n(ok[f"{who}_wins"].max()))
        put(mac + "Med", n(ok[f"{who}_wins"].median()))
        put(mac + "RtwoLo", f(ok[f"{who}_r2"].min()))
        put(mac + "RtwoHi", f(ok[f"{who}_r2"].max()))
        two = [sign_p(w, m) for w, m in zip(ok[f"{who}_wins"], ok[f"{who}_n"])]
        put(mac + "Sig", n(sum(x < 0.05 for x in two)))
        put(mac + "PLo", p(min(two)))
        put(mac + "PHi", p(max(two)))
        # How many splits fall below / above the range expected when every
        # summary of a field is equally likely to lead (script 29). The text
        # quotes these counts instead of saying "most".
        put(mac + "Below", n((ok[f"{who}_wins"] < ex["range95_lo"]).sum()))
        put(mac + "Above", n((ok[f"{who}_wins"] > ex["range95_hi"]).sum()))
        put(mac + "Valid", n(len(ok)))
    put("PermSplits", n(len(sp)))
    put("PermRows", n(ps["perm_rows"]))
    put("PermRepeats", n(ps["perm_repeats"]))

    jk = js("census_jackknife.json")
    put("JackLo", n(min(jk["range"])))
    put("JackHi", n(max(jk["range"])))
    put("JackBase", n(jk["baseline_wins"]))
    put("JackWorstP", p(sign_p(min(jk["range"]), jk["n_comparable"])))
    put("JackStates", n(jk["n_states"]))
    put("JackRows", n(jk["n_rows_attributed"]))

    ad = js("admission_rule_sensitivity.json")
    gs = ad["grid_span"]
    grid = tab("admission_rule_grid.csv")
    two = [sign_p(k, m) for k, m in zip(grid["noncentral_wins"], grid["n_counted"])]
    put("AdmCells", n(len(grid)))
    put("AdmSig", n(sum(x < 0.05 for x in two)))
    put("AdmWorstP", p(max(two)))
    worst = grid.iloc[int(np.argmax(two))]
    put("AdmWorstWins", n(worst["noncentral_wins"]))
    put("AdmWorstN", n(worst["n_counted"]))
    put("AdmShareLo", pct(gs["count_share_min"]))
    put("AdmShareHi", pct(gs["count_share_max"]))
    put("AdmModLo", n(gs["modelled_min"]))
    put("AdmModHi", n(gs["modelled_max"]))
    band = ad["coverage_insensitivity_band"]
    put("AdmCovLo", f(band["identical_from"], 3))
    put("AdmCovHi", f(band["identical_up_to"], 3))
    pr = ad["protected_rule"]
    put("AdmPairsWith", n(pr["census_pairs_with_rule"]))
    put("AdmPairsWithout", n(pr["census_pairs_without_rule"]))
    put("AdmModWithout", n(pr["modelled_without_rule"]))
    oc = ad["column_order_sensitivity"]
    put("AdmOrderShuffles", n(oc["n_shuffles"]))
    put("AdmImmuneN", n(oc["immune_subset"]["n"]))
    put("AdmImmuneWins", n(oc["immune_subset"]["noncentral_wins"]))
    put("AdmImmuneP", p(sign_p(oc["immune_subset"]["noncentral_wins"],
                               oc["immune_subset"]["n"])))

    mc = js("multiplicity_control.json")
    put("FdrRejected", n(mc["control"]["q_0.05"]["n_rejected"]))
    put("FdrTests", n(mc["control"]["q_0.05"]["n_tests"]))

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
    put("LtJackN", n(jj["n_replicates"]))
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
                          ("prism_tmin_min", "TpTmin", 2)):
        rr = tp.loc[feat]
        put(mac, f(rr["point_estimate"], nd))
        put(mac + "Lo", f(rr["ci_lo"], nd))
        put(mac + "Hi", f(rr["ci_hi"], nd))
        put(mac + "Width", f(100 * rr["width_frac"], 1))
        put(mac + "Above", f(rr["pct_above"], 1))
    trip = tp.loc[["prism_tdmean_min", "eng2_wetbulb_night",
                   "eng_night_cooling_eff"], "width_frac"]
    put("TpTripletWidthMax", f(100 * trip.max(), 1))

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
    put("ResCounties", n(rs["n_counties"]))
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
        put("ResChrN", n(by_src.get("County Health Rankings", 0)))
        put("ResSaipeN", n(by_src.get("Small Area Income and Poverty Estimates", 0)))
        put("ResEconN", n(by_src.get("Regional economic accounts", 0)))
        put("ResSummaryN", n(by_src.get("Summary indicators of the database", 0)))
        arms = sv["arms"]
        put("ResNoChrN", n(arms["without_county_health_rankings"]["n_columns"]))
        put("ResNoChrRtwo", f(arms["without_county_health_rankings"]["oof_r2"], 2))
        put("ResAcsRtwo", f(arms["american_community_survey_only"]["oof_r2"], 2))

    mi = js("residual_spatial_structure.json")
    put("ResMoranI", f(mi["morans_i"], 3))
    put("ResMoranP", p(mi["p_value"]))

    # The reference of every SHAP value: the model's average prediction. A
    # turning point is where a variable's contribution crosses this level.
    LINES.append(f"\n% ---- {C.OOF_A.name}")
    oof = pd.read_csv(C.OOF_A)
    put("OutcomeMean", f(oof["meanlifeexpectency"].mean(), 1))
    put("PredMean", f(oof["pred"].mean(), 1))

    # How correlated the two summaries the census compares are (script 31).
    pc_path = C.DERIVED / "census_pair_correlation.json"
    if pc_path.exists():
        LINES.append(f"\n% ---- {pc_path.name}")
        pc = json.loads(pc_path.read_text())
        rs_ = [abs(x["r"]) for x in pc["pairs"] if x["r"] is not None]
        put("CensusPairRMed", f(pc["median_abs_r"], 2))
        put("CensusPairRHighN", n(sum(1 for v in rs_ if v >= 0.8)))

    # The reference configuration every refit uses (scripts 04, 12, 22).
    LINES.append("\n% ---- nce_refit.json (refit protocol)")
    proto = json.load(open(C.DERIVED / "nce_refit.json"))["protocol"]
    put("RefitTrees", n(proto["n_estimators"]))
    put("RefitLR", f(proto["learning_rate"], 2))
    put("RefitDepth", n(proto["max_depth"]))
    put("RefitShapN", n(proto["nshap"]))
    put("RefitShapBg", n(proto["nbg"]))

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
        put("VerXgbRef", str(vr.get("xgboost", "")))
        put("VerShapRef", str(vr.get("shap", "")))
        put("VerHoldRef", f(lv["spatial_holdout_r2"]["reference"]))
        put("VerMoranRef", f(lv["residual_morans_i"]["reference"], 3))
        put("VerPermShift", n(lv["permutation_count_shift"]["production_max"]))
        put("VerPermSame", n(lv["permutation_count_shift"]["production_splits_identical"]))

    write_tables(Path(a.out).parent / "tables_generated.tex", tp, inv, ps, ex)
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
    ("Abstract, 3.5, 5: lifetime correlation is inverse",
     lambda v: v["LtRho"] < 0),
    ("Abstract, 3.5, 5: lifetime correlation is 'not significant'",
     lambda v: v["LtPermP"] >= 0.05),
    ("Abstract, 3.3: census count lies within the exchangeable range",
     lambda v: v["ExchLo"] <= v["StatNonCentralWins"] <= v["ExchHi"] and v["ExchP"] >= 0.05),
    ("Abstract, 3.3: mutual information favours the mean or median",
     lambda v: v["MIcentralWins"] > v["StatFieldsComparable"] / 2),
    ("3.4, 4.x: permutation counts never exceed the exchangeable range",
     lambda v: v["PermProdAbove"] == 0 and v["PermSecAbove"] == 0),
    ("3.4, 4.x: permutation count moves between splits",
     lambda v: v["PermProdHi"] - v["PermProdLo"] >= 3),
    ("3.2: holding out whole states lowers skill",
     lambda v: v["HoldStateRtwo"] < v["Rtwo"]),
    ("3.6: only 10% perturbation moves a value by more than one width",
     lambda v: v["PertTenMax"] > 1 and v["PertFiveMax"] <= 1),
    ("3.6: no null curve passes the stability rule",
     lambda v: v["NullCrossPasses"] == 0),
    ("3.6, abstract: turning points 'barely move' across regions (<= 5% of range)",
     lambda v: v["RegMerTripletMax"] <= 5 and v["RegMedTripletMax"] <= 5),
    ("2.x, 3.6: crossings lie near the middle of each distribution (25 to 75%)",
     lambda v: all(25 <= v[k] <= 75 for k in ("TpDewAbove", "TpWetAbove", "TpNceAbove"))),
    ("3.7: residuals 'weakly but significantly' clustered",
     lambda v: v["ResMoranP"] < 0.05 and abs(v["ResMoranI"]) < 0.1),
    ("Abstract, 3.7, 5: over half of the residual variance unexplained",
     lambda v: v["ResSdohRtwo"] < 0.5),
    ("2.x: refit configuration reproduces production skill (within 0.01)",
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


UNIT_TEX = {"degC": "$^{\\circ}$C", "%": "\\% of year", "ug/m3": "10\\,$\\mu$g\\,m$^{-3}$"}
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


def write_tables(path, tp, inv, ps, ex):
    """Two appendix tables, generated so no value in them is typed in."""
    out = ["% tables_generated.tex -- written by analysis/build_paper_constants.py",
           "% Do not edit by hand.", ""]

    # Table A1: every located turning point and the three criteria
    found_all = bool((tp["found_frac"] >= 1).all())
    out += [r"\begin{table}[H]", r"\caption{Every candidate turning point. The "
            r"interval is the 2.5th to 97.5th percentile of the located value over "
            r"the bootstrap resamples, and its width is given as a share of the "
            r"variable's observed range. A candidate is stable when a crossing is "
            r"found in at least 60\% of resamples, the width is below 40\% and the "
            r"full-sample value lies inside the interval."
            + (" A crossing was found in every resample for every candidate."
               if found_all else "") + r"\label{tab:turning}}",
            r"\centering\footnotesize",
            r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
            r"Variable & Unit & Value & 95\% interval & Width (\%) & Stable \\",
            r"\midrule"]
    for feat, r in tp.sort_values("width_frac").iterrows():
        unit = UNIT_TEX.get(str(r["unit"]), "dimensionless")
        nd = 3 if abs(r["point_estimate"]) < 2 else 2
        out.append(f"{SHORT_TEX.get(feat, feat)} & {unit} & "
                   f"{f(r['point_estimate'], nd)} & "
                   f"{f(r['ci_lo'], nd)} to {f(r['ci_hi'], nd)} & "
                   f"{f(100 * r['width_frac'], 1)} & "
                   f"{'yes' if bool(r['stable']) else 'no'} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table A2: the census under every instrument
    rows = [("Interventional SHAP, main run", "decomposes the prediction",
             inv["interventional_published"]),
            ("Interventional SHAP, recomputed", "decomposes the prediction",
             inv["interventional_recomputed"]),
            ("Path-dependent TreeSHAP", "decomposes the prediction",
             inv["path_dependent"]),
            ("Path-dependent TreeSHAP, out of fold", "decomposes the prediction",
             inv["oof_path_dependent"]),
            ("Total split gain", "counts tree use", inv["total_gain"]),
            ("Mutual information", "marginal, no model", inv["mutual_information"])]
    out += [r"\begin{table}[H]", r"\caption{The census count under each "
            r"instrument: the number of the 47 comparable fields in which the "
            r"non-central summary is credited with more than the central one. "
            r"If every summary of a field were equally likely to lead, the "
            f"expected count would be {ex['expected']:.1f}, with a 95\\% range "
            f"of {ex['range95_lo']} to {ex['range95_hi']}. Mutual information "
            r"compares the two summaries the main attribution selected. "
            r"Permutation importance is given as its range over five "
            r"county-disjoint held-out splits.\label{tab:instruments}}",
            r"\centering\small",
            r"\begin{tabular}{@{}llr@{}}", r"\toprule",
            r"Instrument & What it measures & Non-central leads \\",
            r"\midrule"]
    for lab, kind, e in rows:
        out.append(f"{lab} & {kind} & {int(e['noncentral_wins'])} of "
                   f"{int(e['n_comparable'])} \\\\")
    sp = pd.DataFrame(ps["splits"])
    for who, lab in (("production", "Permutation, production learner"),
                     ("second", "Permutation, second learner")):
        ok = sp[sp[f"{who}_valid"]]
        out.append(f"{lab} & loss when one column is broken & "
                   f"{int(ok[f'{who}_wins'].min())} to "
                   f"{int(ok[f'{who}_wins'].max())} of "
                   f"{int(ok[f'{who}_n'].iloc[0])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    # Table A4: the species of the lifetime test (script 25)
    lt = json.load(open(C.DERIVED / "lifetime_literature_test.json"))
    sp = sorted(lt["species_table"], key=lambda r: r["lifetime_days"])
    names = {"pm2 5": "Fine particulate matter (reanalysis)",
             "nitrogen monoxide": "Nitric oxide"}
    out += [r"\begin{table}[H]", r"\caption{The species of the lifetime test, "
            r"ordered by the tropospheric lifetime taken from the literature, "
            r"with the number of duration and concentration pairs admitted for "
            r"each, the median ratio of duration to concentration importance and "
            r"whether duration was credited more. All series are from the "
            r"atmospheric composition reanalysis.\label{tab:species}}",
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

    # Table A3: the learner and its settings, from the tournament output
    ls = json.load(open(C.DERIVED / "learner_settings.json"))
    xp = ls["params"]["XGB"]
    names = {"n_estimators": "Number of trees", "max_depth": "Maximum depth",
             "learning_rate": "Learning rate", "subsample": "Row subsample",
             "colsample_bytree": "Column subsample per tree",
             "min_child_weight": "Minimum child weight",
             "reg_alpha": "L1 regularisation", "reg_lambda": "L2 regularisation"}
    out += [r"\begin{table}[H]", r"\caption{The fitted learner. Settings of the "
            r"XGBoost model and the out-of-fold skill of each learner compared on "
            r"identical county-grouped folds.\label{tab:learner}}",
            r"\centering\small", r"\begin{tabular}{@{}lr@{}}", r"\toprule",
            r"Setting & Value \\", r"\midrule"]
    for k, lab in names.items():
        v = xp[k]
        out.append(f"{lab} & {v if isinstance(v, int) else f'{v:.3g}'} \\\\")
    out.append(r"\midrule")
    out.append(r"Learner & Out-of-fold $R^{2}$ (mean absolute error, years) \\")
    out.append(r"\midrule")
    for k, v in ls["scores"].items():
        if "r2" in v:
            out.append(f"{k} & {v['r2']:.3f} ({v['mae']:.3f}) \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(out) + "\n")
    print(f"  [saved] {path}")


if __name__ == "__main__":
    main()
