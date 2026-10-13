#!/usr/bin/env python3
"""
paper_figures.py -- every numbered figure of the manuscript, in one place.

Figure 1 (the methodology diagram) and the graphical abstract are drawn by
figure01_methodology_and_graphical_abstract.py and can be replaced by
hand-made versions saved under the same file names. Figures 2 to 10 are drawn here, each by one function, each reading
only result files written by the numbered analysis scripts. No value in any
figure is typed in.

House style, applied to every figure by one function so no figure can drift:
  * one sans-serif face (DejaVu Sans), regular weight throughout; the only
    bold is the panel letter;
  * 8.5 pt labels and 7.5 pt ticks at the authored width of 5.5 in, which is
    the text width of the MDPI layout, so the printed sizes are these sizes;
  * panel letters A, B, C in 9.5 pt bold, top left of each panel, each
    followed by a short panel title in regular weight;
  * colour by meaning, the same meaning in every figure:
        central summary (mean, median)          blue    #0072B2
        non-central summary (quantile, width)   orange  #E69F00
        stable / passes                          green   #009E73
        fails / not located                      red     #D55E00
        reference, null, background              grey
    taken from the Okabe-Ito palette, which separates under the common forms
    of colour-vision deficiency;
  * every axis carries its unit in parentheses, or says it has none;
  * each figure carries a short title in regular weight above its panels
    (the TITLES table below); the caption carries the explanation.

Output: ../Figures/Figure02_performance.png and .pdf, and so on, PNG at
600 dpi and PDF as vector.

    python3 paper_figures.py                 # all of them
    python3 paper_figures.py --only 3,6      # just Figures 3 and 6
    python3 paper_figures.py --dpi 150 --outdir ../_preview    # a quick look
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402
from matplotlib.patches import Patch                           # noqa: E402

# ===========================================================================
#  STYLE
# ===========================================================================
W = 5.5                       # inches, the MDPI text width
BLUE, ORANGE = "#0072B2", "#E69F00"
GREEN, RED = "#009E73", "#D55E00"
PINK, SKY = "#CC79A7", "#56B4E9"
GREY, LIGHT, INK = "#8C8C8C", "#D9D9D9", "#222222"
FS, FS_TICK, FS_LET = 8.5, 7.5, 9.5

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS, "font.weight": "normal",
    "axes.titlesize": FS, "axes.titleweight": "normal",
    "axes.labelsize": FS, "axes.labelweight": "normal",
    "axes.labelcolor": INK, "axes.edgecolor": "#555555",
    "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": FS_TICK, "legend.frameon": False,
    "lines.linewidth": 1.2, "patch.linewidth": 0.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK, "mathtext.default": "regular",
})

DPI = 600
OUT = C.FIGURES


def letter(ax, s):
    """Register a panel letter. It is placed at save time, at the outer left
    edge of the panel including its tick labels, so a letter can never sit on
    a label or fall off the canvas whatever the panel's margins are."""
    fig = ax.figure
    if not hasattr(fig, "_letters"):
        fig._letters = []
    fig._letters.append((ax, s))


# Figure titles and panel titles, one place for all of them. The figure title
# is set above the panels in regular weight; each panel carries its letter in
# bold followed by a short title in regular weight.
TITLES = {
    "Figure02_performance": (
        "Predictive performance of the environment-only model",
        {"A": "Skill against baselines", "B": "Learning curve",
         "C": "Error by outcome bracket", "D": "Skill under outcome permutation"}),
    "Figure03_two_descriptions": (
        "One field, two descriptions",
        {"A": "Formaldehyde (CAMS)", "B": "Night cooling efficiency (MODIS)",
         "C": "Position in the full ranking", "D": "Agreement of the two NCE forms"}),
    "Figure04_drop_one_refit": (
        "Drop-one refit of night cooling efficiency",
        {"A": "Mean-form attribution", "B": "Mean-form rank",
         "C": "Predictive skill"}),
    "Figure05_rashomon_census": (
        "Attribution, information and skill across summaries",
        {"A": "Attribution (SHAP)", "B": "Information (MI)",
         "C": "Matched refit", "D": "Disjoint modality sets"}),
    "Figure06_invariance": (
        "Stability of the census",
        {"A": "Eight attribution instruments", "B": "State jackknife",
         "C": "Admission-rule grid"}),
    "Figure07_lifetime": (
        "Atmospheric lifetime and the duration advantage",
        {"A": "Duration advantage by species", "B": "Ratio against lifetime",
         "C": "Class-boundary sweep"}),
    "Figure08_turning_points": (
        "Stability of the located turning points",
        {"A": "Bootstrap interval width", "B": "Permuted attribution",
         "C": "Predictor noise"}),
    "Figure09_climate_invariance": (
        "Nocturnal turning points across climate regions",
        {"A": "Minimum dew point", "B": "Nocturnal wet-bulb",
         "C": "Night cooling, tail form", "D": "West minus east shift"}),
    "Figure10_failure_geography": (
        "Geography of model error",
        {"A": "County residual", "B": "Predicting the residual",
         "C": "Socioeconomic correlates"}),
}
FS_TITLE = FS + 1.0


def _place_letters(fig, panel_titles=None):
    """Bold panel letter at the outer left edge of each panel (tick labels
    included), followed on the same line by the panel title in regular
    weight, both above anything the panel already carries on top."""
    panel_titles = panel_titles or {}
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, s in getattr(fig, "_letters", []):
        bb = ax.get_tightbbox(r).transformed(inv)
        pos = ax.get_position()
        y = max(pos.y1, bb.y1) + 0.006
        t = fig.text(bb.x0, y, s, fontsize=FS_LET, fontweight="bold",
                     ha="left", va="bottom", color=INK)
        t._is_letter = True
        title = panel_titles.get(s)
        if title:
            w = t.get_window_extent(r).transformed(inv).width
            fig.text(bb.x0 + w + 0.012, y, title, fontsize=FS, ha="left",
                     va="bottom", color=INK)


def _place_title(fig, text):
    """The figure title, centred above everything else on the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = fig.get_tightbbox(r).y1 / fig.get_figheight()   # inches to figure
    fig.text(0.5, top + 0.02, text, fontsize=FS_TITLE, ha="center",
             va="bottom", color=INK)


def save(fig, stem):
    fig_title, panel_titles = TITLES.get(stem, ("", {}))
    # every text regular weight first, then the panel letters alone in bold
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontweight("normal")
    _place_letters(fig, panel_titles)
    if fig_title:
        _place_title(fig, fig_title)
    OUT.mkdir(parents=True, exist_ok=True)
    # Matplotlib clips an axis label to the canvas when it computes the tight
    # box, so a long label can lose its last characters. Listing every text
    # explicitly makes the saved area grow to hold it instead.
    extra = fig.get_default_bbox_extra_artists() + [
        t for ax in fig.axes for t in (ax.xaxis.label, ax.yaxis.label)
        if t.get_visible() and t.get_text()] + list(fig.texts)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=DPI, facecolor="white",
                    bbox_inches="tight", pad_inches=0.04,
                    bbox_extra_artists=extra)
    plt.close(fig)
    print(f"  [saved] {OUT.name}/{stem}.png and .pdf at {DPI} dpi")


def js(name):
    return json.load(open(C.DERIVED / name))


def tab(name):
    p = C.DERIVED / name
    return pd.read_csv(p) if p.exists() else C.load_table(name.replace(".csv", ""))


def sign_p(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


# Readable names, with units, for the variables the figures label.
NAMES = {
    "prism_tdmean_min": ("Minimum dew point", "°C"),
    "eng2_night_cooling_efficiency": ("Night cooling efficiency, mean form", "dimensionless"),
    "eng_night_cooling_eff": ("Night cooling efficiency, tail form", "dimensionless"),
    "lst_night_1km_mean": ("Mean night land-surface temperature", "°C"),
    "eng2_wetbulb_night": ("Nocturnal wet-bulb temperature", "°C"),
    "eng2_wetbulb_hot": ("Warm-hour wet-bulb temperature", "°C"),
    "eng2_hcho_no2_ratio": ("Formaldehyde to NO$_2$ ratio", "dimensionless"),
    "pm25_mean": ("Annual mean PM$_{2.5}$", "product units"),
    "prism_tmin_min": ("Minimum daily minimum temperature", "°C"),
    "cams_fot_formaldehyde_above75th_percentile": ("Formaldehyde, time above P75", "% of year"),
    "usda_cropland_usda_corn_pct": ("Cropland in corn", "fraction of area"),
}
SHORT = {
    "prism_tdmean_min": "Minimum dew point",
    "eng2_night_cooling_efficiency": "Night cooling, mean form",
    "eng_night_cooling_eff": "Night cooling, tail form",
    "lst_night_1km_mean": "Mean night surface temperature",
    "eng2_wetbulb_night": "Nocturnal wet-bulb",
    "eng2_wetbulb_hot": "Warm-hour wet-bulb",
    "eng2_hcho_no2_ratio": "Formaldehyde to NO$_2$ ratio",
    "pm25_mean": "Mean fine particulate matter",
    "prism_tmin_min": "Minimum daily low temperature",
    "cams_fot_formaldehyde_above75th_percentile": "Formaldehyde, time above 75th pct.",
    "usda_cropland_usda_corn_pct": "Cropland in corn",
}
NOCTURNAL = ("prism_tdmean_min", "eng2_wetbulb_night", "eng_night_cooling_eff")


# ===========================================================================
#  FIGURE 2  The model is worth interrogating
# ===========================================================================
def fig02():
    base = js("baselines.json")
    hold = js("spatial_holdout.json")
    rank = js("rank_stability.json")
    null = js("null_skill.json")
    br = tab("bracket_error.csv")

    fig = plt.figure(figsize=(W, 4.6))
    gs = fig.add_gridspec(2, 2, left=0.30, right=0.98, top=0.93, bottom=0.10,
                          hspace=0.55, wspace=1.15)

    # A  baselines on identical folds, and the state holdout
    ax = fig.add_subplot(gs[0, 0])
    rows = [("Training mean", base["naive_mean"]["r2"], GREY),
            ("Training mean by year", base["naive_year_mean"]["r2"], GREY),
            ("Night-time lights only", base["ntl_only"]["r2"], GREY),
            ("Latitude, longitude, year", base["coords_only"]["r2"], GREY),
            ("Four nearest counties", base["knn4_spatial"]["r2"], GREY),
            ("This model, states held out", hold["r2"], SKY),
            ("This model", base["satellite_model"]["r2"], BLUE)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c) in zip(y, rows):
        ax.barh(yi, max(v, 0), color=c, height=0.62)
        ax.text(max(v, 0) + 0.015, yi, f"{v:.3f}".replace("-0.000", "0.000"),
                va="center", fontsize=FS_TICK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Out-of-fold $R^2$ (dimensionless)")
    letter(ax, "A")

    # B  learning curve
    ax = fig.add_subplot(gs[0, 1])
    lc = rank["learning_curve"]
    pts = sorted([v for v in lc.values() if isinstance(v, dict)
                  and "county_share" in v], key=lambda v: v["county_share"])
    xs = [100 * p["county_share"] for p in pts]
    ax.plot(xs, [p["r2"] for p in pts], "-o", color=BLUE, ms=4)
    ax.set_xlabel("Training counties (%)")
    ax.set_ylabel("Out-of-fold $R^2$\n(dimensionless)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0.4, 0.7)
    letter(ax, "B")

    # C  error by outcome bracket
    ax = fig.add_subplot(gs[1, 0])
    lab = br["bracket"].astype(str).tolist()
    yy = np.arange(len(lab))[::-1]
    ax.barh(yy, br["mae"], color=BLUE, height=0.62)
    for yi, v, n in zip(yy, br["mae"], br["n"]):
        ax.text(v + 0.1, yi, f"{v:.2f}  (n = {int(n):,})", va="center",
                fontsize=FS_TICK)
    ax.set_yticks(yy, [f"{s} years" for s in lab])
    ax.set_xlim(0, 11.5)
    ax.set_xlabel("Mean absolute error (years)")
    ax.set_ylabel("Observed life expectancy")
    letter(ax, "C")

    # D  where the explained variance lives
    ax = fig.add_subplot(gs[1, 1])
    items = [("Observed", base["satellite_model"]["r2"], BLUE),
             ("Years shuffled\nwithin county", null["within_county"]["r2"], SKY),
             ("Series swapped\nbetween counties", null["between_county"]["r2"], GREY),
             ("Fully shuffled", null["full"]["r2"], GREY)]
    yy = np.arange(len(items))[::-1]
    for yi, (lab_, v, c) in zip(yy, items):
        ax.barh(yi, v, color=c, height=0.62)
        ax.text(max(v, 0) + 0.02, yi, f"{v:+.3f}", va="center",
                fontsize=FS_TICK)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks(yy, [i[0] for i in items])
    ax.set_xlim(-0.1, 0.95)
    ax.set_xlabel("Out-of-fold $R^2$\n(dimensionless)")
    letter(ax, "D")
    save(fig, "Figure02_performance")


# ===========================================================================
#  FIGURE 3  One field, two descriptions
# ===========================================================================
def fig03():
    import pyarrow.parquet as pq
    rk = pd.read_csv(C.RANKING_A).sort_values("rank")
    val = dict(zip(rk["feature"], rk["mean_abs_shap"]))
    rnk = dict(zip(rk["feature"], rk["rank"]))

    fig = plt.figure(figsize=(W, 4.9))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.98, top=0.94, bottom=0.10,
                          hspace=0.60, wspace=0.75)

    def pair(ax, items, title):
        y = np.arange(len(items))[::-1]
        for yi, (lab, f, c) in zip(y, items):
            ax.barh(yi, val[f], color=c, height=0.6)
            ax.text(val[f] * 1.15, yi, f"rank {int(rnk[f])}", va="center",
                    fontsize=FS_TICK)
        ax.set_yticks(y, [i[0] for i in items])
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 3.0)
        ax.set_xlabel("Mean |SHAP| (years)")

    ax = fig.add_subplot(gs[0, 0])
    pair(ax, [("Time above P75", "cams_fot_formaldehyde_above75th_percentile", ORANGE),
              ("Time above P90", "cams_fot_formaldehyde_above_90th_percentile", ORANGE),
              ("Mean column", "cams_total_column_formaldehyde_x1e9", BLUE)],
         "CAMS formaldehyde")
    letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    pair(ax, [("Tail form", C.NCE_TAIL, ORANGE), ("Mean form", C.NCE_MEAN, BLUE)],
         "MODIS night cooling efficiency")
    letter(ax, "B")

    # C  where the four sit in the ranking of all admitted variables
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(rk["rank"], rk["mean_abs_shap"], color=GREY, lw=1.0)
    for f, c, xt, yt, lab in (
            ("cams_fot_formaldehyde_above75th_percentile", ORANGE, 90, 0.30, "HCHO, time above P75"),
            (C.NCE_TAIL, ORANGE, 90, 0.11, "NCE, tail form"),
            (C.NCE_MEAN, BLUE, 230, 0.040, "NCE, mean form"),
            ("cams_total_column_formaldehyde_x1e9", BLUE, 290, 0.012, "HCHO, mean column")):
        ax.plot(rnk[f], val[f], "o", color=c, ms=5, mec="white", mew=0.6,
                zorder=3)
        ax.annotate(lab, (rnk[f], val[f]), xytext=(xt, yt), fontsize=FS_TICK,
                    va="center", arrowprops=dict(arrowstyle="-",
                    color="#777777", lw=0.5))
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(0, len(rk) + 5)
    ax.set_xlabel(f"Rank among {len(rk)} admitted variables")
    ax.set_ylabel("Mean |SHAP| (years)")
    letter(ax, "C")

    # D  the two constructions against each other
    ax = fig.add_subplot(gs[1, 1])
    d = pq.read_table(C.FRAME, columns=[C.NCE_TAIL, C.NCE_MEAN]).to_pandas().dropna()
    r = np.corrcoef(d[C.NCE_MEAN], d[C.NCE_TAIL])[0, 1]
    lo = np.percentile(np.r_[d[C.NCE_MEAN], d[C.NCE_TAIL]], 0.2)
    hi = np.percentile(np.r_[d[C.NCE_MEAN], d[C.NCE_TAIL]], 99.8)
    hb = ax.hexbin(d[C.NCE_MEAN], d[C.NCE_TAIL], gridsize=50, mincnt=1,
                   bins="log", cmap="Greys", linewidths=0,
                   extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(0.05, 0.94, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=FS_TICK, va="top")
    ax.set_xlabel("NCE, mean form (dimensionless)")
    ax.set_ylabel("NCE, tail form\n(dimensionless)")
    cb = fig.colorbar(hb, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label("County-years per cell", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.5)
    letter(ax, "D")
    save(fig, "Figure03_two_descriptions")


# ===========================================================================
#  FIGURE 4  Removing one description does not hand its credit to the other
# ===========================================================================
def fig04():
    r = js("nce_refit.json")["conditions"]
    full, dp, dm = r["FULL"], r["DROP_P"], r["DROP_M"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.4),
                            gridspec_kw=dict(width_ratios=[1, 0.85, 1.35]))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.24,
                        wspace=0.85)
    ticks = ["Both", "No tail"]

    # A  attribution of the mean form, with and without its twin
    ax = axs[0]
    for i, v in enumerate([full["mean_shap"], dp["mean_shap"]]):
        ax.bar(i, v, color=BLUE, width=0.55)
        ax.text(i, v * 1.18, f"{v:.3f}", ha="center", fontsize=FS_TICK)
    ax.axhline(full["pct_shap"], color=ORANGE, ls="--", lw=0.9)
    ax.text(-0.42, full["pct_shap"] * 1.15, "tail form, both present",
            fontsize=6.6, va="bottom", ha="left", color=INK)
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 0.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1], ticks, fontsize=6.8)
    ax.set_ylabel("Mean form,\nmean |SHAP| (years)")
    letter(ax, "A")

    # B  its rank among all admitted columns
    ax = axs[1]
    rk = [full["mean_rank"], dp["mean_rank"]]
    ax.plot([0, 1], rk, "-", color=GREY, lw=0.9, zorder=1)
    ax.plot([0, 1], rk, "o", color=BLUE, ms=6, zorder=2)
    for i, v in enumerate(rk):
        ax.text(i + 0.12, v, f"{int(v)}", va="center", ha="left",
                fontsize=FS_TICK)
    ax.set_xticks([0, 1], ticks, fontsize=6.8)
    ax.set_xlim(-0.6, 1.7)
    ax.set_ylim(150, 1)
    ax.set_yticks([1, 50, 100, 150])
    ax.set_ylabel("Mean form, rank\n(1 = most important)")
    letter(ax, "B")

    # C  predictive skill in the three conditions
    ax = axs[2]
    items = [("Both", full["r2"]), ("No tail", dp["r2"]),
             ("No mean", dm["r2"])]
    for i, (lab, v) in enumerate(items):
        ax.plot(i, v, "o", color=BLUE, ms=5)
        ax.text(i, v + 0.0010, f"{v:.4f}", ha="center", fontsize=6.6)
    ax.set_xticks(range(3), [i[0] for i in items], fontsize=6.8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.662, 0.674)
    ax.set_ylabel("Out-of-fold $R^2$\n(dimensionless)")
    letter(ax, "C")
    save(fig, "Figure04_drop_one_refit")


# ===========================================================================
#  FIGURE 5  Attribution follows the description; information does not
# ===========================================================================
def fig05():
    cen = tab("summary_statistic_census.csv")
    cen = cen[cen["comparable"].astype(str).str.lower().isin(("true", "1"))]
    it = tab("information_theory_census.csv")
    refit = js("summary_statistic_refit.json")
    mod = js("modality_arms.json")

    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(2, 2, left=0.12, right=0.98, top=0.90, bottom=0.10,
                          hspace=0.55, wspace=0.42)

    def diag(ax, x, y, wins, xl, yl, lo, hi):
        ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=0.8)
        ax.scatter(x[~wins], y[~wins], s=14, color=BLUE, edgecolor="white",
                   linewidth=0.4, label="central leads", zorder=3)
        ax.scatter(x[wins], y[wins], s=14, color=ORANGE, edgecolor="white",
                   linewidth=0.4, label="non-central leads", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    ax = fig.add_subplot(gs[0, 0])
    w = cen["noncentral_wins"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    diag(ax, cen["central_shap"].to_numpy(), cen["noncentral_shap"].to_numpy(),
         w, "Central summary, mean |SHAP| (years)",
         "Non-central summary,\nmean |SHAP| (years)", 3e-4, 0.5)
    ax.text(0.04, 0.95, f"{int(w.sum())} of {len(w)} fields\nabove the diagonal",
            transform=ax.transAxes, va="top", fontsize=FS_TICK)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE,
                               label="central summary leads under SHAP"),
                        Line2D([], [], marker="o", ls="", color=ORANGE,
                               label="non-central summary leads under SHAP"),
                        Line2D([], [], ls="--", color=GREY, label="equal")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.0),
               fontsize=6.8, handletextpad=0.3, columnspacing=1.2)
    letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    wm = it["noncentral_wins_mi"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    ws = it["noncentral_wins_shap"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    diag(ax, it["central_mi"].to_numpy(), it["noncentral_mi"].to_numpy(), ws,
         "Central summary, MI (nats)", "Non-central summary,\nMI (nats)",
         1e-2, 1.2)
    ax.text(0.04, 0.95, f"{int(wm.sum())} of {len(wm)} fields\nabove the diagonal",
            transform=ax.transAxes, va="top", fontsize=FS_TICK)
    letter(ax, "B")

    # C  the matched pair of panels: equal skill, opposite explanations
    ax = fig.add_subplot(gs[1, 0])
    arms = {a["label"]: a for a in refit["arms"]}
    c_r2 = arms["central only"]["r2"]
    n_r2 = arms["non-central only"]["r2"]
    draws = [d["r2"] for d in refit["random_draws"]]
    ax.bar(0, c_r2, color=BLUE, width=0.55)
    ax.bar(1, n_r2, color=ORANGE, width=0.55)
    ax.scatter(np.full(len(draws), 2), draws, s=14, color=ORANGE,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhspan(min(draws), max(draws), color=ORANGE, alpha=0.12, lw=0)
    for x, v in ((0, c_r2), (1, n_r2)):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=FS_TICK)
    ax.set_xticks([0, 1, 2], ["Central\nonly", "Non-central\nonly",
                              "Non-central,\nrandom draws"], fontsize=6.8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.55, 0.64)
    ax.set_ylabel("Out-of-fold $R^2$,\n47 columns each")
    letter(ax, "C")

    # D  disjoint column sets, near-equal skill
    ax = fig.add_subplot(gs[1, 1])
    a = mod["arms"]
    items = [("All admitted", a["all_admitted"], GREY),
             ("Satellite only", a["satellite_only"], SKY),
             ("Non-satellite only", a["non_satellite"], PINK)]
    for i, (lab, v, c) in enumerate(items):
        ax.bar(i, v["r2"], color=c, width=0.55)
        ax.text(i, v["r2"] + 0.006, f"{v['r2']:.3f}", ha="center",
                fontsize=FS_TICK)
    ax.set_xticks(range(3), [f"All\nadmitted\n{items[0][1]['n_features']} cols",
                             f"Satellite\nonly\n{items[1][1]['n_features']} cols",
                             f"Non-\nsatellite\n{items[2][1]['n_features']} cols"],
                  fontsize=6.8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.55, 0.70)
    ax.set_ylabel("Out-of-fold $R^2$\n(dimensionless)")
    letter(ax, "D")
    save(fig, "Figure05_rashomon_census")


# ===========================================================================
#  FIGURE 6  The count is a property of attribution, not of the analysis
# ===========================================================================
def fig06():
    inv = js("attribution_invariance.json")["estimators"]
    ps = js("permutation_across_splits.json")
    ex = js("census_exchangeable_null.json")
    jk = tab("census_jackknife.csv")
    grid = tab("admission_rule_grid.csv")
    n = 47

    fig = plt.figure(figsize=(W, 5.6))
    gtop = fig.add_gridspec(1, 1, left=0.40, right=0.98, top=0.96, bottom=0.60)
    gbot = fig.add_gridspec(1, 2, left=0.11, right=0.98, top=0.36, bottom=0.08,
                            wspace=0.45)

    # A  the same count, eight instruments
    ax = fig.add_subplot(gtop[0, 0])
    rows = [("Interventional SHAP, published", inv["interventional_published"], ORANGE),
            ("Interventional SHAP, recomputed", inv.get("interventional_recomputed"), ORANGE),
            ("Path-dependent TreeSHAP", inv["path_dependent"], ORANGE),
            ("Path-dependent TreeSHAP, out of fold", inv["oof_path_dependent"], ORANGE),
            ("Split gain", inv["total_gain"], ORANGE)]
    rows = [r for r in rows if r[1]]
    y = list(np.arange(len(rows) + 3)[::-1])
    lo, hi, mid = ex["range95_lo"], ex["range95_hi"], ex["expected"]
    ax.axvspan(lo, hi, color=LIGHT, lw=0, zorder=0)
    ax.axvline(mid, color=GREY, ls="--", lw=0.8, zorder=1)
    labels = []
    for yi, (lab, e, c) in zip(y, rows):
        ax.plot(e["noncentral_wins"], yi, "o", color=c, ms=6, zorder=3)
        ax.text(e["noncentral_wins"] + 0.8, yi, str(e["noncentral_wins"]),
                va="center", fontsize=FS_TICK,
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
        labels.append(lab)
    k = len(rows)
    sp = pd.DataFrame(ps["splits"])
    for j, (who, lab, c) in enumerate((("production", "Permutation, production learner", PINK),
                                       ("second", "Permutation, second learner", PINK))):
        yi = y[k + j]
        vals = sp[f"{who}_wins"].astype(int).to_numpy()
        ax.plot([vals.min(), vals.max()], [yi, yi], color=c, lw=1.4, zorder=2)
        ax.plot(vals, np.full(len(vals), yi), "o", color=c, ms=4, mfc="white",
                zorder=3)
        ax.text(vals.max() + 0.8, yi, f"{vals.min()} to {vals.max()}",
                va="center", fontsize=FS_TICK,
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
        labels.append(lab + f" ({len(vals)} splits)")
    mi = inv["mutual_information"]
    yi = y[k + 2]
    ax.plot(mi["noncentral_wins"], yi, "o", color=SKY, ms=6, zorder=3)
    ax.text(mi["noncentral_wins"] + 0.8, yi, str(mi["noncentral_wins"]),
            va="center", ha="left", fontsize=FS_TICK)
    labels.append("Mutual information (no model)")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, n)
    ax.set_xlabel(f"Fields where the non-central summary leads (of {n})")
    handles = [Line2D([], [], marker="o", ls="", color=ORANGE, label="decomposes a prediction"),
               Line2D([], [], marker="o", ls="", color=PINK, mfc="white", label="permutes a column"),
               Line2D([], [], marker="o", ls="", color=SKY, label="marginal, no model"),
               Patch(color=LIGHT, label="95% range, summaries interchangeable"),
               Line2D([], [], ls="--", color=GREY, label="expected, summaries interchangeable")]
    ax.legend(handles=handles, loc="upper center", ncol=2,
              bbox_to_anchor=(0.25, -0.20), columnspacing=1.2,
              handletextpad=0.3, fontsize=6.8)
    letter(ax, "A")

    # B  remove one state at a time
    ax = fig.add_subplot(gbot[0, 0])
    v = jk.sort_values("noncentral_wins")["noncentral_wins"].to_numpy()
    ax.plot(np.arange(1, len(v) + 1), v, "o", color=ORANGE, ms=3)
    ax.axhspan(lo, hi, color=LIGHT, lw=0, zorder=0)
    ax.axhline(mid, color=GREY, ls="--", lw=0.8)
    ax.set_ylim(0, n)
    ax.set_xlabel("State removed (sorted)")
    ax.set_ylabel(f"Non-central leads (of {n})")
    letter(ax, "B")

    # C  every placement of the admission rules
    ax = fig.add_subplot(gbot[0, 1])
    g = grid.copy()
    g["share"] = 100 * g["noncentral_wins"] / g["n_counted"]
    for exm, c, mk in ((True, ORANGE, "o"), (False, GREY, "s")):
        s = g[g["protected_pairs_exempt"] == exm]
        ax.scatter(s["n_modelled"], s["share"], s=14, color=c, marker=mk,
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="exemption on" if exm else "exemption off")
    ax.axhline(100 * ex["mean_share_noncentral"], color=GREY, ls="--", lw=0.8)
    ax.axhline(50, color=GREY, ls=":", lw=0.8)
    ax.set_ylim(40, 100)
    ax.set_xlabel("Columns admitted")
    ax.set_ylabel("Fields where non-central\nleads (%)")
    ax.legend(loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8,
              fontsize=6.6)
    letter(ax, "C")
    save(fig, "Figure06_invariance")


# ===========================================================================
#  FIGURE 7  Atmospheric lifetime orders the duration advantage
# ===========================================================================
def fig07():
    lt = js("lifetime_literature_test.json")
    sp = pd.DataFrame(lt["species_table"]).sort_values("lifetime_days")
    pairs = tab("duration_vs_concentration.csv")
    pairs["species"] = pairs["species"].str.lower().str.strip()
    cut = float(lt["published_cutoff"])
    fig = plt.figure(figsize=(W, 4.8))
    gs = fig.add_gridspec(2, 2, left=0.30, right=0.98, top=0.95, bottom=0.22,
                          hspace=0.62, wspace=0.50)

    # A  per species, the ratio, ordered by lifetime
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(sp))[::-1]
    for yi, (_, r) in zip(y, sp.iterrows()):
        c = ORANGE if r["lifetime_days"] < cut else BLUE
        ax.barh(yi, r["median_ratio"], color=c, height=0.6, left=0)
        pr = pairs[pairs["species"] == r["species"]]["ratio"].to_numpy()
        ax.plot(pr, np.full(len(pr), yi), "|", color=INK, ms=6, mew=0.8)
    ax.axvline(1, color=GREY, ls="--", lw=0.8)
    nice = {"pm2 5": "PM$_{2.5}$", "hydroxyl radical": "hydroxyl radical"}

    def days(d):
        if d >= 1:
            return f"{d:g} d"
        if d >= 0.01:
            return f"{d:.2g} d"
        e = int(np.floor(np.log10(d)))
        m = d / 10 ** e
        return (f"$10^{{{e}}}$ d" if abs(m - 1) < 0.05
                else f"${m:.0f}\\times10^{{{e}}}$ d")
    ax.set_yticks(y, [f"{nice.get(s, s)} ({days(d)})" for s, d in
                      zip(sp["species"], sp["lifetime_days"])])
    ax.set_xscale("log")
    ax.set_xlim(0.1, 120)
    ax.set_xlabel("Duration / concentration\nmean |SHAP| ratio")
    handles = [Patch(color=ORANGE, label=f"lifetime < {cut:g} days"),
               Patch(color=BLUE, label=f"lifetime ≥ {cut:g} days"),
               Line2D([], [], marker="|", ls="", color=INK, label="each percentile")]
    fig.legend(handles=handles, loc="lower center", fontsize=6.8, ncol=3,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.55, 0.0))
    letter(ax, "A")

    # B  ratio against lifetime, species as the unit
    ax = fig.add_subplot(gs[0, 1])
    for _, r in sp.iterrows():
        c = ORANGE if r["lifetime_days"] < cut else BLUE
        ax.plot(r["lifetime_days"], r["median_ratio"], "o", color=c, ms=4.5)
    ax.axhline(1, color=GREY, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-6, 200)
    ax.set_xlabel("Tropospheric lifetime (days)")
    ax.set_ylabel("Ratio\n(dimensionless)")
    rf = lt["rank_free_test"]
    ax.text(0.03, 0.96, f"Spearman {rf['spearman_rho']:+.2f}\n"
            f"permutation p = {rf['permutation_p']:.3f}\n"
            f"{lt['n_species']} species",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6)
    letter(ax, "B")

    # C  the class boundary swept
    ax = fig.add_subplot(gs[1, 1])
    rows = [r for r in lt["cutoff_sweep"]["rows"] if r["unit"] == "species"]
    cs = [r["cutoff_days"] for r in rows]
    sh = [r["short_wins"] / r["n_short"] for r in rows]
    lg = [r["long_wins"] / r["n_long"] for r in rows]
    ax.plot(cs, sh, "-o", color=ORANGE, ms=4, label="short-lived")
    ax.plot(cs, lg, "-s", color=BLUE, ms=4, label="longer-lived")
    ax.set_xscale("log")
    tk = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
          if min(cs) <= t <= max(cs)]
    ax.set_xticks(tk, [f"{t:g}" for t in tk])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Class boundary (days)")
    ax.set_ylabel("Share where duration\nwins (fraction)")
    ax.legend(loc="lower right", fontsize=6.6, handletextpad=0.3)
    letter(ax, "C")
    save(fig, "Figure07_lifetime")


# ===========================================================================
#  FIGURE 8  Turning points, and what the procedure returns from noise
# ===========================================================================
def fig08():
    tp = tab("turning_points.csv").set_index("feature")
    nc = js("null_crossing.json")["per_variable"]
    pt = tab("perturbation_thresholds.csv")
    order = tp.sort_values("width_frac").index.tolist()

    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(2, 2, left=0.28, right=0.98, top=0.92, bottom=0.10,
                          hspace=0.75, wspace=0.10, width_ratios=[1, 1])
    y = np.arange(len(order))[::-1]

    # A  interval width
    ax = fig.add_subplot(gs[0, 0])
    for yi, f in zip(y, order):
        r = tp.loc[f]
        c = GREEN if bool(r["stable"]) else RED
        ax.barh(yi, 100 * r["width_frac"], color=c, height=0.6)
    ax.axvline(40, color=GREY, ls="--", lw=0.8)
    ax.text(36, y[0], "40% criterion", fontsize=6.4, ha="right",
            va="center", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_yticks(y, [SHORT.get(f, f) for f in order])
    ax.set_xlabel("95% interval width\n(% of observed range)")
    handles = [Patch(color=GREEN, label="stable"), Patch(color=RED, label="not stable")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.6, ncol=2,
              bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, columnspacing=1.0)
    letter(ax, "A")

    # B  the same procedure on permuted attribution
    ax = fig.add_subplot(gs[0, 1])
    for yi, f in zip(y, order):
        rec = nc.get(f, {})
        ax.barh(yi, rec.get("permuted_crossing_found", 0), color=LIGHT,
                height=0.6)
        ax.plot(rec.get("permuted_gate_passed", 0), yi, "o", color=RED,
                ms=4, zorder=3)
    ax.set_yticks(y, [])
    nd = next(iter(nc.values()))["n_draws"]
    ax.set_xlim(0, nd)
    ax.set_xlabel(f"Permutations (of {nd})")
    handles = [Patch(color=LIGHT, label="crossing located"),
               Line2D([], [], marker="o", ls="", color=RED, label="criterion passed")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.6, ncol=2,
              bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, columnspacing=1.0)
    letter(ax, "B")

    # C  movement under predictor perturbation
    ax = fig.add_subplot(gs[1, :])
    for f, g in pt.groupby("feature"):
        g = g.sort_values("magnitude")
        c = PINK if f in NOCTURNAL else LIGHT
        ax.plot(100 * g["magnitude"], g["shift_over_interval_width"].abs(),
                "-", color=c, lw=0.9, zorder=2 if f in NOCTURNAL else 1)
    med = pt.groupby("magnitude")["shift_over_interval_width"].apply(
        lambda s: s.abs().median())
    ax.plot(100 * med.index, med.values, "-o", color=INK, ms=4, lw=1.4,
            zorder=3)
    ax.axhline(1, color=GREY, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.set_xlabel("Gaussian noise added to the predictor (% of its standard deviation)")
    ax.set_ylabel("Shift of the located value\n(published interval widths)")
    handles = [Line2D([], [], color=INK, marker="o", label="median, 11 variables"),
               Line2D([], [], color=PINK, label="nocturnal triplet"),
               Line2D([], [], color=LIGHT, label="other variables")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.8)
    letter(ax, "C")
    save(fig, "Figure08_turning_points")


# ===========================================================================
#  FIGURE 9  Nocturnal turning points that do not move with climate
# ===========================================================================
def fig09():
    from scipy.interpolate import UnivariateSpline
    dep, _ = C.dependence_table()
    tp = tab("turning_points.csv").set_index("feature")
    med = js("threshold_regional_invariance_all.json")
    mer = js("threshold_regional_invariance_100W_all.json")

    fig = plt.figure(figsize=(W, 5.6))
    top = fig.add_gridspec(1, 3, left=0.11, right=0.98, top=0.96, bottom=0.66,
                           wspace=0.45)
    bot = fig.add_gridspec(1, 1, left=0.30, right=0.98, top=0.50, bottom=0.15)
    for i, f in enumerate(NOCTURNAL):
        ax = fig.add_subplot(top[0, i])
        x = dep[f].to_numpy(float)
        v = dep["shap_" + f].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(v)
        x, v = x[ok], v[ok]
        lo, hi = np.percentile(x, [1, 99])
        qs = np.quantile(x[(x >= lo) & (x <= hi)], np.linspace(0, 1, 46))
        bx, by = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            s = (x >= a) & (x < b)
            if s.sum() > 2:
                bx.append(np.median(x[s]))
                by.append(np.median(v[s]))
        bx, by = np.array(bx), np.array(by)
        ax.plot(bx, by, "o", color=GREY, ms=2.2)
        r = tp.loc[f]
        ax.axvspan(r["ci_lo"], r["ci_hi"], color=PINK, alpha=0.35, lw=0)
        ax.axvline(r["point_estimate"], color=PINK, lw=1.0)
        ax.axhline(0, color="#555555", lw=0.6)
        nm, unit = NAMES[f]
        ax.set_xlabel(f"{SHORT[f]}\n({unit})", fontsize=7.8)
        if i == 0:
            ax.set_ylabel("SHAP (years)")
        u = "" if unit == "dimensionless" else f" {unit}"
        # the label goes in whichever left corner the curve leaves empty
        left_high = np.nanmedian(by[: max(len(by) // 3, 1)]) > 0
        ax.text(0.04, 0.05 if left_high else 0.95,
                f"{r['point_estimate']:.3g}{u}", transform=ax.transAxes,
                va="bottom" if left_high else "top", fontsize=FS_TICK,
                color=INK)
        letter(ax, "ABC"[i])

    # D  every candidate, two ways of cutting the country in two
    ax = fig.add_subplot(bot[0, 0])
    FLOOR, CEIL = 0.05, 60.0
    rows = []
    for f in tp.index:
        a = med["results"].get(f, {})
        b = mer["results"].get(f, {})
        rows.append((f, a.get("delta_pct_of_range"), b.get("delta_pct_of_range")))
    rows.sort(key=lambda r: (r[1] if r[1] is not None else 99))
    y = np.arange(len(rows))[::-1]
    clipped = False
    for yi, (f, a, b) in zip(y, rows):
        c = PINK if f in NOCTURNAL else GREY
        for v, dy, mk, face in ((a, 0.14, "o", c), (b, -0.14, "^", "white")):
            if v is None:
                continue
            v = abs(float(v))
            clipped |= v < FLOOR
            ax.plot(min(max(v, FLOOR), CEIL), yi + dy, mk, color=c, mfc=face,
                    ms=4.5, zorder=3)
        if b is None:
            ax.text(FLOOR * 1.1, yi - 0.14, "not located west of 100° W",
                    fontsize=6.4, va="center", color=INK)
    ax.set_yticks(y, [SHORT.get(r[0], r[0]) for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, CEIL)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.set_xlabel("|West minus east| (% of the variable's observed range)"
                  + ("\nvalues below 0.05 drawn at 0.05" if clipped else ""))
    handles = [Line2D([], [], marker="o", ls="", color=GREY,
                      label=f"split at median longitude ({abs(med['cut_degrees_east']):.1f}° W)"),
               Line2D([], [], marker="^", ls="", color=GREY, mfc="white",
                      label="split at the 100th meridian"),
               Line2D([], [], marker="s", ls="", color=PINK, label="nocturnal triplet")]
    ax.legend(handles=handles, loc="upper center", fontsize=6.6, ncol=2,
              bbox_to_anchor=(0.35, -0.30 if clipped else -0.22),
              handletextpad=0.3, columnspacing=1.2)
    letter(ax, "D")
    save(fig, "Figure09_climate_invariance")


# ===========================================================================
#  FIGURE 10  Where a physical model is blind
# ===========================================================================
def fig10():
    import geopandas as gpd
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    cty = tab("residual_social_determinants_counties.csv")
    res = js("residual_social_determinants.json")
    g = gpd.read_file(os.path.join(HERE, "conus_counties_5070.gpkg"), layer="county")
    g["fips"] = g["fips"].astype(int)
    m = g.merge(cty, left_on="fips", right_on=C.GROUP, how="left")

    fig = plt.figure(figsize=(W, 5.4))
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.97, bottom=0.10,
                          hspace=0.28, wspace=0.55, height_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, :])
    div = LinearSegmentedColormap.from_list(
        "div", ["#8A3A00", RED, "#F0B27A", "#F2F2F2", "#8FC7E8", BLUE, "#00355A"])
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    m[m["resid"].isna()].plot(ax=ax, color="#E6E6E6", linewidth=0)
    m[m["resid"].notna()].plot(ax=ax, column="resid", cmap=div, norm=norm,
                               linewidth=0)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=div)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.01, aspect=40, shrink=0.6)
    cb.set_label("County mean residual, observed minus predicted (years)",
                 fontsize=FS_TICK)
    cb.ax.tick_params(labelsize=6.8)
    letter(ax, "A")

    ax = fig.add_subplot(gs[1, 0])
    items = [(f"{res['n_sdoh_columns']} socioeconomic\ncolumns, never\nin the model",
              res["oof_r2_residual_from_sdoh"], GREY),
             (f"{res['n_env_columns']} environmental\ncolumns, all\nin the model",
              res["oof_r2_residual_from_environment"], BLUE)]
    for i, (lab, v, c) in enumerate(items):
        ax.bar(i, v, color=c, width=0.55)
        ax.text(i, v + (0.02 if v >= 0 else -0.07), f"{v:+.2f}", ha="center",
                fontsize=FS_TICK)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks([0, 1], [i[0] for i in items], fontsize=6.6)
    ax.set_ylim(-0.3, 0.6)
    ax.set_ylabel("Out-of-fold $R^2$ for the\nresidual (dimensionless)")
    letter(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    ind = sorted(res["indicators"], key=lambda r: r["rho"])
    yy = np.arange(len(ind))
    short = {"Population below the poverty line (%)": "Poverty",
             "Median household income (USD)": "Median income",
             "Unemployment rate (%)": "Unemployment",
             "Without health insurance (%)": "Uninsured",
             "Bachelor's degree or higher (%)": "Bachelor's degree",
             "Disability rate (%)": "Disability"}
    for yi, r in zip(yy, ind):
        c = RED if r["rho"] < 0 else BLUE
        ax.barh(yi, r["rho"], color=c, height=0.6)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks(yy, [short.get(r["label"], r["label"]) for r in ind])
    ax.set_xlim(-0.45, 0.45)
    ax.set_xlabel("Spearman correlation\nwith the residual")
    letter(ax, "C")
    save(fig, "Figure10_failure_geography")


FIGS = {2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07,
        8: fig08, 9: fig09, 10: fig10}


def main():
    global DPI, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    DPI = a.dpi
    if a.outdir:
        from pathlib import Path
        OUT = Path(a.outdir).resolve()
    want = ([int(x) for x in a.only.split(",") if x.strip()] if a.only
            else sorted(FIGS))
    failed = []
    for k in want:
        print(f"Figure {k}")
        try:
            FIGS[k]()
        except Exception as e:                         # report, keep going
            import traceback
            traceback.print_exc()
            failed.append(k)
    if failed:
        raise SystemExit(f"figures that failed: {failed}")


if __name__ == "__main__":
    main()
