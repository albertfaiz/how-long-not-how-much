#!/usr/bin/env python3
"""
paper4_figures_all_in_one.py -- every figure of Paper 4 from one file.

Figure 1 (study design), Figures 2 to 8 and the graphical abstract, drawn
from the result files the analysis scripts wrote. Nothing in this folder is
imported: the paths, the style and the code of every figure are below, so the
look of all figures can be changed here without touching the analysis. The
figure code is the code that drew the published figures; with the settings
as shipped, this file reproduces them.

WHAT TO EDIT is section 1, SETTINGS. Each setting says what it does, and the
most common ones can also be given on the command line.

Every run ends with a layout check. If a change makes two pieces of text
overlap, pushes a label onto another panel or off the canvas, the run names
the figure and the texts and exits with an error, so a broken figure cannot
pass unnoticed. The figures are still written, so you can look at them.

By default the figures go to ../Figures_styled/, so the paper's own Figures/
folder is never overwritten by an experiment. When a style is final, write it
to the paper with  --outdir ../Figures .

    python3 paper4_figures_all_in_one.py                         # everything
    python3 paper4_figures_all_in_one.py --only 1,5,8,ga         # some of them
    python3 paper4_figures_all_in_one.py --palette vibrant --labels "(A)"
    python3 paper4_figures_all_in_one.py --font-scale 1.1 --dpi 1200
    python3 paper4_figures_all_in_one.py --title-bold --check-only

READS   results_studyA_*/ and paper_final/derived/ (found by walking up from
        this file, or set SATSDOH_ROOT), ../constants_base.tex and
        ../constants_results.tex, conus_counties_5070.gpkg beside this file
NEEDS   numpy, pandas, matplotlib, geopandas, pyarrow
WRITES  Figure01_methodology ... Figure08_failure_geography, GraphicalAbstract
        as PNG (raster, DPI below) and PDF (vector)
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402
from matplotlib.patches import Patch                           # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text                               # noqa: E402

# ===========================================================================
#  1. SETTINGS  (the part to edit)
# ===========================================================================
DPI = 600                  # PNG resolution in dots per inch. 600 is the
                           # minimum this file accepts; 1200 for a poster.
FORMATS = ("png", "pdf")   # PDF is vector and does not depend on DPI.

PALETTE = "paper"          # "paper"   = the published colours (Okabe-Ito)
                           # "vibrant" = the same meanings, more saturated
                           # or add your own entry to PALETTES below.

PANEL_LABEL = "{}"         # how the panel letter is printed; {} is the letter
                           #   "{}"   ->  A        "({})" ->  (A)
                           #   "{}."  ->  A.       "{})"  ->  A)
PANEL_LABEL_BOLD = True
SHOW_PANEL_TITLES = True   # the short title after each panel letter
PANEL_TITLE_BOLD = False
SHOW_FIGURE_TITLES = True  # the one-line title above Figures 2 to 8
FIGURE_TITLE_BOLD = False

FONT_SCALE = 1.0           # multiplies every font size: 1.1 is 10% larger.
                           # Every value from 0.9 to 1.1 was checked and
                           # passes; above 1.1 the layout check tells you
                           # which label needs shortening.
FONT_FAMILY = "DejaVu Sans"

OUTDIR = None              # None -> ../Figures_styled ; or any folder
LAYOUT_CHECK = True        # stop with an error when text collides

# Colours carry meaning, and the meaning is the same in every figure:
#   central   mean or median, the model itself            (blue)
#   noncentral quantile, spread, duration                 (orange)
#   good      passes, stable                              (green)
#   bad       fails, not located, negative                (red)
#   accent    the three night-time variables              (pink)
#   second    a secondary series, held-out states         (sky blue)
#   stages    the four stages of Figure 1 and the abstract (white text on them)
#   diverging the residual map of Figure 8, red to blue through light grey
# Both palettes keep blue against orange and green against red apart under
# the common forms of colour-vision deficiency.
PALETTES = {
    "paper": dict(
        central="#0072B2", noncentral="#E69F00", good="#009E73",
        bad="#D55E00", accent="#CC79A7", second="#56B4E9",
        stages=["#0072B2", "#6A3D9A", "#D55E00", "#009E73"],
        diverging=["#8A3A00", "#D55E00", "#F0B27A", "#F2F2F2", "#8FC7E8",
                   "#0072B2", "#00355A"]),
    "vibrant": dict(
        central="#0057E0", noncentral="#FF9500", good="#00B377",
        bad="#F23D0C", accent="#E3349F", second="#12B5F5",
        stages=["#0057E0", "#7B2FBE", "#E0360A", "#00905F"],
        diverging=["#7A2400", "#F23D0C", "#F7AE8A", "#F2F2F2", "#8CCBF5",
                   "#0057E0", "#002C70"]),
}

# neutral inks, shared by every palette
GREY, LIGHT, INK = "#8C8C8C", "#D9D9D9", "#222222"
FILL = "#F4F4F4"

# sizes in points at the authored width; FONT_SCALE multiplies them all
W = 5.5                    # inches, the MDPI text width; do not change
FS, FS_TICK, FS_LET = 8.5, 7.5, 9.5
FS_TITLE = FS + 1.0


# ===========================================================================
#  2. WHERE THE DATA ARE  (from _config.py, unchanged)
# ===========================================================================
HERE = Path(__file__).resolve().parent


def _find_root() -> Path:
    env = os.environ.get("SATSDOH_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise SystemExit(f"SATSDOH_ROOT is set to {p}, which does not exist")
    here = HERE / "_"
    for q in [here.parent, *here.parents]:
        # Study A is the only run this paper reads. Study B (environment plus
        # socioeconomic data) is optional: no script here needs it.
        if list(q.glob("results_studyA_*")):
            return q
    # last resort: the folder whose name starts with "satellite"
    for q in here.parents:
        if q.name.lower().startswith("satellite"):
            return q
    raise SystemExit(
        "Could not find the project root. It is the folder that contains\n"
        "results_studyA_*. Either run this from inside\n"
        "that folder's tree, or set SATSDOH_ROOT to point at it:\n"
        "    export SATSDOH_ROOT=/Users/you/Desktop/satellite_sdoh")


def _one(pattern: str, parent: Path) -> Path:
    """The single directory matching a glob, newest if several."""
    hits = sorted(parent.glob(pattern))
    if not hits:
        return parent / pattern.replace("*", "MISSING")
    return hits[-1]


ROOT = _find_root()

# the two folders the main ML run produced
STUDY_A = _one("results_studyA_*", ROOT)
STUDY_B = _one("results_studyB_*", ROOT)

# where this analysis package lives, and where it writes
ANALYSIS = Path(__file__).resolve().parent
PAPER = ANALYSIS.parent                    # .../paper4_merged_sensors


def _named(env_name, default):
    v = os.environ.get(env_name)
    return Path(v).expanduser().resolve() if v else default


_RUN = (os.environ.get("SATSDOH_RUN") or "").strip().strip("_")
DERIVED = _named("SATSDOH_DERIVED", PAPER / ("derived" + (f"_{_RUN}" if _RUN else "")))
FRAME = STUDY_A / "studyA_reference_frame.parquet"
RANKING_A = STUDY_A / "studyA_shap_ranking.csv"
GROUP = "fips"
NCE_TAIL = "eng_night_cooling_eff"
NCE_MEAN = "eng2_night_cooling_efficiency"
COUNTY_FILE = next((p for p in (HERE / "conus_counties_5070.gpkg",
                                PAPER / "analysis" / "conus_counties_5070.gpkg")
                    if p.exists()), HERE / "conus_counties_5070.gpkg")


def table_candidates(stem: str):
    """
    Every path this project might have written the table `stem` to.

    A name like "A_cv_folds", "studyA_cv_folds" or "cv_folds" all mean the same
    table, and the study it belongs to is carried by the prefix. A B_ prefix
    sends the search to the Study B folders first, an A_ prefix or no prefix to
    Study A's.
    """
    stem = str(stem)
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    study = "A"
    for pre in ("studyA_", "studyB_", "A_", "B_"):
        if stem.startswith(pre):
            study = "B" if "B" in pre else "A"
            stem = stem[len(pre):]
            break
    names = [f"{stem}.csv", f"{study}_{stem}.csv", f"study{study}_{stem}.csv"]
    home = STUDY_B if study == "B" else STUDY_A
    folders = [DERIVED, home / "tables", home, PAPER / "derived"]
    # the other study, last, because a handful of tables were written once into
    # whichever folder the script that made them happened to be run from
    other = STUDY_A if study == "B" else STUDY_B
    folders += [other / "tables", other]
    for folder in folders:
        for n in names:
            yield folder / n


def load_table(stem: str, **kw):
    """
    Read a CSV by its bare name, from wherever in this project it lives.

        load_table("bracket_error")        # any of the three spellings above

    Raises with the full list of places tried, so a missing table is a question
    about which script has not been run rather than a stack trace.
    """
    import pandas as pd
    tried = []
    for p in table_candidates(stem):
        if p in tried:
            continue
        tried.append(p)
        if p.exists():
            return pd.read_csv(p, **kw)
    raise SystemExit(
        f"No table found for {stem!r}. Looked in:\n"
        + "\n".join(f"    {p}" for p in tried)
        + "\n    If the first of those is the one you expect, the script that\n"
          "    writes it has not been run yet; RUN_ORDER.md says which.")


def dependence_table():
    """
    (table, feature names) for script 05's attribution surface.

    The table is WIDE: one column per candidate holding the predictor value
    and one more named `shap_<candidate>` holding that row's attribution. Two
    later scripts assumed a long table with feature/x/shap columns and failed
    on the first read; the idiom is written down once here so a third does not.

        d, feats = C.dependence_table()
        x = d[f].to_numpy(float)
        v = d["shap_" + f].to_numpy(float)
    """
    d = load_table("shap_dependence_candidates")
    feats = [c for c in d.columns if "shap_" + c in d.columns]
    if not feats:
        raise SystemExit(
            "shap_dependence_candidates.csv carries no candidate paired with "
            "a shap_ column.\n    Columns: " + ", ".join(map(str, d.columns)))
    return d, feats


# ===========================================================================
#  3. HOUSE STYLE  (applied to every figure by save(), so none can drift)
# ===========================================================================
PAL = PALETTES[PALETTE]
BLUE = ORANGE = GREEN = RED = PINK = SKY = None
OUT = None
WRITE = True


def apply_settings():
    """Turn section 1 into colours, fonts and the output folder. Called once
    by main() after the command line has been read."""
    global PAL, BLUE, ORANGE, GREEN, RED, PINK, SKY, OUT
    if DPI < 600:
        raise SystemExit(f"DPI is {DPI}; 600 is the minimum for these figures.")
    if PALETTE not in PALETTES:
        raise SystemExit(f"PALETTE {PALETTE!r} is not one of {sorted(PALETTES)}")
    if "{}" not in PANEL_LABEL:
        raise SystemExit('PANEL_LABEL must contain {} where the letter goes, e.g. "({})"')
    PAL = PALETTES[PALETTE]
    BLUE, ORANGE = PAL["central"], PAL["noncentral"]
    GREEN, RED = PAL["good"], PAL["bad"]
    PINK, SKY = PAL["accent"], PAL["second"]
    OUT = Path(OUTDIR).expanduser().resolve() if OUTDIR else PAPER / "Figures_styled"
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FS, "font.weight": "normal",
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


def letter(ax, s):
    """Register a panel letter. It is placed at save time, at the outer left
    edge of the panel including its tick labels, so a letter can never sit on
    a label or fall off the canvas whatever the panel's margins are."""
    fig = ax.figure
    if not hasattr(fig, "_letters"):
        fig._letters = []
    fig._letters.append((ax, s))


# Figure titles and panel titles, one place for all of them.
TITLES = {
    "Figure02_performance": (
        "Predictive performance of the environment-only model",
        {"A": "Skill against baselines", "B": "Learning curve",
         "C": "Error by outcome bracket", "D": "Skill under outcome permutation"}),
    "Figure03_three_fields": (
        "Three fields, several descriptions",
        {"A": "Formaldehyde (CAMS)", "B": "Night cooling efficiency (MODIS)",
         "C": "Dew point (PRISM)", "D": "Position in the full ranking"}),
    "Figure04_drop_one_refit": (
        "Drop-one refit of night cooling efficiency",
        {"A": "Mean-form attribution", "B": "Mean-form rank",
         "C": "Predictive skill"}),
    "Figure05_composition_lifetime": (
        "Atmospheric composition and lifetime",
        {"A": "Chemical families", "B": "75th against 90th percentile",
         "C": "Duration advantage", "D": "Advantage against lifetime"}),
    "Figure06_turning_point_curves": (
        "Turning points of the attribution curves",
        {"A": "Minimum dew point", "B": "Night wet-bulb", "C": "Night cooling",
         "D": "Formaldehyde duration", "E": "Formaldehyde to NO$_2$",
         "F": "Dew point by quartile"}),
    "Figure07_turning_point_stability": (
        "Stability of the turning points",
        {"A": "Interval width", "B": "Permuted", "C": "Regional shift",
         "D": "Predictor noise"}),
    "Figure08_failure_geography": (
        "Geography of model error",
        {"A": "County residual", "B": "Predicting the residual",
         "C": "Socioeconomic correlates"}),
}


def _scale_fonts(fig):
    """Multiply every font size in the figure by FONT_SCALE. Tick labels are
    scaled through their axis, so ticks drawn later get the same size."""
    if FONT_SCALE == 1.0:
        return
    fig.canvas.draw()
    ticks = set()
    for ax in fig.axes:
        for which, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            tk = axis.get_major_ticks() + axis.get_minor_ticks()
            for t in tk:
                ticks.update((id(t.label1), id(t.label2)))
            if tk:
                ax.tick_params(axis=which, which="both",
                               labelsize=tk[0].label1.get_fontsize() * FONT_SCALE)
    for t in fig.findobj(Text):
        if id(t) not in ticks:
            t.set_fontsize(t.get_fontsize() * FONT_SCALE)


def _make_room(fig):
    """Larger text needs more room. With FONT_SCALE above 1 the canvas grows
    taller by the same factor, the left margin and the vertical gaps widen
    with it, and _resolve() widens the horizontal gaps only where text
    collides. The width stays the MDPI text width."""
    if FONT_SCALE <= 1.0:
        return
    from matplotlib.gridspec import GridSpec
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h * FONT_SCALE)
    seen = []
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        if ss is None:
            continue
        gs = ss.get_topmost_subplotspec().get_gridspec()
        if isinstance(gs, GridSpec) and gs not in seen:
            seen.append(gs)
            p = gs.get_subplot_params(fig)
            gs.update(left=min(p.left * FONT_SCALE, 0.45),
                      hspace=p.hspace * FONT_SCALE)


def _tick_owner(fig):
    """id of every tick label -> (axes, 'x' or 'y')."""
    own = {}
    for ax in fig.axes:
        for which, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                own[id(t.label1)] = own[id(t.label2)] = (ax, which)
    return own


def _wrap_once(label):
    """Break the longest line of a label at the space nearest its middle."""
    lines = label.split("\n")
    spaced = [i for i in range(len(lines)) if " " in lines[i]]
    if not spaced:
        return label
    k = max(spaced, key=lambda i: len(lines[i]))
    ln = lines[k]
    spaces = [i for i, c in enumerate(ln) if c == " "]
    if not spaces:
        return label
    cut = min(spaces, key=lambda i: abs(i - len(ln) / 2))
    lines[k:k + 1] = [ln[:cut], ln[cut + 1:]]
    return "\n".join(lines)


def _resolve(fig):
    """Only when FONT_SCALE is above 1: find the texts that the larger size
    makes collide and give them room, in up to six rounds. Crowded tick
    labels of one axis are eased back towards their original size; any other
    collision widens the gaps between panels. A figure with no collision is
    left exactly as it is."""
    if FONT_SCALE <= 1.0:
        return
    from matplotlib.gridspec import GridSpec
    base = {}
    for _ in range(6):
        clashes = _find_problems(fig, canvas=False)
        if not clashes:
            return
        own, widen = _tick_owner(fig), False
        for _msg, ti, tj in clashes:
            oi = own.get(id(ti))
            oj = own.get(id(tj)) if tj is not None else None
            if oi and oj and oi[0] is oj[0] and oi[1] != oj[1]:
                # an x and a y tick label meeting at a panel's corner
                for ax, which in (oi, oj):
                    axis = ax.xaxis if which == "x" else ax.yaxis
                    size = axis.get_major_ticks()[0].label1.get_fontsize()
                    floor = base.setdefault((id(ax), which), size / FONT_SCALE)
                    ax.tick_params(axis=which, labelsize=max(floor, size * 0.95))
            elif oi and oj and oi == oj:
                ax, which = oi
                axis = ax.xaxis if which == "x" else ax.yaxis
                labels = [t.get_text() for t in axis.get_ticklabels()]
                size = axis.get_major_ticks()[0].label1.get_fontsize()
                # first wrap a crowded multi-word label onto one more line
                wrapped = [_wrap_once(x) if x in (ti.get_text(), tj.get_text())
                           else x for x in labels]
                if wrapped != labels and which == "x":
                    axis.set_ticklabels(wrapped)
                    ax.tick_params(axis=which, labelsize=size)
                else:
                    # then ease the size back, never below the published size
                    floor = base.setdefault((id(ax), which), size / FONT_SCALE)
                    ax.tick_params(axis=which, labelsize=max(floor, size * 0.95))
            else:
                widen = True
        if widen:
            seen = []
            for ax in fig.axes:
                ss = ax.get_subplotspec()
                if ss is None:
                    continue
                gs = ss.get_topmost_subplotspec().get_gridspec()
                if isinstance(gs, GridSpec) and gs not in seen:
                    seen.append(gs)
                    p = gs.get_subplot_params(fig)
                    gs.update(wspace=p.wspace * 1.12)


def _place_letters(fig, panel_titles=None):
    """Panel letter at the outer left edge of each panel (tick labels
    included), followed on the same line by the panel title, both above
    anything the panel already carries on top."""
    panel_titles = panel_titles or {}
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, s in getattr(fig, "_letters", []):
        bb = ax.get_tightbbox(r).transformed(inv)
        pos = ax.get_position()
        # a long axis label can reach above the panel; the letter goes above it
        labs = [a.label.get_window_extent(r).transformed(inv).y1
                for a in (ax.xaxis, ax.yaxis)
                if a.label.get_visible() and a.label.get_text()]
        y = max([pos.y1, bb.y1] + labs) + 0.006
        t = fig.text(bb.x0, y, PANEL_LABEL.format(s), fontsize=FS_LET * FONT_SCALE,
                     fontweight="bold" if PANEL_LABEL_BOLD else "normal",
                     ha="left", va="bottom", color=INK)
        t._is_letter = True
        title = panel_titles.get(s) if SHOW_PANEL_TITLES else None
        if title:
            w = t.get_window_extent(r).transformed(inv).width
            fig.text(bb.x0 + w + 0.012, y, title, fontsize=FS * FONT_SCALE,
                     fontweight="bold" if PANEL_TITLE_BOLD else "normal",
                     ha="left", va="bottom", color=INK)


def _place_title(fig, text):
    """The figure title, centred above everything else on the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = fig.get_tightbbox(r).y1 / fig.get_figheight()   # inches to figure
    fig.text(0.5, top + 0.02, text, fontsize=FS_TITLE * FONT_SCALE, ha="center",
             va="bottom", color=INK,
             fontweight="bold" if FIGURE_TITLE_BOLD else "normal")


def save(fig, stem):
    """Figures 2 to 8: fonts, letters and title, the layout check, then the
    files. The canvas grows to hold every text (bbox_inches='tight')."""
    fig_title, panel_titles = TITLES.get(stem, ("", {}))
    # every text regular weight first, then the panel letters alone in bold
    for t in fig.findobj(Text):
        t.set_fontweight("normal")
    _make_room(fig)
    _scale_fonts(fig)
    _resolve(fig)
    _place_letters(fig, panel_titles)
    if fig_title and SHOW_FIGURE_TITLES:
        _place_title(fig, fig_title)
    check_layout(fig, stem, canvas=False)
    if WRITE:
        OUT.mkdir(parents=True, exist_ok=True)
        # Matplotlib clips an axis label to the canvas when it computes the
        # tight box, so a long label can lose its last characters. Listing
        # every text explicitly makes the saved area grow to hold it instead.
        extra = fig.get_default_bbox_extra_artists() + [
            t for ax in fig.axes for t in (ax.xaxis.label, ax.yaxis.label)
            if t.get_visible() and t.get_text()] + list(fig.texts)
        for ext in FORMATS:
            fig.savefig(OUT / f"{stem}.{ext}", dpi=DPI, facecolor="white",
                        bbox_inches="tight", pad_inches=0.04,
                        bbox_extra_artists=extra)
        print(f"  [saved] {OUT.name}/{stem} ({', '.join(FORMATS)}) at {DPI} dpi")
    plt.close(fig)


def _save(fig, stem: Path, dpi=None):
    """Figure 1 and the graphical abstract: fixed canvas, same check."""
    check_layout(fig, stem.name, canvas=True)
    if WRITE:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in FORMATS:
            fig.savefig(f"{stem}.{ext}", dpi=DPI, facecolor="white")
        print(f"  [saved] {stem.parent.name}/{stem.name} ({', '.join(FORMATS)}) at {DPI} dpi")
    plt.close(fig)


# ===========================================================================
#  4. LAYOUT CHECK
# ===========================================================================
#  Draws the figure once and records the box of every text actually drawn.
#  Three faults are reported:
#    * two texts whose boxes overlap,
#    * a text lying on the plotting area of a panel it does not belong to,
#    * (Figure 1 and the abstract, whose canvas is fixed) a text leaving the
#      canvas.
#  Text boxes are measured tight around the letters, so two texts must stay
#  at least GAP_PT points apart; a text may touch a panel's plotting area by
#  up to 1.5 points (its own blank margin).
PROBLEMS = []
TOL_PT = 1.5
GAP_PT = 0.8


def _drawn_texts(fig):
    drawn, orig, orig_extent = [], Text.draw, Text.get_window_extent

    def rec(self, renderer):
        orig(self, renderer)
        if self.get_visible() and self.get_text().strip() and self.get_alpha() != 0:
            # the text alone: an annotation's own extent includes its leader line
            drawn.append((self, orig_extent(self, renderer)))
    Text.draw = rec
    try:
        fig.canvas.draw()
    finally:
        Text.draw = orig
    return drawn


def _overlap(a, b, tol):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol
            and min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def _crowd(a, b, gap):
    """Side by side closer than `gap`, or one on top of the other."""
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > -gap
            and min(a.y1, b.y1) - max(a.y0, b.y0) > gap / 2)


def _owner(t):
    """The panel a text belongs to; None for the letters and titles, which
    belong to the figure."""
    return t.axes


def _find_problems(fig, canvas):
    """[(message, text, other text or None)] for every fault on the canvas."""
    tol = TOL_PT * fig.dpi / 72
    drawn = _drawn_texts(fig)
    found = []
    for i in range(len(drawn)):
        ti, bi = drawn[i]
        for j in range(i + 1, len(drawn)):
            tj, bj = drawn[j]
            if _crowd(bi, bj, GAP_PT * fig.dpi / 72):
                found.append((f"texts overlap or touch: {ti.get_text()!r} and "
                              f"{tj.get_text()!r}", ti, tj))
        own = _owner(ti)
        for ax in fig.axes:
            if ax is own or not ax.get_visible():
                continue
            if own is not None and (ax in getattr(own, "child_axes", [])
                                    or own in getattr(ax, "child_axes", [])):
                continue
            if _overlap(bi, ax.bbox, tol):
                found.append((f"text {ti.get_text()!r} lies on another panel",
                              ti, None))
                break
        if canvas:
            fb = fig.bbox
            if (bi.x0 < fb.x0 - tol or bi.x1 > fb.x1 + tol
                    or bi.y0 < fb.y0 - tol or bi.y1 > fb.y1 + tol):
                found.append((f"text {ti.get_text()!r} runs off the canvas",
                              ti, None))
    return found


def check_layout(fig, name, canvas):
    if not LAYOUT_CHECK:
        return
    msgs = [m for m, _a, _b in _find_problems(fig, canvas)]
    for f in dict.fromkeys(msgs):
        PROBLEMS.append((name, f))
        print(f"  [LAYOUT] {name}: {f}")


# ===========================================================================
#  5. FIGURE 1 AND THE GRAPHICAL ABSTRACT
# ===========================================================================
def read_constants(paper: Path) -> dict:
    """Every macro of the two constants files, resolved the way TeX resolves
    them: files in the order main.tex loads them, \\providecommand only when
    the name is still undefined, \\newcommand and \\renewcommand always.
    Values come back as printable text ({,} becomes a comma, $-$ a minus)."""
    vals = {}
    head = re.compile(r"\\(newcommand|renewcommand|providecommand)\{\\([A-Za-z]+)\}\{")
    for name in ("constants_base.tex", "constants_results.tex"):
        text = "\n".join(line.split("%")[0] if not line.lstrip().startswith("%")
                         else "" for line in (paper / name).read_text().splitlines())
        pos = 0
        while True:
            m = head.search(text, pos)
            if not m:
                break
            depth, i = 1, m.end()
            while depth and i < len(text):
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            value, pos = text[m.end():i - 1], i
            kind, key = m.group(1), m.group(2)
            if kind == "providecommand" and key in vals:
                continue
            vals[key] = (value.replace("{,}", ",").replace("$-$", "\u2212")
                         .replace("\\,", " ").replace("$", ""))
    return vals


def box(ax, x, y, w, h, fc=FILL, ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.8))


def arrow(ax, x0, y0, x1, y1, scale=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=scale, lw=0.9, color=GREY))


# ---------------------------------------------------------------- Figure 1
# The stage colours come from the palette (PALETTES, section 1).


def _bullets(fig, ax, x, y, items, width, size, gap):
    """Bullet lists with a hanging indent. Each item is wrapped to the box
    width by measuring the rendered text, so nothing runs past its box."""
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def wide(txt):
        t = ax.text(0, 0, txt, fontsize=size)
        bb = t.get_window_extent(renderer=r)
        t.remove()
        (a0, _), (a1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return a1 - a0

    line_h = size / 72 * 1.32 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / fig.get_figheight()
    for item in items:
        words, lines, cur = item.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if cur and wide(trial) > width - 0.22:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        ax.text(x, y, "\u2022", fontsize=size, va="top", ha="left", color=INK)
        for k, ln in enumerate(lines):
            ax.text(x + 0.20, y - k * line_h, ln, fontsize=size, va="top",
                    ha="left", color=INK)
        y -= len(lines) * line_h + gap
    return y


def _wrap(fig, ax, txt, width, size):
    """Split text into lines that fit `width` data units at `size` points."""
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def wide(t_):
        t = ax.text(0, 0, t_, fontsize=size)
        bb = t.get_window_extent(renderer=r)
        t.remove()
        (a0, _), (a1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return a1 - a0
    lines, cur = [], ""
    for w in txt.split():
        trial = (cur + " " + w).strip()
        if cur and wide(trial) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur]


def figure01(K: dict, out: Path):
    """Study design: four stages, each named and broken into its sub-steps.
    It carries the design only; results are left to Figures 2 to 10."""
    H0, YTOP = 3.35, 6.7
    fig = plt.figure(figsize=(W, H0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, YTOP)
    ax.axis("off")
    unit = H0 / YTOP                       # inches per data unit, kept fixed

    stages = [
        ("Stage 1", "Reduce to the county-year", [
            f"{K['Nmodalities']} sensing and reanalysis sources",
            f"{K['Ncounties']} counties, {K['YearStart']} to {K['YearEnd']}",
            "one pass over each field's values",
            "mean, median, quantiles, spread",
            "time above national levels for each gas"]),
        ("Stage 2", "Admit, model and attribute", [
            "coverage gate and correlation filter",
            f"{K['Nfeatpruned']} columns, no socioeconomic or clinical input",
            "XGBoost, eight county-grouped folds",
            "interventional SHAP, out of fold"]),
        ("Stage 3", "Compare descriptions", [
            "three fields, several summaries each",
            "formaldehyde, night cooling, dew point",
            "drop-one and modality-removal refits",
            "chemical families of the composition signal"]),
        ("Stage 4", "Test what survives", [
            f"atmospheric lifetime, {K['LtSpecies']} species",
            "turning points under bootstrap and noise",
            "turning points across two regional splits",
            "model error against socioeconomic data, used only for this audit"]),
    ]
    x0, wbox, gap = 0.12, 2.52, 0.24
    head_top, head_h = YTOP - 0.12, 1.45 * FONT_SCALE
    lowest = YTOP
    for i, (tag, title, items) in enumerate(stages):
        x = x0 + i * (wbox + gap)
        ax.text(x + 0.14, head_top - 0.30 * FONT_SCALE, tag, fontsize=7.2 * FONT_SCALE, color="white",
                va="center", ha="left", zorder=3)
        for k, ln in enumerate(_wrap(fig, ax, title, wbox - 0.28, 7.9 * FONT_SCALE)):
            ax.text(x + 0.14, head_top - (0.75 + k * 0.42) * FONT_SCALE, ln, fontsize=7.9 * FONT_SCALE,
                    color="white", va="center", ha="left", zorder=3)
        lowest = min(lowest, _bullets(fig, ax, x + 0.14, head_top - head_h - 0.22,
                                      items, wbox - 0.14, 7.0 * FONT_SCALE, 0.16))
    bottom = lowest - 0.05
    for i in range(len(stages)):
        x = x0 + i * (wbox + gap)
        ax.add_patch(FancyBboxPatch((x, bottom), wbox, head_top - bottom,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#F6F6F6", ec="none", zorder=0))
        ax.add_patch(FancyBboxPatch((x, head_top - head_h), wbox, head_h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=PAL["stages"][i], ec="none", zorder=1))
        if i < len(stages) - 1:
            arrow(ax, x + wbox + 0.03, head_top - head_h / 2,
                  x + wbox + gap - 0.03, head_top - head_h / 2, scale=8)
    # trim the empty space below the lowest box, keeping the text scale
    ymin = bottom - 0.12
    ax.set_ylim(ymin, YTOP)
    fig.set_size_inches(W, (YTOP - ymin) * unit)
    _save(fig, out / "Figure01_methodology")


# ---------------------------------------------------------------- abstract
def graphical_abstract(K: dict, out: Path):
    """The visual abstract, in the style of Figure 1: four stages left to
    right, each with its key numbers. Landscape, 6600 x 2820 px at 600 dpi,
    above MDPI's minimum of 1100 x 560 px."""
    fig = plt.figure(figsize=(11.0, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(1.0, 10.4)
    ax.axis("off")
    fs, fsn = 11 * FONT_SCALE, 15 * FONT_SCALE

    ax.text(11, 9.75, "How long, not how much: summaries of satellite data and life expectancy",
            fontsize=17 * FONT_SCALE, ha="center", va="center")

    stages = [
        ("1  Data and model", PAL["stages"][0], [
            ("text", f"{K['Nmodalities']} satellite and gridded modalities"),
            ("text", f"{K['Ncounties']} United States counties"),
            ("text", f"{K['YearStart']} to {K['YearEnd']}"),
            ("text", "no socioeconomic input"),
            ("num", f"R\u00b2 = {K['Rtwo']}"),
            ("text", "out of fold, county-grouped"),
        ]),
        ("2  Two descriptions", PAL["stages"][1], [
            ("text", "formaldehyde, time above P75"),
            ("num", f"rank {K['HchoDurRank']}"),
            ("text", "same field, annual mean"),
            ("num", f"rank {K['HchoConRank']}"),
            ("text", "night cooling, two forms"),
            ("num", f"{K['NCEratio']}\u00d7 apart"),
        ]),
        ("3  Chemistry", PAL["stages"][2], [
            ("text", "oxidised organic compounds carry"),
            ("num", f"{K['FamOxVOCPct']}% of composition credit"),
            ("text", "duration advantage vs gas lifetime"),
            ("num", f"\u03c1 = {K['LtRho']}"),
            ("text", "short-lived species favouring duration"),
            ("num", f"{K['LtShortWins']} of {K['LtShortN']}"),
        ]),
        ("4  Turning points and error", PAL["stages"][3], [
            ("text", "turning points, moisture and night cooling"),
            ("num", f"dew point {K['TpDew']} \u00b0C"),
            ("num", f"wet-bulb {K['TpWet']} \u00b0C"),
            ("text", "shift across the 100th meridian"),
            ("num", f"\u2264 {K['RegMerTripletMax']}% of range"),
            ("text", "social data predict the model's error"),
            ("num", f"R\u00b2 = {K['ResSdohRtwo']}"),
        ]),
    ]
    x0, wbox, gap = 0.3, 5.05, 0.36
    labels = []
    for i, (head, col, rows) in enumerate(stages):
        x = x0 + i * (wbox + gap)
        box(ax, x, 2.15, wbox, 6.65)
        ax.add_patch(FancyBboxPatch((x, 8.0), wbox, 0.8,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=col, ec="none"))
        ax.text(x + 0.22, 8.4, head, fontsize=fs + 1, color="white",
                va="center", ha="left")
        y = 7.35
        for kind, txt in rows:
            if kind == "text":
                labels.append(_fit_text(fig, ax, x + 0.22, y, txt, fs - 1.5,
                                        wbox - 0.40, color="#444444"))
                y -= 0.58
            else:
                ax.text(x + 0.22, y, txt, fontsize=fsn if len(txt) < 22 else fs + 1,
                        va="center", color=INK)
                y -= 0.85
        if i < len(stages) - 1:
            arrow(ax, x + wbox + 0.03, 5.2, x + wbox + gap - 0.03, 5.2, scale=16)

    # one size for every label: the largest that lets the longest one fit
    size = min(lab.get_fontsize() for lab in labels)
    for lab in labels:
        lab.set_fontsize(size)
    _fit_text(fig, ax, 11, 1.55, "How a field is summarised changes what the model "
              "credits, up to 70-fold; gas lifetimes and stable night-time turning "
              "points are consistent with the physics.", fs, 21.4, ha="center", color=INK)
    _save(fig, out / "GraphicalAbstract")


def _fit_text(fig, ax, x, y, txt, size, maxw, **kw):
    """Draw text and shrink it until it fits inside maxw data units, so no
    label runs past the edge of its box whatever fonts the machine has."""
    t = ax.text(x, y, txt, fontsize=size, va="center", **kw)
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    while size > 6:
        bb = t.get_window_extent(renderer=r)
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if x1 - x0 <= maxw:
            break
        size -= 0.25
        t.set_fontsize(size)
    return t


# ===========================================================================
#  6. FIGURES 2 TO 8
# ===========================================================================
def js(name):
    return json.load(open(DERIVED / name))


def tab(name):
    p = DERIVED / name
    return pd.read_csv(p) if p.exists() else load_table(name.replace(".csv", ""))


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

# Chemical families of the atmospheric-composition columns. The first rule a
# column name matches decides its family, and every composition column falls
# in a named family. build_paper_constants.py imports these same rules, so the
# shares printed in the text are the shares drawn in Figure 5A.
FAMILY_RULES = [
    ("Oxidised VOC", ("formaldehyde", "hydrogen_peroxide")),
    ("Alkanes", ("propane", "ethane")),
    ("Sulphur species", ("sulphur_dioxide", "sulphate")),
    ("Reactive nitrogen", ("nitrogen_dioxide", "nitrogen_monoxide", "nitric_acid",
                           "peroxyacetyl")),
    ("Oxidants", ("ozone", "hydroxyl")),
    ("Biogenic VOC", ("isoprene",)),
    ("Carbon monoxide", ("carbon_monoxide", "carbonmonoxide")),
    ("Carbonaceous aerosol", ("black_carbon", "organic_matter")),
    ("Dust and sea salt", ("dust", "sea_salt")),
    ("Particulate matter", ("pm1", "pm$1$", "pm2_5", "pm10", "total_aod")),
]


def cams_family(feature):
    for fam_, keys in FAMILY_RULES:
        if any(k in feature for k in keys):
            return fam_
    return "Other composition"


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
#  FIGURE 3  Three fields, several descriptions
# ===========================================================================
def fig03():
    rk = pd.read_csv(RANKING_A).sort_values("rank")
    val = dict(zip(rk["feature"], rk["mean_abs_shap"]))
    rnk = dict(zip(rk["feature"], rk["rank"]))

    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(2, 2, left=0.22, right=0.98, top=0.94, bottom=0.10,
                          hspace=0.62, wspace=0.95)

    def bars(ax, items):
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
    bars(ax, [("Time above P75", "cams_fot_formaldehyde_above75th_percentile", ORANGE),
              ("Time above P90", "cams_fot_formaldehyde_above_90th_percentile", ORANGE),
              ("Annual mean", "cams_total_column_formaldehyde_x1e9", BLUE)])
    letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    bars(ax, [("Tail form", NCE_TAIL, ORANGE), ("Mean form", NCE_MEAN, BLUE)])
    letter(ax, "B")

    # C  seven summaries of one dew-point field over the cells of a county
    ax = fig.add_subplot(gs[1, 0])
    names = {"min": "Minimum", "stddev": "Variability", "p10": "10th percentile",
             "p90": "90th percentile", "p50": "Median", "mean": "Mean",
             "max": "Maximum"}
    dp = rk[rk["feature"].str.startswith("prism_tdmean_")].sort_values("rank")
    items = [(names[f.split("prism_tdmean_")[1]], f,
              BLUE if f.endswith(("_mean", "_p50")) else ORANGE)
             for f in dp["feature"]]
    bars(ax, items)
    letter(ax, "C")

    # D  where the descriptions sit in the ranking of all admitted variables;
    # labels stand in one column at the right, joined to their points by
    # leaders that never cross, because the points and labels share an order
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rk["rank"], rk["mean_abs_shap"], color=GREY, lw=1.0)
    marks = (("cams_fot_formaldehyde_above75th_percentile", ORANGE, "HCHO, time above P75"),
             (NCE_TAIL, ORANGE, "NCE, tail form"),
             ("prism_tdmean_min", ORANGE, "Dew point, minimum"),
             (NCE_MEAN, BLUE, "NCE, mean form"),
             ("cams_total_column_formaldehyde_x1e9", BLUE, "HCHO, annual mean"),
             ("prism_tdmean_mean", BLUE, "Dew point, mean"))
    ys = (0.55, 0.20, 0.075, 0.028, 0.0105, 0.0003)
    xs = (250, 250, 250, 250, 250, 160)
    for (f, c, lab), yt, xt in zip(marks, ys, xs):
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
    letter(ax, "D")
    save(fig, "Figure03_three_fields")


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
#  FIGURE 5  Where the composition signal sits, and atmospheric lifetime
# ===========================================================================
def fig05():
    rk = pd.read_csv(RANKING_A)
    comp = rk[rk["feature"].str.startswith("cams")].copy()
    comp["family"] = comp["feature"].map(cams_family)
    tot = comp["mean_abs_shap"].sum()
    fam = (comp.groupby("family")["mean_abs_shap"].agg(["sum", "size"])
           .sort_values("sum"))
    lt = js("lifetime_literature_test.json")
    sp = pd.DataFrame(lt["species_table"]).sort_values("lifetime_days")
    pairs = tab("duration_vs_concentration.csv")
    pairs["species"] = pairs["species"].str.lower().str.strip()
    cut = float(lt["published_cutoff"])
    nice = {"pm2 5": "PM$_{2.5}$", "pm1": "PM$_1$",
            "nitrogen monoxide": "nitric oxide"}

    fig = plt.figure(figsize=(W, 6.0))
    gs = fig.add_gridspec(2, 2, left=0.30, right=0.98, top=0.95, bottom=0.18,
                          hspace=0.58, wspace=1.05, height_ratios=[0.85, 1.2])

    # A  share of composition attribution by chemical family
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(fam))
    cols = [ORANGE if f_ == "Oxidised VOC" else BLUE for f_ in fam.index]
    share = 100 * fam["sum"] / tot
    ax.barh(y, share, color=cols, height=0.62)
    for yi, (v, k) in zip(y, zip(share, fam["size"])):
        ax.text(v + 1.5, yi, f"{v:.1f} (n = {k})", va="center", fontsize=6.8)
    ax.set_yticks(y, fam.index)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Share of composition\nattribution (%)")
    letter(ax, "A")

    # B  each species' time above the 75th against the 90th percentile
    ax = fig.add_subplot(gs[0, 1])
    du = comp[comp["feature"].str.contains("_fot_")
              & comp["feature"].str.contains("percentile")].copy()
    du["pct"] = np.where(du["feature"].str.contains("75th"), 75, 90)
    du["species"] = (du["feature"].str.replace("cams_fot_", "", regex=False)
                     .str.replace(r"_?above_?\d+th_percentile", "", regex=True)
                     .str.replace("$1$", "1", regex=False).str.replace("_", " "))
    piv = (du.pivot_table(index="species", columns="pct", values="mean_abs_shap")
           .dropna().sort_values(75))
    yy = np.arange(len(piv))
    for yi, (_, r) in zip(yy, piv.iterrows()):
        ax.plot([r[75], r[90]], [yi, yi], color=LIGHT, lw=1.4, zorder=1)
    ax.plot(piv[75], yy, "o", color=INK, ms=5, zorder=2)
    ax.plot(piv[90], yy, "s", color=GREY, mfc="white", ms=4.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 1.0)
    ax.set_yticks(yy, [nice.get(s_, s_) for s_ in piv.index])
    ax.set_xlabel("Mean |SHAP| of the\nduration (years)")
    letter(ax, "B")

    # C  per species, the duration to concentration ratio, ordered by lifetime
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(sp))[::-1]
    for yi, (_, r) in zip(y, sp.iterrows()):
        c = ORANGE if r["lifetime_days"] < cut else BLUE
        ax.barh(yi, r["median_ratio"], color=c, height=0.6, left=0)
        pr = pairs[pairs["species"] == r["species"]]["ratio"].to_numpy()
        ax.plot(pr, np.full(len(pr), yi), "|", color=INK, ms=6, mew=0.8)
    ax.axvline(1, color=GREY, ls="--", lw=0.8)

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
    letter(ax, "C")

    # D  the ratio against lifetime, species as the unit
    ax = fig.add_subplot(gs[1, 1])
    for _, r in sp.iterrows():
        c = ORANGE if r["lifetime_days"] < cut else BLUE
        ax.plot(r["lifetime_days"], r["median_ratio"], "o", color=c, ms=4.5)
    ax.axhline(1, color=GREY, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-6, 200)
    ax.set_xticks([1e-5, 1e-3, 1e-1, 1e1])
    ax.set_ylim(0.15, 1000)
    ax.set_xlabel("Atmospheric lifetime (days)")
    ax.set_ylabel("Ratio (dimensionless)")
    rf = lt["rank_free_test"]
    rho = f"{rf['spearman_rho']:+.2f}".replace("-", "\u2212")
    ax.text(0.04, 0.96, f"Spearman {rho}\n"
            f"permutation p = {rf['permutation_p']:.3f}\n"
            f"{lt['n_species']} species",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6)
    letter(ax, "D")

    handles = [Line2D([], [], marker="o", ls="", color=INK,
                      label="B: time above the 75th percentile"),
               Line2D([], [], marker="s", ls="", color=GREY, mfc="white",
                      label="B: time above the 90th percentile"),
               Patch(color=ORANGE, label=f"C, D: lifetime < {cut:g} days"),
               Patch(color=BLUE, label=f"C, D: lifetime ≥ {cut:g} days"),
               Line2D([], [], marker="|", ls="", color=INK,
                      label="C: ratio at each percentile")]
    fig.legend(handles=handles, loc="lower center", fontsize=6.8, ncol=2,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.55, 0.0))
    save(fig, "Figure05_composition_lifetime")


# ===========================================================================
#  FIGURE 6  Turning points of the attribution curves
# ===========================================================================
CURVE_XLABEL = {
    "prism_tdmean_min": "Minimum dew point\n(°C)",
    "eng2_wetbulb_night": "Night-time wet-bulb\ntemperature (°C)",
    "eng_night_cooling_eff": "Night cooling, tail\nform (dimensionless)",
    "cams_fot_formaldehyde_above75th_percentile":
        "Formaldehyde, time above\n75th percentile (% of year)",
    "eng2_hcho_no2_ratio": "Formaldehyde to NO$_2$\nratio (dimensionless)",
}


def _attribution_curve(ax, dep, tp, f):
    """Median SHAP in 45 equal-count bins between the 1st and 99th
    percentiles, the located turning point and its 95% bootstrap interval."""
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
    ax.set_xlabel(CURVE_XLABEL[f], fontsize=7.8)
    unit = NAMES[f][1]
    u = "" if unit == "dimensionless" else f" {unit}"
    left_high = np.nanmedian(by[: max(len(by) // 3, 1)]) > 0
    ax.text(0.04, 0.05 if left_high else 0.95,
            f"{r['point_estimate']:.3g}{u}", transform=ax.transAxes,
            va="bottom" if left_high else "top", fontsize=FS_TICK, color=INK)


def fig06():
    dep, _ = dependence_table()
    tp = tab("turning_points.csv").set_index("feature")
    st = load_table("A_temp_strata")
    feats = list(NOCTURNAL) + ["cams_fot_formaldehyde_above75th_percentile",
                               "eng2_hcho_no2_ratio"]
    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(2, 3, left=0.11, right=0.98, top=0.94, bottom=0.15,
                          hspace=0.95, wspace=0.68)
    for i, f in enumerate(feats):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        _attribution_curve(ax, dep, tp, f)
        if i % 3 == 0:
            ax.set_ylabel("SHAP (years)")
        letter(ax, "ABCDE"[i])

    # F  the dew-point crossing within quartiles of minimum temperature
    ax = fig.add_subplot(gs[1, 2])
    pe = tp.loc["prism_tdmean_min", "point_estimate"]
    for i, r in st.reset_index(drop=True).iterrows():
        if np.isfinite(r["zero_cross"]):
            ax.plot(i, r["zero_cross"], "o", color=PINK, ms=5)
            ax.text(i, r["zero_cross"] + 0.3, f"{r['zero_cross']:.2f}",
                    ha="center", fontsize=6.6)
        else:
            ax.text(i, pe - 1.2, "none", ha="center", va="center",
                    fontsize=6.6, color=INK)
    ax.axhline(pe, color=GREY, ls="--", lw=0.8)
    ax.set_xticks(range(len(st)), [s.split()[0] for s in st["stratum"]])
    ax.set_xlim(-0.6, len(st) - 0.4)
    ax.set_ylim(pe - 2, pe + 2)
    ax.set_xlabel("Quartile of minimum\ntemperature (coldest Q1)", fontsize=7.8)
    ax.set_ylabel("Crossing (°C)")
    letter(ax, "F")
    save(fig, "Figure06_turning_point_curves")


# ===========================================================================
#  FIGURE 7  The turning points under bootstrap, noise and two regional splits
# ===========================================================================
def fig07():
    tp = tab("turning_points.csv").set_index("feature")
    nc = js("null_crossing.json")["per_variable"]
    pt = tab("perturbation_thresholds.csv")
    med = js("threshold_regional_invariance_all.json")
    mer = js("threshold_regional_invariance_100W_all.json")
    order = tp.sort_values("width_frac").index.tolist()

    fig = plt.figure(figsize=(W, 5.4))
    top = fig.add_gridspec(1, 3, left=0.34, right=0.98, top=0.93, bottom=0.53,
                           wspace=0.14)
    bot = fig.add_gridspec(1, 1, left=0.34, right=0.98, top=0.27, bottom=0.08)
    y = np.arange(len(order))[::-1]

    # A  interval width
    ax = fig.add_subplot(top[0, 0])
    for yi, f in zip(y, order):
        r = tp.loc[f]
        ax.barh(yi, 100 * r["width_frac"], color=GREEN if bool(r["stable"]) else RED,
                height=0.6)
    ax.axvline(40, color=GREY, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.set_yticks(y, [SHORT.get(f, f) for f in order])
    ax.set_xlabel("95% interval width\n(% of range)")
    letter(ax, "A")

    # B  the same procedure on permuted attribution
    ax = fig.add_subplot(top[0, 1])
    for yi, f in zip(y, order):
        rec = nc.get(f, {})
        ax.barh(yi, rec.get("permuted_crossing_found", 0), color=LIGHT, height=0.6)
        ax.plot(rec.get("permuted_gate_passed", 0), yi, "o", color=RED, ms=4,
                zorder=3)
    ax.set_yticks(y, [])
    nd = next(iter(nc.values()))["n_draws"]
    ax.set_xlim(0, nd)
    ax.set_xlabel(f"Permutations\n(of {nd})")
    letter(ax, "B")

    # C  west minus east, at the median longitude and at the 100th meridian
    ax = fig.add_subplot(top[0, 2])
    FLOOR, CEIL = 0.05, 60.0
    for yi, f in zip(y, order):
        c = PINK if f in NOCTURNAL else GREY
        a = med["results"].get(f, {}).get("delta_pct_of_range")
        b = mer["results"].get(f, {}).get("delta_pct_of_range")
        for v, dy, mk, face in ((a, 0.16, "o", c), (b, -0.16, "^", "white")):
            if v is None:
                ax.plot(CEIL * 0.75, yi + dy, "x", color=c, ms=4.5, zorder=3)
                continue
            ax.plot(min(max(abs(float(v)), FLOOR), CEIL), yi + dy, mk, color=c,
                    mfc=face, ms=4.5, zorder=3)
    ax.set_yticks(y, [])
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, CEIL)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.set_xlabel("|West minus east|\n(% of range)")
    letter(ax, "C")

    handles = [Patch(color=GREEN, label="stable"),
               Patch(color=RED, label="not stable"),
               Patch(color=LIGHT, label="crossing located in a permutation"),
               Line2D([], [], marker="o", ls="", color=RED,
                      label="permutation passed the criterion"),
               Line2D([], [], marker="o", ls="", color=GREY,
                      label=f"split at {abs(med['cut_degrees_east']):.1f}° W"),
               Line2D([], [], marker="^", ls="", color=GREY, mfc="white",
                      label="split at 100° W"),
               Line2D([], [], marker="x", ls="", color=GREY,
                      label="no crossing in one half"),
               Line2D([], [], marker="s", ls="", color=PINK,
                      label="night-time moisture and cooling")]
    fig.legend(handles=handles, loc="center", fontsize=6.6, ncol=2,
               handletextpad=0.3, columnspacing=1.2,
               bbox_to_anchor=(0.62, 0.375))

    # D  movement under predictor perturbation
    ax = fig.add_subplot(bot[0, 0])
    for f, g in pt.groupby("feature"):
        g = g.sort_values("magnitude")
        c = PINK if f in NOCTURNAL else LIGHT
        ax.plot(100 * g["magnitude"], g["shift_over_interval_width"].abs(),
                "-", color=c, lw=0.9, zorder=2 if f in NOCTURNAL else 1)
    mm = pt.groupby("magnitude")["shift_over_interval_width"].apply(
        lambda s: s.abs().median())
    ax.plot(100 * mm.index, mm.values, "-o", color=INK, ms=4, lw=1.4, zorder=3)
    ax.axhline(1, color=GREY, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Gaussian noise added to the predictor (% of its standard deviation)")
    ax.set_ylabel("Shift (interval\nwidths)")
    handles = [Line2D([], [], color=INK, marker="o",
                      label=f"median, {len(order)} variables"),
               Line2D([], [], color=PINK, label="night-time moisture and cooling"),
               Line2D([], [], color=LIGHT, label="other variables")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.6)
    letter(ax, "D")
    save(fig, "Figure07_turning_point_stability")


# ===========================================================================
#  FIGURE 8  Where a physical model is blind
# ===========================================================================
def fig08():
    import geopandas as gpd
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    cty = tab("residual_social_determinants_counties.csv")
    res = js("residual_social_determinants.json")
    g = gpd.read_file(COUNTY_FILE, layer="county")
    g["fips"] = g["fips"].astype(int)
    m = g.merge(cty, left_on="fips", right_on=GROUP, how="left")

    fig = plt.figure(figsize=(W, 5.4))
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.97, bottom=0.10,
                          hspace=0.28, wspace=0.55, height_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, :])
    div = LinearSegmentedColormap.from_list(
        "div", PAL["diverging"])
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    m[m["resid"].isna()].plot(ax=ax, color="#E6E6E6", linewidth=0)
    m[m["resid"].notna()].plot(ax=ax, column="resid", cmap=div, norm=norm,
                               linewidth=0)
    # state outlines, dissolved from the same county polygons (state = FIPS // 1000)
    states = g.assign(state=g["fips"] // 1000).dissolve(by="state")
    states.boundary.plot(ax=ax, color="#3A3A3A", linewidth=0.35)
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
    save(fig, "Figure08_failure_geography")


# ===========================================================================
#  7. RUN
# ===========================================================================
def _schematic(which):
    K = read_constants(PAPER)
    (figure01 if which == "1" else graphical_abstract)(K, OUT)


FIGS = {"1": lambda: _schematic("1"), "2": fig02, "3": fig03, "4": fig04,
        "5": fig05, "6": fig06, "7": fig07, "8": fig08,
        "ga": lambda: _schematic("ga")}


def main():
    global DPI, PALETTE, PANEL_LABEL, FONT_SCALE, OUTDIR, WRITE
    global FIGURE_TITLE_BOLD, PANEL_LABEL_BOLD
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="e.g. 1,5,8,ga")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--palette", default=PALETTE, choices=sorted(PALETTES))
    ap.add_argument("--labels", default=PANEL_LABEL, help='e.g. "({})" or "(A)"')
    ap.add_argument("--font-scale", type=float, default=FONT_SCALE)
    ap.add_argument("--title-bold", action="store_true")
    ap.add_argument("--labels-regular", action="store_true",
                    help="panel letters in regular weight")
    ap.add_argument("--outdir", default=OUTDIR or "")
    ap.add_argument("--check-only", action="store_true",
                    help="draw and check every figure, write nothing")
    a = ap.parse_args()
    DPI, PALETTE, FONT_SCALE = a.dpi, a.palette, a.font_scale
    # accept "(A)" as well as "({})"
    PANEL_LABEL = a.labels if "{}" in a.labels else a.labels.replace("A", "{}", 1)
    OUTDIR = a.outdir or None
    WRITE = not a.check_only
    FIGURE_TITLE_BOLD = FIGURE_TITLE_BOLD or a.title_bold
    PANEL_LABEL_BOLD = PANEL_LABEL_BOLD and not a.labels_regular
    apply_settings()
    want = ([x.strip().lower() for x in a.only.split(",") if x.strip()]
            if a.only else list(FIGS))
    print(f"palette {PALETTE}, panel label {PANEL_LABEL.format('A')}, "
          f"font scale {FONT_SCALE}, {DPI} dpi -> {OUT}")
    failed = []
    for k in want:
        if k not in FIGS:
            raise SystemExit(f"unknown figure {k!r}; choose from {list(FIGS)}")
        print(f"Figure {k}")
        try:
            FIGS[k]()
        except Exception:                              # report, keep going
            import traceback
            traceback.print_exc()
            failed.append(k)
    if failed:
        raise SystemExit(f"figures that failed: {failed}")
    if PROBLEMS:
        raise SystemExit(f"{len(PROBLEMS)} layout problem(s), listed above as "
                         "[LAYOUT]. The files were written so you can see "
                         "them; a smaller FONT_SCALE or a shorter label "
                         "usually clears them.")
    print("layout check passed: no overlapping or misplaced text")


if __name__ == "__main__":
    main()
