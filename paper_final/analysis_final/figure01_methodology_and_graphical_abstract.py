#!/usr/bin/env python3
"""
figure01_methodology_and_graphical_abstract.py -- the two schematic figures.

READS   ../constants_base.tex and ../constants_results.tex, the same generated
        constants the manuscript prints, so every number in these figures is
        the number in the text
WRITES  ../Figures/Figure01_methodology.png and .pdf     (Figure 1)
        ../Figures/GraphicalAbstract.png and .pdf       (uploaded separately)

Both are placeholders in the sense that a hand-drawn version can replace
either: save it under the same file name and the manuscript picks it up.
They are drawn here so that the paper compiles complete without one.

House style as in paper_figures.py: DejaVu Sans, regular weight only, the
Okabe-Ito colours with the same meaning as in Figures 2 to 10.

    python3 figure01_methodology_and_graphical_abstract.py
    python3 figure01_methodology_and_graphical_abstract.py --paper ../../paper_final
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ---------------------------------------------------------------- settings
HERE = Path(__file__).resolve().parent
PAPER_DEFAULT = HERE.parent           # the folder holding main.tex
DPI = 600
W = 5.5                               # inches, the MDPI text width

BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PINK, SKY, GREY, INK = "#CC79A7", "#56B4E9", "#8C8C8C", "#222222"
FILL = "#F4F4F4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.weight": "normal",
    "text.color": INK, "mathtext.default": "regular",
    "pdf.fonttype": 42, "figure.facecolor": "white",
})


# ---------------------------------------------------------------- constants
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
# One colour per stage, so no two stages share a colour (white text on each
# passes a contrast check for large text).
STAGE_COLOURS = ["#0072B2", "#6A3D9A", "#D55E00", "#009E73"]


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
    head_top, head_h = YTOP - 0.12, 1.45
    lowest = YTOP
    for i, (tag, title, items) in enumerate(stages):
        x = x0 + i * (wbox + gap)
        ax.text(x + 0.14, head_top - 0.30, tag, fontsize=7.2, color="white",
                va="center", ha="left", zorder=3)
        for k, ln in enumerate(_wrap(fig, ax, title, wbox - 0.28, 7.9)):
            ax.text(x + 0.14, head_top - 0.75 - k * 0.42, ln, fontsize=7.9,
                    color="white", va="center", ha="left", zorder=3)
        lowest = min(lowest, _bullets(fig, ax, x + 0.14, head_top - head_h - 0.22,
                                      items, wbox - 0.14, 7.0, 0.16))
    bottom = lowest - 0.05
    for i in range(len(stages)):
        x = x0 + i * (wbox + gap)
        ax.add_patch(FancyBboxPatch((x, bottom), wbox, head_top - bottom,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#F6F6F6", ec="none", zorder=0))
        ax.add_patch(FancyBboxPatch((x, head_top - head_h), wbox, head_h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=STAGE_COLOURS[i], ec="none", zorder=1))
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
    fs, fsn = 11, 15

    ax.text(11, 9.75, "How long, not how much: summaries of satellite data and life expectancy",
            fontsize=17, ha="center", va="center")

    stages = [
        ("1  Data and model", STAGE_COLOURS[0], [
            ("text", f"{K['Nmodalities']} satellite and gridded modalities"),
            ("text", f"{K['Ncounties']} United States counties"),
            ("text", f"{K['YearStart']} to {K['YearEnd']}"),
            ("text", "no socioeconomic input"),
            ("num", f"R\u00b2 = {K['Rtwo']}"),
            ("text", "out of fold, county-grouped"),
        ]),
        ("2  Two descriptions", STAGE_COLOURS[1], [
            ("text", "formaldehyde, time above P75"),
            ("num", f"rank {K['HchoDurRank']}"),
            ("text", "same field, annual mean"),
            ("num", f"rank {K['HchoConRank']}"),
            ("text", "night cooling, two forms"),
            ("num", f"{K['NCEratio']}\u00d7 apart"),
        ]),
        ("3  Chemistry", STAGE_COLOURS[2], [
            ("text", "oxidised organic compounds carry"),
            ("num", f"{K['FamOxVOCPct']}% of composition credit"),
            ("text", "duration advantage vs gas lifetime"),
            ("num", f"\u03c1 = {K['LtRho']}"),
            ("text", "short-lived species favouring duration"),
            ("num", f"{K['LtShortWins']} of {K['LtShortN']}"),
        ]),
        ("4  Turning points and error", STAGE_COLOURS[3], [
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


def _save(fig, stem: Path, dpi: int = DPI):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  [saved] {stem.name}.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(PAPER_DEFAULT),
                    help="folder holding main.tex and the two constants files")
    ap.add_argument("--outdir", default="",
                    help="where to write; default <paper>/Figures")
    a = ap.parse_args()
    paper = Path(a.paper).resolve()
    out = Path(a.outdir).resolve() if a.outdir else paper / "Figures"
    K = read_constants(paper)
    figure01(K, out)
    graphical_abstract(K, out)


if __name__ == "__main__":
    main()
