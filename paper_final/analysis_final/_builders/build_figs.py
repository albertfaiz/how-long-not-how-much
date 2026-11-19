"""Assemble paper4_figures_all_in_one.py from paper_figures.py, the Figure 1
script and the path part of _config.py. The figure code is copied verbatim;
only the imports of _config are resolved and the style hooks inserted."""
import re, sys
from pathlib import Path
A = Path(sys.argv[1])
pf = (A / "paper_figures.py").read_text()
f1 = (A / "figure01_methodology_and_graphical_abstract.py").read_text()
cfg = (A / "_config.py").read_text()

def cut(src, start, end):
    a = src.index(start)
    b = src.index(end, a)
    return src[a:b]

# ---------------------------------------------------------------- pieces
titles = cut(pf, "TITLES = {", "FS_TITLE = FS + 1.0")
helpers_figs = cut(pf, "def js(name):", "FIGS = {")
for name in ("DERIVED", "load_table", "RANKING_A", "NCE_TAIL", "NCE_MEAN",
             "FRAME", "dependence_table", "GROUP"):
    helpers_figs = helpers_figs.replace("C." + name, name)
assert "C." not in re.sub(r"[A-Za-z_]C\.", "", helpers_figs), "unresolved C."
helpers_figs = helpers_figs.replace(
    'gpd.read_file(os.path.join(HERE, "conus_counties_5070.gpkg"), layer="county")',
    'gpd.read_file(COUNTY_FILE, layer="county")')
old_div = '["#8A3A00", RED, "#F0B27A", "#F2F2F2", "#8FC7E8", BLUE, "#00355A"]'
assert old_div in helpers_figs
helpers_figs = helpers_figs.replace(old_div, 'PAL["diverging"]')

schem = cut(f1, "def read_constants", "def _save(")
schem = schem.replace(
    '# One colour per stage, so no two stages share a colour (white text on each\n'
    '# passes a contrast check for large text).\n'
    'STAGE_COLOURS = ["#0072B2", "#6A3D9A", "#D55E00", "#009E73"]\n',
    '# The stage colours come from the palette (PALETTES, section 1).\n')
assert "STAGE_COLOURS = " not in schem
schem = schem.replace("STAGE_COLOURS[", 'PAL["stages"][')
reps = [
    ("head_top, head_h = YTOP - 0.12, 1.45", "head_top, head_h = YTOP - 0.12, 1.45 * FONT_SCALE"),
    ("ax.text(x + 0.14, head_top - 0.30, tag, fontsize=7.2,",
     "ax.text(x + 0.14, head_top - 0.30 * FONT_SCALE, tag, fontsize=7.2 * FONT_SCALE,"),
    ("_wrap(fig, ax, title, wbox - 0.28, 7.9)", "_wrap(fig, ax, title, wbox - 0.28, 7.9 * FONT_SCALE)"),
    ("ax.text(x + 0.14, head_top - 0.75 - k * 0.42, ln, fontsize=7.9,",
     "ax.text(x + 0.14, head_top - (0.75 + k * 0.42) * FONT_SCALE, ln, fontsize=7.9 * FONT_SCALE,"),
    ("items, wbox - 0.14, 7.0, 0.16)", "items, wbox - 0.14, 7.0 * FONT_SCALE, 0.16)"),
    ("fs, fsn = 11, 15", "fs, fsn = 11 * FONT_SCALE, 15 * FONT_SCALE"),
    ("fontsize=17, ha=\"center\"", "fontsize=17 * FONT_SCALE, ha=\"center\""),
]
for a, b in reps:
    assert schem.count(a) == 1, a
    schem = schem.replace(a, b)

paths = cut(cfg, "def _find_root() -> Path:", "def _outdir(")
paths = paths.replace("here = Path(__file__).resolve()", "here = HERE / \"_\"")
paths = re.sub(r"\n# A rerun of the analysis.*", "\n", paths, flags=re.S)
tcand = cut(cfg, "def table_candidates(stem: str):", "def load_table(")
ltab = cut(cfg, "def load_table(stem: str, **kw):", "# -----")
dept = cut(cfg, "def dependence_table():", "def census_pairs(")

T = Path(sys.argv[2]).read_text()
out = (T.replace("@@TITLES@@", titles.rstrip() + "\n")
        .replace("@@PATHS@@", paths.rstrip() + "\n")
        .replace("@@TABLES@@", tcand.rstrip() + "\n\n\n" + ltab.rstrip() + "\n\n\n" + dept.rstrip() + "\n")
        .replace("@@SCHEMATIC@@", schem.rstrip() + "\n")
        .replace("@@FIGURES@@", helpers_figs.rstrip() + "\n"))
assert "@@" not in out
Path(sys.argv[3]).write_text(out)
print("written", sys.argv[3], len(out.splitlines()), "lines")
