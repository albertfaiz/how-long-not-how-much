#!/usr/bin/env python3
"""Build index.html for the paper-4 repository from the manuscript source.

Every number on the page is read from constants_*.tex (the macros the paper
uses), every figure caption and every quoted paragraph is converted from
main.tex, so the page cannot disagree with the paper. Explanatory text and the
glossary are written here; they introduce no number of their own.
"""
import os, re, json, html, sys
import tex2html as T

C = T.C


def M(k):
    if k not in C:
        sys.exit(f"missing macro {k}")
    return C[k]


def fill(s):
    """Replace {{Macro}} with its value; fail loudly on an unknown macro."""
    return re.sub(r"\{\{([A-Za-z]+)\}\}", lambda m: M(m.group(1)), s)


CAP = dict(T.captions())


def figure(n, img, alt):
    cap = CAP[img]
    title, rest = cap.split(". ", 1)
    return f'''
<figure class="fig" id="fig{n}">
  <a href="site/figures/{img}" target="_blank" rel="noopener"><img src="site/figures/{img}" alt="{html.escape(alt)}" loading="lazy"></a>
  <figcaption><span class="fignum">Figure {n}.</span> <span class="figtitle">{title}.</span> {rest}</figcaption>
</figure>'''


def paras(start, n=1):
    return "\n".join(f"<p>{p}</p>" for p in T.para(start, n))


def gen_table(label):
    tg = open(os.path.join(T.PAPER, 'tables_generated.tex')).read()
    i = tg.index('\\label{' + label + '}')
    s = tg.rfind('\\begin{table}', 0, i)
    e = tg.index('\\end{table}', i)
    block = tg[s:e]
    cap = block[block.index('\\caption{') + 9:block.index('\\label{')]
    body = block[block.index('\\toprule') + 8:block.index('\\bottomrule')]
    rows = []
    for line in body.split('\\\\'):
        line = line.replace('\\midrule', '').strip()
        if not line:
            continue
        cells = [T.conv(c.strip()) for c in line.split('&')]
        rows.append(cells)
    head, data = rows[0], rows[1:]
    th = "".join(f"<th>{c}</th>" for c in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in data)
    return f'''<div class="tablewrap"><table><caption>{T.conv(cap)}</caption><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'''


def tile(big, label, note=""):
    n = f'<span class="tnote">{note}</span>' if note else ""
    return f'<div class="tile"><span class="tbig">{big}</span><span class="tlabel">{label}</span>{n}</div>'


TITLE = "Predictive Multiplicity in Environmental Health: Information Theory Exposes How Summary Statistics Shape Machine Learning Attribution"
AUTHORS = ("Faiz Ahmad, David J. Lary<sup>*</sup>, Shisir Ruwali, Samyak Shrestha, Adam Aker, "
           "John Waczak, Md Shaid Hasan Niloy, Rittik Patra and Tunde Dare Awoyinka")
REPO = "https://github.com/albertfaiz/how-long-not-how-much"
DOI = "https://doi.org/10.5281/zenodo.23353493"

tiles = "".join([
    tile("{{Nobs}}", "county-years", "{{Ncounties}} counties, {{YearStart}} to {{YearEnd}}"),
    tile("\\(R^2\\) = {{Rtwo}}", "on unseen counties", "95% interval {{RtwoCIlo}} to {{RtwoCIhi}}; mean absolute error {{MAEmodel}} years"),
    tile("1st vs {{HchoConRank}}th", "one formaldehyde field, two summaries", "time above the 75th percentile against the annual mean: {{HchoRatio}}-fold"),
    tile("{{StatNonCentralWins}} of {{StatFieldsComparable}}", "fields where SHAP favours a non-central summary", "{{ExchExpected}} expected if summaries were interchangeable (95% range {{ExchLo}} to {{ExchHi}})"),
    tile("{{MIcentralWins}} of {{StatFieldsComparable}}", "fields where mutual information favours the mean or median", "two-sided sign test \\(p\\) = {{MIsignP}}"),
    tile("{{StatRefitGap}}", "difference in \\(R^2\\)", "central-only against non-central-only models of the same fields"),
    tile("\\(\\rho\\) = {{LtRho}}", "lifetime against the duration advantage", "{{LtSpecies}} species; one-sided permutation \\(p\\) = {{LtPermP}}"),
    tile("&le; {{RegMerTripletMax}}%", "shift of three turning points across the 100th meridian", "minimum dew point, night-time wet-bulb temperature, night cooling efficiency"),
    tile("\\(R^2\\) = {{ResSdohRtwo}}", "socioeconomic columns never used, predicting the model's error", "over half of the residual variance remains unexplained"),
])

idea_svg = '''
<svg class="diagram" viewBox="0 0 760 300" role="img" aria-labelledby="ideaTitle ideaDesc">
  <title id="ideaTitle">One field, several descriptions</title>
  <desc id="ideaDesc">The sub-annual formaldehyde record of a county is reduced in two ways. The time above the national 75th percentile ranks first of 581 inputs; the annual mean of the same record ranks 196th.</desc>
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="arrhead"/></marker></defs>
  <rect x="20" y="95" width="190" height="110" rx="10" class="box"/>
  <text x="115" y="125" class="t1" text-anchor="middle">One field</text>
  <text x="115" y="148" class="t2" text-anchor="middle">CAMS formaldehyde,</text>
  <text x="115" y="166" class="t2" text-anchor="middle">every time step of the year,</text>
  <text x="115" y="184" class="t2" text-anchor="middle">one county</text>
  <path d="M210 130 L300 80" class="ln" marker-end="url(#arr)"/>
  <path d="M210 170 L300 220" class="ln" marker-end="url(#arr)"/>
  <rect x="305" y="40" width="210" height="80" rx="10" class="box acc1"/>
  <text x="410" y="70" class="t1" text-anchor="middle">Duration</text>
  <text x="410" y="92" class="t2" text-anchor="middle">% of the year above the</text>
  <text x="410" y="109" class="t2" text-anchor="middle">national 75th percentile</text>
  <rect x="305" y="180" width="210" height="80" rx="10" class="box acc2"/>
  <text x="410" y="212" class="t1" text-anchor="middle">Concentration</text>
  <text x="410" y="236" class="t2" text-anchor="middle">annual mean</text>
  <path d="M515 80 L590 80" class="ln" marker-end="url(#arr)"/>
  <path d="M515 220 L590 220" class="ln" marker-end="url(#arr)"/>
  <rect x="595" y="45" width="150" height="70" rx="10" class="box"/>
  <text x="670" y="74" class="t1" text-anchor="middle">rank 1 of {{Nfeatpruned}}</text>
  <text x="670" y="96" class="t2" text-anchor="middle">{{HchoDurSHAP}} years</text>
  <rect x="595" y="185" width="150" height="70" rx="10" class="box"/>
  <text x="670" y="214" class="t1" text-anchor="middle">rank {{HchoConRank}} of {{Nfeatpruned}}</text>
  <text x="670" y="236" class="t2" text-anchor="middle">{{HchoConSHAP}} years</text>
  <text x="670" y="150" class="t3" text-anchor="middle">mean |SHAP|, same model,</text>
  <text x="670" y="166" class="t3" text-anchor="middle">same pixels</text>
</svg>'''

css = r'''
:root{--bg:#fbfbf9;--panel:#ffffff;--ink:#1d2433;--muted:#5b6475;--rule:#e3e5ea;--accent:#1f5f99;--accent2:#c2601a;--tint:#eef3f8;--code:#f3f4f6;--maxw:1080px}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11151c;--panel:#171c25;--ink:#e6e9ef;--muted:#a3abba;--rule:#2a313d;--accent:#7fb2e5;--accent2:#f0a36b;--tint:#1b2430;--code:#1e2530}}
:root[data-theme="dark"]{--bg:#11151c;--panel:#171c25;--ink:#e6e9ef;--muted:#a3abba;--rule:#2a313d;--accent:#7fb2e5;--accent2:#f0a36b;--tint:#1b2430;--code:#1e2530}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;font-size:17px;line-height:1.62}
a{color:var(--accent)}
header.hero{background:linear-gradient(135deg,#0f2a44,#1f5f99);color:#fff;padding:56px 16px 40px}
.hero .inner,.wrap{max-width:var(--maxw);margin:0 auto}
.hero h1{font-size:clamp(1.55rem,3.2vw,2.35rem);line-height:1.22;margin:0 0 16px;font-weight:600}
.hero .authors{opacity:.95;margin:0 0 4px}
.hero .aff{opacity:.8;font-size:.92rem;margin:0 0 18px}
.hero .status{display:inline-block;border:1px solid rgba(255,255,255,.45);border-radius:999px;padding:3px 12px;font-size:.85rem;margin-bottom:20px}
.btns{display:flex;flex-wrap:wrap;gap:10px}
.btn{display:inline-block;padding:9px 16px;border-radius:8px;background:rgba(255,255,255,.14);color:#fff;text-decoration:none;border:1px solid rgba(255,255,255,.35);font-size:.95rem}
.btn:hover{background:rgba(255,255,255,.26)}
nav.toc{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--rule)}
nav.toc .wrap{display:flex;gap:18px;overflow-x:auto;padding:10px 16px;white-space:nowrap;font-size:.92rem}
nav.toc a{text-decoration:none;color:var(--muted)}
nav.toc a:hover{color:var(--accent)}
main .wrap{padding:0 16px}
section{padding:34px 0 10px;border-bottom:1px solid var(--rule)}
h2{font-size:1.55rem;margin:0 0 14px;font-weight:600}
h3{font-size:1.15rem;margin:26px 0 8px;font-weight:600}
p{margin:0 0 14px}
.lede{font-size:1.06rem;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0}
@media (max-width:860px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:520px){.grid{grid-template-columns:1fr}}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:4px}
.tbig{font-size:1.5rem;color:var(--accent);font-weight:600}
.tlabel{font-size:.95rem}
.tnote{font-size:.82rem;color:var(--muted)}
.fig{margin:22px 0;background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:12px}
.fig img{width:100%;height:auto;display:block;border-radius:6px;background:#fff}
figcaption{font-size:.9rem;color:var(--muted);margin-top:10px}
.fignum{color:var(--ink);font-weight:600}
.box-note{background:var(--tint);border-left:4px solid var(--accent);border-radius:8px;padding:12px 16px;margin:16px 0}
.box-note.warn{border-left-color:var(--accent2)}
.box-note h4{margin:0 0 6px;font-size:1rem;font-weight:600}
.eq{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:8px 14px;margin:12px 0}
.eqlabel{font-size:.82rem;color:var(--muted)}
.tablewrap{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem;background:var(--panel)}
caption{caption-side:top;text-align:left;color:var(--muted);font-size:.88rem;padding:0 0 8px}
th,td{border-bottom:1px solid var(--rule);padding:6px 10px;text-align:left;vertical-align:top}
th{font-weight:600}
pre{background:var(--code);border-radius:8px;padding:12px;overflow-x:auto;font-size:.86rem}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}
pre code{overflow-wrap:normal}
dl.gloss dt{font-weight:600;margin-top:12px}
dl.gloss dd{margin:2px 0 0 0;color:var(--ink)}
.diagram{width:100%;height:auto;max-width:760px;display:block;margin:10px auto}
.diagram .box{fill:var(--panel);stroke:var(--rule);stroke-width:1.5}
.diagram .acc1{stroke:var(--accent2)}
.diagram .acc2{stroke:var(--accent)}
.diagram .ln{stroke:var(--muted);stroke-width:1.6;fill:none}
.diagram .arrhead{fill:var(--muted)}
.diagram .t1{fill:var(--ink);font-size:15px;font-weight:600}
.diagram .t2{fill:var(--muted);font-size:12.5px}
.diagram .t3{fill:var(--muted);font-size:11.5px}
footer{padding:30px 16px 50px;color:var(--muted);font-size:.88rem}
.small{font-size:.88rem;color:var(--muted)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.two>div,.grid>div{min-width:0}
mjx-container[display="true"]{max-width:100%;overflow-x:auto;overflow-y:hidden}
@media (max-width:760px){.two{grid-template-columns:1fr}body{font-size:16px}}
'''

body = f'''
<header class="hero"><div class="inner">
  <span class="status">Manuscript submitted to <em>Sensors</em> (MDPI), 2026</span>
  <h1>{TITLE}</h1>
  <p class="authors">{AUTHORS}</p>
  <p class="aff">Department of Physics, The University of Texas at Dallas, Richardson, Texas, USA. <sup>*</sup>Correspondence: david.lary@utdallas.edu</p>
  <div class="btns">
    <a class="btn" href="{REPO}">Code and results (GitHub)</a>
    <a class="btn" href="{DOI}">Data (Zenodo)</a>
    <a class="btn" href="#reproduce">Reproduce</a>
    <a class="btn" href="#cite">Cite</a>
  </div>
</div></header>

<nav class="toc" aria-label="Sections"><div class="wrap">
  <a href="#glance">At a glance</a><a href="#abstract">Abstract</a><a href="#idea">The idea</a><a href="#design">Data and design</a>
  <a href="#methods">Methods and equations</a><a href="#results">Results</a><a href="#survives">What survives</a>
  <a href="#practice">Practice and limits</a><a href="#reproduce">Reproduce</a><a href="#glossary">Glossary</a><a href="#cite">Cite</a>
</div></nav>

<main><div class="wrap">

<section id="glance">
  <h2>At a glance</h2>
  <p class="lede">A machine learning model of United States county life expectancy, built only from satellite and gridded environmental data, is used to ask a question about method: when the model ranks its inputs, how much of the ranking reflects the environment, and how much reflects the summary statistic chosen to describe each field?</p>
  <div class="grid">{fill(tiles)}</div>
  <p class="small">Every number on this page is read from the files that generate the manuscript's numbers (<code>paper_final/constants_*.tex</code>), and every figure caption is taken from the manuscript.</p>
  <figure class="fig"><img src="site/figures/GraphicalAbstract.png" alt="Graphical abstract of the study" loading="lazy"></figure>
</section>

<section id="abstract">
  <h2>Abstract</h2>
  <p>{T.abstract()}</p>
</section>

<section id="idea">
  <h2>The idea in one picture</h2>
  <p>A satellite or reanalysis product describes each county with many values per year. A model needs one number per county and year, so the record is summarised, usually by its annual mean. The same record can also be summarised by a percentile, a spread or the share of time spent above a reference level. These are different descriptions of one measurement, not different measurements.</p>
  {fill(idea_svg)}
  <p>For one formaldehyde field, the two descriptions land at opposite ends of the model's ranking. The paper asks whether this is a general property of attribution, whether the information in the data follows the same pattern, and what in the fitted model is robust to the choice.</p>
</section>

<section id="design">
  <h2>Data and design</h2>
  {paras("We linked satellite-derived and gridded environmental measurements", 2)}
  {figure(1, "Figure01_methodology.png", "Study design in four stages")}
  <h3>Outcome</h3>
  {paras("The outcome is county life expectancy at birth", 1)}
  <h3>Two descriptions of one field</h3>
  {paras("Duration is not a rescaled mean", 1)}
</section>

<section id="methods">
  <h2>Methods and equations</h2>
  <p class="lede">The equations below are those of the manuscript (equations 1, 2 and A1) and the standard definitions of the quantities it reports. Symbols are defined beneath each one.</p>

  <h3>Duration of a species above a national reference level</h3>
  <div class="eq">\\[ D^{{(K)}}_{{c,i,t}} = \\frac{{100}}{{S}}\\sum_{{s=1}}^{{S}}\\mathbf{{1}}\\!\\left[x_{{c,i,t}}(s) > q_c^{{(K)}}\\right] \\qquad (1) \\]</div>
  <p>\\(x_{{c,i,t}}(s)\\) is the value of species \\(c\\) in county \\(i\\), year \\(t\\), at time step \\(s\\) of the \\(S\\) reanalysis time steps of the year; \\(q_c^{{(K)}}\\) is the \\(K\\)th percentile of that species over all counties, time steps and years; \\(\\mathbf{{1}}[\\cdot]\\) is one when its condition holds. \\(D\\) is in per cent of the year and was computed at \\(K\\) = 75 and 90. The concentration description is the annual mean, \\(\\bar x_{{c,i,t}} = S^{{-1}}\\sum_s x_{{c,i,t}}(s)\\).</p>

  <h3>Night cooling efficiency, two constructions</h3>
  <div class="eq">\\[ \\text{{NCE}}_{{\\text{{tail}}}} = \\frac{{\\text{{LST}}^{{\\text{{day}}}}_{{p90}} - \\text{{LST}}^{{\\text{{night}}}}_{{p10}}}}{{\\text{{LST}}^{{\\text{{day}}}}_{{p90}}}}, \\qquad \\text{{NCE}}_{{\\text{{mean}}}} = \\frac{{\\text{{LST}}^{{\\text{{day}}}}_{{\\mu}} - \\text{{LST}}^{{\\text{{night}}}}_{{\\mu}}}}{{\\text{{LST}}^{{\\text{{day}}}}_{{\\mu}}}} \\qquad (2) \\]</div>
  <p>LST is MODIS land-surface temperature in degrees Celsius; \\(p90\\) and \\(p10\\) are the 90th and 10th percentiles of the county's pixels and \\(\\mu\\) their mean. NCE is an index, not a fraction of heat. The two constructions correlate at {M("NCEcorr")} over all county-years.</p>

  <h3>Night-time wet-bulb temperature</h3>
  <p>Wet-bulb temperature is computed with the empirical formula of Stull (2011), from air temperature \\(T\\) (°C) and relative humidity \\(RH\\) (%):</p>
  <div class="eq">\\[ T_w = T\\arctan\\!\\left[0.151977\\,(RH + 8.313659)^{{1/2}}\\right] + \\arctan(T + RH) - \\arctan(RH - 1.676331) + 0.00391838\\,RH^{{3/2}}\\arctan(0.023101\\,RH) - 4.686035 \\]</div>
  <p>For the night-time form, \\(T\\) is the annual-mean daily minimum temperature and the humidity is that of the minimum dew point \\(T_d\\), with \\(RH = 100\\,\\exp\\!\\left(\\tfrac{{a T_d}}{{b + T_d}}\\right)\\big/\\exp\\!\\left(\\tfrac{{a T}}{{b + T}}\\right)\\), \\(a\\) = 17.625 and \\(b\\) = 243.04 °C (as implemented in <code>engineered_features_v2.py</code>). The minimum dew point of a county is the annual-mean dew point of its driest 4 km PRISM cell, not the lowest daily value of the year.</p>

  <h3>Model and skill</h3>
  <p>The model is a gradient-boosted tree ensemble (XGBoost; Chen and Guestrin, 2016): a sum of \\(K\\) regression trees fitted in sequence, each to the errors of those before it,</p>
  <div class="eq">\\[ \\hat y_i = \\sum_{{k=1}}^{{K}} f_k(\\mathbf{{x}}_i), \\qquad \\mathcal{{L}} = \\sum_i \\ell(y_i, \\hat y_i) + \\sum_k \\Omega(f_k),\\quad \\Omega(f) = \\gamma T_f + \\tfrac{{1}}{{2}}\\lambda \\lVert w \\rVert^2 \\]</div>
  <p>where \\(T_f\\) is the number of leaves of tree \\(f\\) and \\(w\\) its leaf values (the standard regularised objective of XGBoost). Skill is measured on counties held out by eight-fold county-grouped cross-validation:</p>
  <div class="eq">\\[ R^2 = 1 - \\frac{{\\sum_i (y_i - \\hat y_i)^2}}{{\\sum_i (y_i - \\bar y)^2}}, \\qquad \\text{{MAE}} = \\frac{{1}}{{n}}\\sum_i \\lvert y_i - \\hat y_i \\rvert, \\qquad r_i = y_i - \\hat y_i \\]</div>
  <p>with \\(r_i\\) the residual, observed minus predicted life expectancy. The settings were fixed before this study:</p>
  {gen_table("tab:learner")}

  <h3>Attribution: Shapley additive explanations (SHAP)</h3>
  <p>The attribution \\(\\phi_j\\) of input \\(j\\) to one prediction is its Shapley value over the set of inputs \\(N\\):</p>
  <div class="eq">\\[ \\phi_j = \\sum_{{S \\subseteq N \\setminus \\{{j\\}}}} \\frac{{|S|!\\,(|N| - |S| - 1)!}}{{|N|!}}\\,\\bigl[v(S \\cup \\{{j\\}}) - v(S)\\bigr], \\qquad v_{{\\text{{int}}}}(S) = \\mathbb{{E}}_{{X_{{\\bar S}}}}\\!\\left[f(x_S, X_{{\\bar S}})\\right] \\qquad (\\text{{A}}1) \\]</div>
  <p>In the interventional form \\(v_{{\\text{{int}}}}\\), the absent inputs \\(X_{{\\bar S}}\\) are drawn from a background sample independently of the present ones, so an input is credited with what the model does with it. The attributions add up to the prediction:</p>
  <div class="eq">\\[ f(\\mathbf{{x}}) = \\mathbb{{E}}[f(X)] + \\sum_{{j \\in N}} \\phi_j(\\mathbf{{x}}), \\qquad \\mathbb{{E}}[f(X)] = {M("PredMean")}\\ \\text{{years}} \\]</div>
  {paras("The main attribution is SHapley Additive exPlanations", 1)}

  <h3>Mutual information: what the data hold without a model</h3>
  <p>Mutual information measures how much knowing one variable reduces uncertainty about another, in nats:</p>
  <div class="eq">\\[ I(X;Y) = \\iint p(x,y)\\,\\ln\\frac{{p(x,y)}}{{p(x)\\,p(y)}}\\,dx\\,dy \\;=\\; H(Y) - H(Y \\mid X), \\qquad H(Y) = -\\int p(y)\\ln p(y)\\,dy \\]</div>
  <p>It is zero when the two are independent and grows with any dependence, linear or not. It was estimated with the \\(k\\)-nearest-neighbour estimator of Kraskov, Stögbauer and Grassberger (\\(k\\) = {M("MIneighbours")}),</p>
  <div class="eq">\\[ \\hat I(X;Y) = \\psi(k) + \\psi(n) - \\bigl\\langle \\psi(n_x + 1) + \\psi(n_y + 1) \\bigr\\rangle \\]</div>
  <p>where \\(\\psi\\) is the digamma function, \\(n\\) the number of observations, and \\(n_x\\), \\(n_y\\) the numbers of points within each observation's \\(k\\)th-neighbour distance in the \\(x\\) and \\(y\\) directions. Mutual information is marginal: it scores a column alone, before any model.</p>

  <h3>The census and its reference distribution</h3>
  {paras("For each comparable field the census records", 1)}
  <div class="eq">\\[ p_i = \\frac{{n_i^{{\\text{{nc}}}}}}{{n_i}}, \\qquad T = \\sum_{{i=1}}^{{{M("StatFieldsComparable")}}} B_i,\\ \\ B_i \\sim \\text{{Bernoulli}}(p_i), \\qquad \\mathbb{{E}}[T] = \\sum_i p_i = {M("ExchExpected")} \\]</div>
  <p>\\(n_i\\) is the number of admitted summaries of field \\(i\\) and \\(n_i^{{\\text{{nc}}}}\\) the number that are non-central. \\(T\\) follows a Poisson-binomial distribution, computed exactly; its central 95% range is {M("ExchLo")} to {M("ExchHi")}. For the mutual-information comparison of a fixed pair, the reference is a fair coin, \\(B \\sim \\text{{Binomial}}(n, \\tfrac{{1}}{{2}})\\), tested two-sided: \\(p = \\min\\{{1,\\ 2\\min[P(B \\ge k),\\,P(B \\le k)]\\}}\\).</p>
  {paras("Attribution divides credit, and a gap between two correlated columns", 1)}

  <h3>Atmospheric lifetime</h3>
  {paras("If duration carries information that concentration does not, one candidate", 1)}
  <div class="box-note"><h4>Lifetime, not half-life</h4><p>The atmospheric (here tropospheric) lifetime \\(\\tau\\) of a species is the standard quantity of atmospheric chemistry: its burden divided by its removal rate, the e-folding time of its decay when sources stop. For removal by the hydroxyl radical, \\(\\tau = 1/(k_{{\\text{{OH}}}}[\\text{{OH}}])\\). A half-life, the term of radioactive decay, is \\(t_{{1/2}} = \\tau\\ln 2 \\approx 0.69\\,\\tau\\). The paper uses lifetimes from the literature (Seinfeld and Pandis; Atkinson and Arey; and others cited there).</p></div>
  <p>The test statistic is Spearman's rank correlation, the Pearson correlation of ranks, \\(\\rho = \\text{{corr}}\\bigl(\\text{{rank}}(\\tau_c), \\text{{rank}}(R_c)\\bigr)\\), between the lifetime \\(\\tau_c\\) and the median duration-to-concentration importance ratio \\(R_c\\) of species \\(c\\). Its one-sided \\(p\\) value is the share of {M("LtPerms")} random permutations of the ratios that give a correlation at least as negative as the observed one.</p>

  <h3>Turning points and the stability rule</h3>
  {paras("A turning point is the value of a variable at which its contribution", 2)}
  <div class="eq">\\[ x^{{*}}: \\ \\tilde m(x^{{*}}) = 0, \\qquad \\Pr_{{\\text{{boot}}}}[\\text{{crossing found}}] \\ge 0.60, \\qquad \\frac{{q_{{97.5}}(x^{{*}}) - q_{{2.5}}(x^{{*}})}}{{\\max x - \\min x}} < 0.40, \\qquad x^{{*}}_{{\\text{{full}}}} \\in [q_{{2.5}}, q_{{97.5}}] \\]</div>
  <p>\\(\\tilde m(x)\\) is the median SHAP value in 45 equal-count bins of the variable between its 1st and 99th percentiles, and the quantiles are taken over {M("TpBoot")} bootstrap resamples of the county-years.</p>

  <h3>Residual geography and the socioeconomic audit</h3>
  <div class="eq">\\[ I = \\frac{{n}}{{\\sum_{{i\\ne j}} w_{{ij}}}}\\;\\frac{{\\sum_{{i\\ne j}} w_{{ij}}\\,(\\bar r_i - \\bar r)(\\bar r_j - \\bar r)}}{{\\sum_i (\\bar r_i - \\bar r)^2}} \\]</div>
  <p>Here \\(\\bar r_i\\) is the out-of-fold residual averaged within county \\(i\\), \\(\\bar r\\) its average over counties, and \\(w_{{ij}}\\) the spatial weights of the four-nearest-neighbour graph. For the six pre-declared indicators, the Holm-adjusted \\(p\\) value of the \\(k\\)th smallest of \\(m\\) = 6 is \\(\\tilde p_{{(k)}} = \\max_{{j \\le k}} \\min\\{{1,\\,(m - j + 1)\\,p_{{(j)}}\\}}\\).</p>
  {paras("We averaged the out-of-fold residual (observed minus predicted life", 1)}
</section>

<section id="results">
  <h2>Results</h2>
  <h3>A model worth interrogating</h3>
  {paras("From the physical environment alone, the model predicted", 2)}
  {figure(2, "Figure02_performance.png", "Predictive performance of the environment-only model")}
  <h3>One field, two descriptions</h3>
  {paras("The leading variable of the fitted model is the share of the year", 2)}
  {figure(3, "Figure03_two_descriptions.png", "Attribution of two descriptions of one field")}
  {figure(4, "Figure04_drop_one_refit.png", "Drop-one refit of night cooling efficiency")}
  <h3>Attribution follows the description; information does not</h3>
  {paras("Two cases do not make a general effect", 4)}
  {figure(5, "Figure05_rashomon_census.png", "Attribution, information and skill across summaries")}
  <h3>A stable pattern, not an accident of the analysis</h3>
  {paras("Four decomposition instruments and total split gain gave counts", 4)}
  {figure(6, "Figure06_invariance.png", "Stability of the census")}
  {gen_table("tab:instruments")}
</section>

<section id="survives">
  <h2>What survives</h2>
  <h3>Atmospheric lifetime broadly tracks the duration advantage</h3>
  {paras("Tropospheric lifetime broadly tracks whether", 1)}
  {figure(7, "Figure07_lifetime.png", "Atmospheric lifetime and the duration advantage")}
  {gen_table("tab:species")}
  <h3>Turning points that survive noise</h3>
  {paras("passed the stability criterion", 2)}
  <div class="box-note warn"><h4>What a turning point is, and is not</h4><p>A turning point is where a variable's contribution to the prediction changes sign around the model's average prediction ({M("PredMean")} years). It describes the fitted model. It is not a threshold of harm, and the paper does not interpret it as a causal or physiological limit.</p></div>
  {figure(8, "Figure08_turning_points.png", "Stability of the located turning points")}
  {gen_table("tab:turning")}
  <h3>Turning points that barely move with climate</h3>
  {paras("A turning point that reflected regional climate would move", 3)}
  {figure(9, "Figure09_climate_invariance.png", "Night-time turning points across climate regions")}
  <h3>Where a physical model is blind</h3>
  {paras("The model's error is geographically structured", 2)}
  {figure(10, "Figure10_failure_geography.png", "Geography of model error")}
  <p class="small">The data behind panel C (3,077 counties: mean out-of-fold residual and the six indicators) are in <code>paper_final/derived/figure10c_residual_and_indicators.csv</code>; script 33 reproduces every bar. The indicators are columns of the archived data table and never entered the model.</p>
  <h3>Conclusions</h3>
  {paras("We have shown that machine learning attribution rankings", 1)}
</section>

<section id="practice">
  <h2>Recommendations and limitations</h2>
  {paras("Four practices follow for studies that use attribution", 1)}
  <h3>Limitations</h3>
  {paras("The unit of analysis is the county", 5)}
</section>

<section id="reproduce">
  <h2>Reproduce</h2>
  <div class="two">
  <div>
  <h3>Get the code and data</h3>
  <pre><code>git clone {REPO}.git
cd how-long-not-how-much
# from Zenodo ({DOI}):
#   Final_ML_Ready_Dataset.csv.zip  -> unzip at the repository root
#   studyA_reference_frame.parquet  -> results_studyA_20260801_004313/</code></pre>
  <h3>Run</h3>
  <pre><code>conda env create -f environment.yml
conda activate satsdoh
bash run_all.sh   # scripts 01 to 34, constants, figures</code></pre>
  </div>
  <div>
  <h3>Library versions</h3>
  <p>The production model was fitted on the Juno cluster of the University of Texas at Dallas in Python 3.10.12. The analysis battery ran in Python {M("VerPython")} with XGBoost {M("VerXgb")}, SHAP {M("VerShap")}, scikit-learn {M("VerSklearn")}, NumPy {M("VerNumpy")}, pandas {M("VerPandas")}, SciPy {M("VerScipy")} and CatBoost {M("VerCatboost")}. Every script is seeded. An earlier run under XGBoost {M("VerXgbRef")} agreed except in the state hold-out \\(R^2\\) ({M("VerHoldRef")} against {M("HoldStateRtwo")}), Moran's \\(I\\) ({M("VerMoranRef")} against {M("ResMoranI")}), the perturbation shifts (second decimal place) and the permutation counts (up to {M("VerPermShift")} fields in a split). No conclusion changed.</p>
  <h3>Built-in checks</h3>
  <p>Script 01 confirms the fitted model is identical across thread counts; script 10 reruns a script and compares its outputs by hash; script 32 checks the data files against each other cell by cell; <code>build_paper_constants.py</code> tests every sentence that describes a number in words against the regenerated value.</p>
  </div>
  </div>
</section>

<section id="glossary">
  <h2>Glossary</h2>
  <p class="small">Standard terms are given their usual meaning. Terms defined in this study are marked (this study).</p>
  <dl class="gloss">
  <dt>Atmospheric (tropospheric) lifetime</dt><dd>Burden of a species divided by its removal rate; the e-folding time of its decay. Not the same as a half-life, which is \\(\\tau \\ln 2\\).</dd>
  <dt>Reanalysis</dt><dd>A gridded reconstruction of past weather or composition from a numerical model corrected by observations (here ERA5 and the CAMS reanalysis).</dd>
  <dt>Chemical transport model</dt><dd>A simulation of how gases and particles are emitted, carried, transformed and removed.</dd>
  <dt>Aerosol optical depth</dt><dd>How much the particles in a vertical column dim sunlight; dimensionless.</dd>
  <dt>PM<sub>2.5</sub></dt><dd>Particulate matter with aerodynamic diameter below 2.5 micrometres.</dd>
  <dt>Land-surface temperature (LST)</dt><dd>The radiative temperature of the ground surface retrieved from thermal infrared, distinct from air temperature.</dd>
  <dt>Dew point</dt><dd>The temperature to which air must cool at constant pressure and moisture to become saturated; a measure of the water the air holds.</dd>
  <dt>Wet-bulb temperature</dt><dd>The lowest temperature evaporation can cool a wet surface to; it combines heat and humidity.</dd>
  <dt>Duration, \\(D\\) (this study)</dt><dd>Share of the year, in per cent, that a species spends above a fixed national percentile (equation 1).</dd>
  <dt>Night cooling efficiency, NCE</dt><dd>Day-to-night drop in land-surface temperature as a fraction of the daytime value (equation 2); introduced in the earlier nocturnal study and used here in two constructions.</dd>
  <dt>Central and non-central summary (this study)</dt><dd>Central: the mean or the median of a field. Non-central: a quantile, the minimum or maximum, or a measure of spread.</dd>
  <dt>Census (this study)</dt><dd>The field-by-field comparison, over all 47 comparable fields, of the best central and the best non-central summary.</dd>
  <dt>Coefficient of determination, \\(R^2\\)</dt><dd>Share of the variation in the outcome that the predictions reproduce, measured here on held-out counties.</dd>
  <dt>County-grouped cross-validation</dt><dd>Folds formed so that no county appears in both training and test data.</dd>
  <dt>SHAP value</dt><dd>The Shapley value of an input for one prediction; the values of all inputs add up to the prediction minus the average prediction.</dd>
  <dt>Interventional and path-dependent TreeSHAP</dt><dd>Two ways of filling absent inputs: independently from a background sample, or by following the tree structure, which approximates conditioning on the present inputs.</dd>
  <dt>Permutation importance</dt><dd>The loss of held-out skill when one column is shuffled while all others stay intact.</dd>
  <dt>Mutual information</dt><dd>Reduction in uncertainty (entropy) about one variable from knowing another; zero for independent variables; here in nats.</dd>
  <dt>Predictive multiplicity and the Rashomon set</dt><dd>Many models fit the same data almost equally well yet attribute differently; the set of such near-optimal models is the Rashomon set (Breiman, 2001; Marx et al., 2020; Fisher et al., 2019).</dd>
  <dt>Spearman rank correlation, \\(\\rho\\)</dt><dd>Pearson correlation of the ranks; measures a monotone association and is less sensitive to extreme values than the Pearson correlation.</dd>
  <dt>Permutation test</dt><dd>A \\(p\\) value from the share of random relabellings that give a statistic at least as extreme as the observed one.</dd>
  <dt>Bootstrap</dt><dd>Resampling the observations with replacement to measure how much an estimate would vary.</dd>
  <dt>Sign test</dt><dd>A test of whether one of two outcomes occurs more often than half the time, against a binomial distribution with probability one half.</dd>
  <dt>Poisson-binomial distribution</dt><dd>The distribution of the number of successes in independent trials with different probabilities.</dd>
  <dt>Moran's \\(I\\)</dt><dd>A measure of spatial autocorrelation: positive when neighbouring values are alike.</dd>
  <dt>Holm procedure</dt><dd>A step-down correction of \\(p\\) values for testing several hypotheses at once.</dd>
  <dt>Turning point (this study)</dt><dd>The value of a variable at which its SHAP contribution changes sign; a property of the fitted model.</dd>
  </dl>
</section>

<section id="cite">
  <h2>Cite</h2>
  <pre><code>@article{{Ahmad2026multiplicity,
  author  = {{Ahmad, Faiz and Lary, David J. and Ruwali, Shisir and Shrestha, Samyak and
             Aker, Adam and Waczak, John and Niloy, Md Shaid Hasan and Patra, Rittik and
             Awoyinka, Tunde Dare}},
  title   = {{Predictive Multiplicity in Environmental Health: Information Theory Exposes
             How Summary Statistics Shape Machine Learning Attribution}},
  journal = {{Sensors}},
  year    = {{2026}},
  note    = {{Submitted}}
}}</code></pre>
  <p>Data: Zenodo, <a href="{DOI}">{DOI}</a>. Code: MIT licence. Data: CC BY 4.0.</p>
  <h3>This work builds on</h3>
  <ul>
    <li>Ahmad, F. et al. Nighttime Thermal Patterns and County Life Expectancy. <em>Remote Sensing</em> 2026, 18, 2330. <a href="https://doi.org/10.3390/rs18142330">doi:10.3390/rs18142330</a></li>
    <li>Shrestha, S. et al. The External Exposome and Life Expectancy: Formaldehyde as a Leading Predictor in U.S. Counties. <em>Air</em> 2026, 4, 10. <a href="https://doi.org/10.3390/air4020010">doi:10.3390/air4020010</a></li>
    <li>Shrestha, S. et al. Atmospheric Exposures and Cardiovascular Mortality in United States Counties. <em>AI Sensors</em> 2026, 2, 8. <a href="https://doi.org/10.3390/aisens2030008">doi:10.3390/aisens2030008</a></li>
    <li>Ruwali, S. et al. Estimating Life Expectancy in the U.S.: Identifying Environmental Determinants and Disparities (2003 to 2019). <em>Sensors</em> 2026, 26, 6478. <a href="https://doi.org/10.3390/s26206478">doi:10.3390/s26206478</a></li>
  </ul>
</section>

</div></main>
<footer><div class="wrap">Department of Physics, The University of Texas at Dallas. Page generated from the manuscript source by <code>site/build_site.py</code>; numbers from <code>paper_final/constants_*.tex</code>. All results are county-level associations from a fitted model.</div></footer>
'''

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictive Multiplicity in Environmental Health</title>
<meta name="description" content="Code, data and results for a study of how the summary statistic of a satellite or reanalysis field shapes machine learning attribution in a model of United States county life expectancy.">
<style>{css}</style>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }}, chtml: {{ scale: 0.98 }} }};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
</head>
<body>
{body}
</body>
</html>
'''
OUT = os.environ.get('SITE_OUT', os.path.join(T.HERE, '..', 'index.html'))
open(OUT, 'w').write(page)
print('written', len(page))
