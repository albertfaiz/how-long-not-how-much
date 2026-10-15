"""Convert the manuscript source of paper 4 into HTML fragments for the
project website (index.html at the repository root).

The paper folder holds main.tex, constants_base.tex, constants_results.tex
and tables_generated.tex. It defaults to ../paper_final next to this
folder; set SITE_PAPER_DIR to point elsewhere. Every number on the page is
read from the constants files, so the page cannot disagree with the paper.
"""
import os, re, json
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.environ.get("SITE_PAPER_DIR", os.path.join(HERE, "..", "paper_final"))


def _tex_value_to_html(v):
    v = v.replace("{,}", ",")
    v = re.sub(r"\$-\$", "\u2212", v)
    def sci(m):
        lt = "< " if m.group(1) else ""
        exp = m.group(3).replace("-", "\u2212")
        return f"{lt}{m.group(2)} \u00d7 10<sup>{exp}</sup>"
    v = re.sub(r"^\$(<)?\s*([0-9.]+)\\times10\^\{([-0-9]+)\}\$$", sci, v)
    return v


def load_macros():
    d = {}
    for f in ("constants_base.tex", "constants_results.tex"):
        txt = open(os.path.join(PAPER, f)).read()
        for m in re.finditer(r"\\(?:re)?newcommand\{\\(\w+)\}\{(.*)\}\s*$", txt, re.M):
            d[m.group(1)] = _tex_value_to_html(m.group(2))
    return d


C = load_macros()
TEX = open(os.path.join(PAPER, "main.tex")).read()
TEX_nc=re.sub(r'(?<!\\)%.*','',TEX)
# label numbering
labels={}
fig=0;tab=0
for m in re.finditer(r'\\begin\{figure\}(.*?)\\end\{figure\}',TEX_nc,re.S):
    fig+=1
    for l in re.findall(r'\\label\{([^}]*)\}',m.group(1)): labels[l]=str(fig)
TG=open(os.path.join(PAPER, 'tables_generated.tex')).read()
# tables: main text table(s) first, then generated appendix tables
for i,l in enumerate(re.findall(r'\\label\{(tab:[^}]*)\}',TG)): labels[l]=f'A{i+1}'
s=0;ss=0;app=False;letter=64
for line in TEX_nc.splitlines():
    if line.startswith('\\appendixstart'): app=True
    m=re.match(r'\\(sub)?section\{(.*)\}',line)
    if m:
        if m.group(1): ss+=1; cur=(f'{s}.{ss}' if not app else f'{chr(letter)}.{ss}')
        else:
            if app: letter+=1; ss=0; cur=chr(letter)
            else: s+=1; ss=0; cur=str(s)
    m2=re.search(r'\\label\{((?:sec|ssec|app)[^}]*)\}',line)
    if m2: labels[m2.group(1)]=cur
def conv(x):
    x=re.sub(r'(?<!\\)%.*','',x)
    x=re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}','',x,flags=re.S)
    x=re.sub(r'\\the\\numexpr\\([A-Za-z]+)-\\([A-Za-z]+)\\relax\{\}',lambda m:str(int(C[m.group(1)].replace(',',''))-int(C[m.group(2)].replace(',',''))),x)
    x=re.sub(r'\\the\\numexpr\\([A-Za-z]+)-1\\relax\{\}',lambda m:str(int(C[m.group(1)])-1),x)
    def mac(m):
        k=m.group(1)
        if k in C: return C[k]
        raise KeyError(k)
    x=re.sub(r'\\label\{[^}]*\}','',x)
    x=re.sub(r'~?\\cite[pt]?\{[^}]*\}','',x)
    x=re.sub(r'(Figure|Table|Section|Appendix)~\\ref\{([^}]*)\}((?:[A-D](?:,[A-D])*)?)',lambda m:(f'Section {labels.get(m.group(2),"?")} of the paper' if m.group(1)=='Section' else f'{m.group(1)} {labels.get(m.group(2),"?")}{m.group(3)}'),x)
    x=re.sub(r'\\ref\{([^}]*)\}',lambda m:labels.get(m.group(1),'?'),x)
    x=re.sub(r'\\,?\\textdegree(\{\})?\s*C',' °C',x); x=x.replace('\\textdegree{}','°').replace('\\textdegree','°').replace('\\,','\u2009').replace('~',' ')
    x=x.replace('\\%','%').replace('---','—').replace('--','–').replace("``",'“').replace("''",'”')
    x=re.sub(r'\\(?:emph|textit)\{([^}]*)\}',r'<em>\1</em>',x)
    x=re.sub(r'\\textbf\{([^}]*)\}',r'\1',x)
    # math: protect, substitute macros inside too
    def math(m):
        inner=re.sub(r'\\([A-Za-z]+)\{\}',lambda mm: C[mm.group(1)] if mm.group(1) in C else mm.group(0),m.group(1))
        inner=inner.replace('\u2212','-').replace('<sup>','^{').replace('</sup>','}').replace(' × ','\\times ')
        return '\\('+inner+'\\)'
    x=re.sub(r'\$([^$]+)\$',math,x)
    x=re.sub(r'\\([A-Za-z]+)\{\}',mac,x)
    x=re.sub(r'\s+',' ',x).strip()
    return x
def captions():
    out=[]
    for m in re.finditer(r'\\begin\{figure\}(.*?)\\end\{figure\}',TEX_nc,re.S):
        body=m.group(1)
        img=re.search(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}',body).group(1)
        i=body.index('\\caption{')+9; depth=1; j=i
        while depth:
            if body[j]=='{': depth+=1
            elif body[j]=='}': depth-=1
            j+=1
        out.append((img.split('/')[-1],conv(body[i:j-1])))
    return out
if __name__=='__main__':
    for img,c in captions(): print(img,'|',c[:200],'\n')
    print({k:v for k,v in labels.items() if k.startswith(('ssec','sec','tab'))})

def para(start, n=1):
    """Return n consecutive paragraphs of main.tex starting at the one containing `start`."""
    body=TEX_nc[TEX_nc.index('\\begin{document}'):]
    paras=re.split(r'\n\s*\n',body)
    for i,p in enumerate(paras):
        if start in re.sub(r'\s+',' ',p):
            sel=paras[i:i+n]
            return [conv(re.sub(r'\\(sub)?section\{[^}]*\}','',x)) for x in sel]
    raise KeyError(start)
def abstract():
    a=TEX[TEX.index('\\abstract{%')+11:]
    a=a[:a.index('\n}\n')]
    return conv(a)
