"""Visual layer for the Streamlit app: theme CSS, HTML components and Plotly styling.

Design idea ("ChainSight"): a block-explorer look with a floating top bar, a 3-D hero and live effects. Near-black charcoal, Bitcoin orange
and gold, monospace ledger numerals, hexagon badges and a "chain" of time-step blocks.
Colour semantics: emerald = licit / healthy, red = illicit / failing, gold = caution,
orange = brand / neutral emphasis.
"""
import streamlit as st

INK, PANEL, PANEL2, BORDER = "#0b0b0e", "#131318", "#1a1a21", "#2a2a35"
TEXT, MUTED = "#f2f2f5", "#9a9aa8"
BTC, BTC_D, GOLD = "#f7931a", "#c46f08", "#ffc857"
LICIT, LICIT_D, ILLICIT, ILLICIT_L = "#34d399", "#0f8f65", "#ff4757", "#ff8a95"
SKY = "#38bdf8"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Unbounded', sans-serif !important; letter-spacing: -0.03em; font-weight: 600 !important; }}
h3 {{ font-size: 1.25rem !important; }}
.stApp {{ background:
    radial-gradient(900px 500px at 8% -8%, rgba(247,147,26,0.13), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(255,71,87,0.07), transparent 55%),
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px) 0 0/34px 34px,
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px) 0 0/34px 34px, {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1250px; }}

/* ---------- top navigation bar ---------- */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: rgba(11,11,14,0.72) !important; backdrop-filter: blur(14px) saturate(140%);
    border-bottom: 1px solid rgba(247,147,26,0.22); box-shadow: 0 8px 30px rgba(0,0,0,0.35); height: 3.6rem; }}
[data-testid="stHeader"] img {{ height: 38px; }}
[data-testid="stTopNavLinkContainer"] {{ gap: 4px; }}
[data-testid="stTopNavLink"], [data-testid="stHeader"] a[href*="localhost"], [data-testid="stHeader"] nav a {{ border-radius: 999px !important; padding: 7px 16px !important;
    font-family: 'Unbounded' !important; font-weight: 600 !important; transition: all .18s ease; color: #c9c9d4 !important; }}
[data-testid="stTopNavLink"], [data-testid="stHeader"] nav a {{ position: relative; overflow: hidden; isolation: isolate; font-family: 'Unbounded' !important; font-size: .74rem !important; font-weight: 500 !important; letter-spacing: .02em;
    transition: color .3s, transform .35s cubic-bezier(.2,.9,.3,1.4), box-shadow .35s; }}
/* liquid fill that sweeps in from the left */
[data-testid="stTopNavLink"]:before, [data-testid="stHeader"] nav a:before {{ content: ''; position: absolute; inset: 0; z-index: -1; border-radius: inherit;
    background: linear-gradient(110deg, rgba(247,147,26,.0), rgba(247,147,26,.38) 45%, rgba(255,200,87,.30)); transform: translateX(-101%) skewX(-18deg); transition: transform .5s cubic-bezier(.2,.8,.2,1); }}
/* underline that grows from the centre */
[data-testid="stTopNavLink"]:after, [data-testid="stHeader"] nav a:after {{ content: ''; position: absolute; left: 50%; bottom: 3px; height: 2px; width: 0; border-radius: 2px; transform: translateX(-50%);
    background: linear-gradient(90deg, #ffc857, #f7931a); box-shadow: 0 0 10px #f7931a; transition: width .35s cubic-bezier(.2,.8,.2,1); }}
[data-testid="stTopNavLink"]:hover, [data-testid="stHeader"] nav a:hover {{ color: #fff !important; transform: translateY(-3px) scale(1.06); box-shadow: 0 10px 24px rgba(247,147,26,.28), inset 0 0 0 1px rgba(247,147,26,.45); }}
[data-testid="stTopNavLink"]:hover:before, [data-testid="stHeader"] nav a:hover:before {{ transform: translateX(0) skewX(0); }}
[data-testid="stTopNavLink"]:hover:after, [data-testid="stHeader"] nav a:hover:after {{ width: 60%; }}
[data-testid="stTopNavLink"]:active, [data-testid="stHeader"] nav a:active {{ transform: translateY(0) scale(.96); }}
[data-testid="stTopNavLink"] [data-testid="stIconMaterial"], [data-testid="stHeader"] nav a span[class*="icon"] {{ transition: transform .5s cubic-bezier(.2,.9,.3,1.6), color .3s; }}
[data-testid="stTopNavLink"]:hover [data-testid="stIconMaterial"], [data-testid="stHeader"] nav a:hover span[class*="icon"] {{ transform: rotate(-14deg) scale(1.3); color: #ffc857; }}
[data-testid="stTopNavLink"][aria-current="page"], [data-testid="stHeader"] nav a[aria-current="page"] {{ background: linear-gradient(135deg, #f7931a, #c46f08) !important; color: #1a0f00 !important;
    box-shadow: 0 0 22px rgba(247,147,26,0.45); }}
[data-testid="stTopNavLink"][aria-current="page"]:after {{ background: #1a0f00; box-shadow: none; }}
/* ripple (added by fx.js on click) */
.cs-rip {{ position: absolute; border-radius: 50%; pointer-events: none; background: rgba(255,200,87,.55); transform: scale(0); animation: rip .6s ease-out forwards; }}
@keyframes rip {{ to {{ transform: scale(4); opacity: 0; }} }}
[data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; }}
.block-container {{ animation: pagein .55s cubic-bezier(.2,.8,.2,1) both; }}
@keyframes pagein {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(18px) scale(.985); }} to {{ opacity: 1; transform: none; }} }}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 rgba(255,71,87,0); }} 50% {{ box-shadow: 0 0 38px rgba(255,71,87,.38); }} }}
@keyframes drift {{ from {{ background-position: 0% 50%; }} to {{ background-position: 200% 50%; }} }}
.chainbox {{ border: 1px solid {BORDER}; border-radius: 14px; padding: 12px 12px 10px 12px; background: rgba(255,255,255,0.02); margin-top: 6px; }}
.chainbox .h {{ font-family: 'JetBrains Mono'; font-size: .64rem; letter-spacing: .12em; color: {MUTED}; margin-bottom: 8px; display:flex; justify-content: space-between; }}
.chainbox .s {{ font-size: .78rem; color: #cfcfd8; margin-top: 8px; }}
.chain {{ display:flex; flex-wrap: wrap; gap: 3px; }}
.blk {{ width: 11px; height: 11px; border-radius: 3px; background: #2a2a35; position: relative; }}
.blk.g {{ background: {LICIT}; }} .blk.a {{ background: {GOLD}; }} .blk.r {{ background: {ILLICIT}; }} .blk.n {{ background: #55556a; }}
.blk.t {{ background: #252532; border: 1px solid #34344a; }} .blk.cur {{ outline: 2px solid {BTC}; outline-offset: 1px; }}
.chain.big .blk {{ width: 22px; height: 22px; border-radius: 5px; }}
.legend {{ display:flex; gap: 14px; flex-wrap: wrap; font-size: .76rem; color: {MUTED}; margin-top: 8px; }}
.legend i {{ display:inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }}

/* ---------- hero ---------- */
.hero {{ position: relative; overflow: hidden; padding: 40px 44px; border-radius: 24px; border: 1px solid #3a3a48; margin-bottom: 24px;
    background: linear-gradient(115deg, rgba(247,147,26,0.22), rgba(19,19,24,0.94) 38%, rgba(255,71,87,0.14) 70%, rgba(247,147,26,0.18)); background-size: 220% 100%;
    animation: drift 14s linear infinite alternate; }}
.hero:before {{ content: ''; position:absolute; inset: 0; opacity: .55;
    background: repeating-linear-gradient(90deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 46px), repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 46px); }}
.hero:after {{ content: ''; position: absolute; right: -90px; top: -90px; width: 380px; height: 380px; border-radius: 50%;
    background: repeating-radial-gradient(circle, rgba(247,147,26,0.0) 0 24px, rgba(247,147,26,0.26) 25px 26px); animation: spin 40s linear infinite; mask-image: radial-gradient(circle, #000 20%, transparent 70%); }}
@keyframes spin {{ to {{ transform: rotate(360deg) scale(1.08); }} }}
.hero > * {{ position: relative; z-index: 1; }}
.hero h1 {{ font-size: 2.05rem; margin: 6px 0 8px 0; line-height: 1.08; max-width: 900px;
    background: linear-gradient(120deg, #ffffff 30%, #ffc857 70%, #f7931a); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero p {{ color: #cbcbd6; font-size: 1.06rem; max-width: 760px; margin: 0; }}
.kicker {{ font-family: 'JetBrains Mono'; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; color: {BTC}; font-weight: 700; }}
.badge {{ display:inline-block; padding: 4px 11px; margin: 14px 8px 0 0; border-radius: 6px; font-size: .76rem; font-family: 'JetBrains Mono';
    border: 1px solid #3a3a48; background: rgba(0,0,0,0.35); color: {TEXT}; }}

/* ---------- components ---------- */
.card, .kpi, .step, .verdict {{ transition: transform .14s ease-out, box-shadow .25s ease, border-color .25s ease; transform-style: preserve-3d; will-change: transform; animation: rise .6s both; }}
.card:hover, .kpi:hover, .step:hover {{ border-color: rgba(247,147,26,.6); box-shadow: 0 14px 40px rgba(0,0,0,.45), 0 0 26px rgba(247,147,26,.14); background-image: radial-gradient(260px circle at var(--gx,50%) var(--gy,0%), rgba(247,147,26,.14), transparent 60%); }}
.card {{ min-height: 168px; background: linear-gradient(180deg, {PANEL2}, {PANEL}); border: 1px solid {BORDER}; border-top: 2px solid {BTC_D}; border-radius: 14px; padding: 18px 20px; height: 100%; }}
.card h4 {{ margin: 0 0 6px 0; font-size: 1.02rem; color: {TEXT}; }}
.card p, .card li {{ color: #b9b9c6; font-size: .92rem; margin: 0; }}
.kpi {{ min-height: 112px; background: linear-gradient(180deg, {PANEL2}, {PANEL}); border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; position: relative; overflow: hidden; }}
.kpi:before {{ content:''; position:absolute; left:0; top:0; bottom:0; width: 3px; background: {BTC}; }}
.kpi .l {{ color: {MUTED}; font-family: 'JetBrains Mono'; font-size: .68rem; text-transform: uppercase; letter-spacing: .1em; }}
.kpi .v {{ font-family: 'JetBrains Mono'; font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin-top: 4px; }}
.kpi .s {{ color: {MUTED}; font-size: .8rem; margin-top: 2px; }}
.tone-good .v {{ color: {LICIT}; }} .tone-bad .v {{ color: {ILLICIT}; }} .tone-warn .v {{ color: {GOLD}; }} .tone-neutral .v {{ color: {TEXT}; }}
.tone-good:before {{ background: {LICIT}; }} .tone-bad:before {{ background: {ILLICIT}; }} .tone-warn:before {{ background: {GOLD}; }}
.explain {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 18px 0; }}
.explain > div {{ border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 15px; background: rgba(255,255,255,0.025); }}
.explain > div > b:first-child {{ display:block; font-family: 'JetBrains Mono'; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:5px; }}
.explain .a > b:first-child {{ color: {SKY}; }} .explain .b > b:first-child {{ color: {GOLD}; }} .explain .c > b:first-child {{ color: {BTC}; }}
.explain span b {{ color: #fff; font-weight: 700; }}
.explain span {{ color: #c8c8d4; font-size: .9rem; }}
.callout {{ border-radius: 12px; padding: 14px 18px; margin: 8px 0 16px 0; border: 1px solid {BORDER}; border-left-width: 4px; font-size: .95rem; color: #e1e1ea; }}
.callout.bad {{ border-color: rgba(255,71,87,.55); background: rgba(255,71,87,.09); }}
.callout.good {{ border-color: rgba(52,211,153,.5); background: rgba(52,211,153,.08); }}
.callout.warn {{ border-color: rgba(255,200,87,.55); background: rgba(255,200,87,.08); }}
.callout.info {{ border-color: rgba(247,147,26,.55); background: rgba(247,147,26,.08); }}
.steps {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 16px 0; }}
.step {{ flex: 1 1 150px; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 14px; background: {PANEL}; }}
.step .n {{ width: 30px; height: 30px; clip-path: polygon(25% 5%, 75% 5%, 100% 50%, 75% 95%, 25% 95%, 0 50%); background: #3a3a48; color: {TEXT};
    font-family: 'JetBrains Mono'; font-weight: 700; display:flex; align-items:center; justify-content:center; font-size: .8rem; margin-bottom: 8px; }}
.step.done {{ border-color: rgba(247,147,26,.5); }} .step.done .n {{ background: {BTC}; color: #1a0f00; }}
.step b {{ color: {TEXT}; font-size: .92rem; }} .step p {{ color: {MUTED}; font-size: .8rem; margin: 3px 0 0 0; }}
.verdict {{ border-radius: 18px; padding: 22px 26px; border: 1px solid {BORDER}; margin: 6px 0 14px 0; }}
.verdict.bad {{ animation: pulse 2.6s ease-in-out infinite; background: linear-gradient(135deg, rgba(255,71,87,.24), rgba(19,19,24,.92)); border-color: rgba(255,71,87,.65); }}
.verdict.good {{ background: linear-gradient(135deg, rgba(52,211,153,.20), rgba(19,19,24,.92)); border-color: rgba(52,211,153,.55); }}
.verdict .t {{ font-family: 'Unbounded'; font-size: 1.35rem; font-weight: 700; }}
.verdict .d {{ color: #d4d4de; margin-top: 4px; }}
.reason {{ display:flex; align-items:center; gap:10px; padding: 9px 12px; border-radius: 10px; margin-bottom: 7px; background: rgba(255,255,255,0.03); border:1px solid {BORDER}; font-size:.9rem; color:#dcdce6; }}
.dot {{ width: 10px; height: 10px; border-radius: 3px; flex: none; transform: rotate(45deg); }}
.pill {{ display:inline-block; padding: 3px 10px; border-radius: 6px; font-size: .76rem; font-weight: 700; font-family: 'JetBrains Mono'; }}
.pill.good {{ background: rgba(52,211,153,.18); color: {LICIT}; }} .pill.bad {{ background: rgba(255,71,87,.18); color: {ILLICIT_L}; }}
.pill.warn {{ background: rgba(255,200,87,.18); color: {GOLD}; }}
.small {{ color: {MUTED}; font-size: .82rem; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px 14px; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Unbounded'; font-weight: 600; }}
.stButton>button, .stDownloadButton>button {{ border-radius: 10px; border: 1px solid {BORDER}; }}
/* ---------- extra polish ---------- */
::selection {{ background: rgba(247,147,26,.45); color: #fff; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }} ::-webkit-scrollbar-track {{ background: {INK}; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, {BTC}, {BTC_D}); border-radius: 8px; border: 2px solid {INK}; }}
.card {{ position: relative; overflow: hidden; }}
.card:before {{ content: ''; position: absolute; top: 0; left: -60%; width: 40%; height: 100%; background: linear-gradient(100deg, transparent, rgba(255,255,255,.07), transparent); transform: skewX(-20deg); transition: left .8s ease; pointer-events: none; }}
.card:hover:before {{ left: 130%; }}
.kpi:hover .v {{ text-shadow: 0 0 18px currentColor; }}
.kpi .v, .badge, .pill {{ transition: text-shadow .3s, transform .25s, border-color .25s, background .25s; }}
.badge:hover {{ transform: translateY(-3px); border-color: {BTC}; background: rgba(247,147,26,.14); }}
.blk {{ transition: transform .2s, box-shadow .2s; }} .blk:hover {{ transform: scale(1.9); z-index: 3; box-shadow: 0 0 12px rgba(247,147,26,.8); }}
.step .n {{ transition: transform .4s cubic-bezier(.2,.9,.3,1.6); }} .step:hover .n {{ transform: rotate(60deg) scale(1.15); }}
.reason, .callout, .explain > div {{ transition: transform .25s, border-color .25s, background .25s; }}
.reason:hover {{ transform: translateX(6px); border-color: rgba(247,147,26,.5); }}
.callout:hover, .explain > div:hover {{ transform: translateY(-3px); border-color: rgba(247,147,26,.45); }}
.kicker:after {{ content: '\\2588'; margin-left: 6px; animation: blink 1.1s steps(2) infinite; }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
.hero h1 {{ background-size: 200% 100%; animation: shine 6s linear infinite; }}
@keyframes shine {{ to {{ background-position: -200% 0; }} }}
.stButton>button, .stDownloadButton>button {{ transition: transform .2s, box-shadow .25s, border-color .25s; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ transform: translateY(-2px); border-color: {BTC}; box-shadow: 0 8px 22px rgba(247,147,26,.3); }}
.stTabs [data-baseweb="tab"] {{ transition: color .25s, transform .25s; }} .stTabs [data-baseweb="tab"]:hover {{ transform: translateY(-2px); color: {BTC}; }}
[data-testid="stHorizontalBlock"] > div:nth-child(2) .kpi {{ animation-delay: .08s; }} [data-testid="stHorizontalBlock"] > div:nth-child(3) .kpi {{ animation-delay: .16s; }}
[data-testid="stHorizontalBlock"] > div:nth-child(4) .kpi {{ animation-delay: .24s; }} [data-testid="stHorizontalBlock"] > div:nth-child(5) .kpi {{ animation-delay: .32s; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
@media (max-width: 800px) {{ .explain {{ grid-template-columns: 1fr; }} .hero h1 {{ font-size: 1.8rem; }} }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker, title, subtitle, badges=()):
    b = "".join(f'<span class="badge">{x}</span>' for x in badges)
    st.markdown(f'<div class="hero"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p>{b}</div>',
                unsafe_allow_html=True)


def explain(what, how, sowhat):
    """Three-part plain-English panel: what you see / how to read it / why it matters."""
    st.markdown(f'<div class="explain"><div class="a"><b>What you are looking at</b><span>{what}</span></div>'
                f'<div class="b"><b>How to read it</b><span>{how}</span></div>'
                f'<div class="c"><b>So what?</b><span>{sowhat}</span></div></div>', unsafe_allow_html=True)


def kpi(label, value, sub="", tone="neutral"):
    return (f'<div class="kpi tone-{tone}"><div class="l">{label}</div><div class="v">{value}</div>'
            f'<div class="s">{sub}</div></div>')


def kpis(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(kpi(*it), unsafe_allow_html=True)


def card(title, body):
    return f'<div class="card"><h4>{title}</h4><p>{body}</p></div>'


def callout(text, tone="info"):
    st.markdown(f'<div class="callout {tone}">{text}</div>', unsafe_allow_html=True)


def steps(items, done_upto=-1):
    html = ""
    for i, (title, desc) in enumerate(items):
        cls = "step done" if i <= done_upto else "step"
        html += f'<div class="{cls}"><div class="n">{i + 1}</div><b>{title}</b><p>{desc}</p></div>'
    st.markdown(f'<div class="steps">{html}</div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f"### {title}")
    if sub:
        st.markdown(f'<div class="small" style="margin-top:-8px;margin-bottom:10px">{sub}</div>', unsafe_allow_html=True)


def chain(blocks, big=False, current=None):
    """Row of small blocks. blocks = [(css_class, tooltip), ...]; one block per time step."""
    html = "".join(f'<div class="blk {c}{" cur" if current == i else ""}" title="{t}"></div>' for i, (c, t) in enumerate(blocks))
    return f'<div class="chain{" big" if big else ""}">{html}</div>'


def chain_legend():
    return ('<div class="legend"><span><i style="background:#252532;border:1px solid #34344a"></i>training</span>'
            f'<span><i style="background:{LICIT}"></i>healthy</span><span><i style="background:{GOLD}"></i>degraded</span>'
            f'<span><i style="background:{ILLICIT}"></i>failing</span><span><i style="background:#55556a"></i>too few cases</span></div>')


def style(fig, height=380, legend=True):
    """Apply the dark theme to a Plotly figure."""
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT, family="Manrope"), margin=dict(l=10, r=10, t=50, b=10),
                      showlegend=legend, legend=dict(bgcolor="rgba(0,0,0,0)"),
                      colorway=[BTC, ILLICIT, SKY, GOLD, "#b79bff"])
    if fig.layout.title.text:              # only style a title that exists (otherwise Plotly shows "undefined")
        fig.update_layout(title_font=dict(family="Unbounded", size=16))
    fig.update_xaxes(gridcolor="#1f1f28", zerolinecolor="#1f1f28")
    fig.update_yaxes(gridcolor="#1f1f28", zerolinecolor="#1f1f28")
    return fig
