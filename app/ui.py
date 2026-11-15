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
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;600&family=Manrope:wght@400;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
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
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: rgba(11,11,14,0.72) !important; backdrop-filter: blur(14px) saturate(140%);
    border-bottom: 1px solid rgba(247,147,26,0.22); box-shadow: 0 8px 30px rgba(0,0,0,0.35); height: 3.6rem; }}
[data-testid="stHeader"] img {{ height: 38px; }}
/* "Lens" navigation: a glass capsule with a liquid highlight that glides to whatever you point at (positioned by fx.py) */
[data-testid="stTopNavLinkContainer"] {{ gap: 0 !important; }}
.cs-nav {{ position: relative; display: flex; align-items: center; gap: 2px; padding: 5px; border-radius: 999px; background: rgba(255,255,255,.035);
    border: 1px solid rgba(247,147,26,.25); box-shadow: inset 0 0 18px rgba(247,147,26,.06), 0 6px 24px rgba(0,0,0,.35); }}
[data-testid="stTopNavLink"], [data-testid="stHeader"] nav a {{ position: relative; z-index: 2; border-radius: 999px !important; padding: 6px 11px !important; background: transparent !important;
    font-family: 'Unbounded' !important; font-size: .64rem !important; font-weight: 500 !important; letter-spacing: .02em; color: #b4b4c2 !important;
    transition: color .25s, transform .3s cubic-bezier(.2,.9,.3,1.5); }}
[data-testid="stTopNavLink"]:hover, [data-testid="stHeader"] nav a:hover {{ color: #fff !important; transform: translateY(-1px); }}
[data-testid="stTopNavLink"]:active {{ transform: scale(.94); }}
[data-testid="stTopNavLink"][aria-current="page"] {{ color: #ffc857 !important; }}
[data-testid="stTopNavLink"] [data-testid="stIconMaterial"] {{ transition: transform .5s cubic-bezier(.2,.9,.3,1.7), color .3s; }}
[data-testid="stTopNavLink"]:hover [data-testid="stIconMaterial"] {{ transform: rotate(-16deg) scale(1.35); }}
.cs-lens, .cs-trail {{ position: absolute; top: 5px; bottom: 5px; left: 0; width: 0; border-radius: 999px; pointer-events: none; z-index: 1; opacity: 0; }}
.cs-lens {{ background: linear-gradient(135deg, #ffb340, #f7931a 55%, #c46f08); box-shadow: 0 0 22px rgba(247,147,26,.55), inset 0 1px 0 rgba(255,255,255,.35);
    transition: transform .42s cubic-bezier(.34,1.45,.5,1), width .42s cubic-bezier(.34,1.45,.5,1), opacity .3s, background .3s, box-shadow .3s; }}
.cs-trail {{ z-index: 0; background: rgba(247,147,26,.5); filter: blur(10px); transition: transform .8s cubic-bezier(.2,.8,.2,1), width .8s cubic-bezier(.2,.8,.2,1), opacity .3s; }}
.cs-nav[data-mode="hover"] .cs-lens {{ background: rgba(247,147,26,.22); box-shadow: 0 0 20px rgba(247,147,26,.35), inset 0 0 0 1px rgba(255,200,87,.55); }}
.cs-nav[data-mode="solid"] .cs-under {{ color: #1a0f00 !important; font-weight: 600 !important; }}
.cs-nav[data-mode="hover"] .cs-under {{ color: #fff !important; }}
/* ripple (added by fx.py on click) */
.cs-rip {{ position: absolute; border-radius: 50%; pointer-events: none; background: rgba(255,200,87,.55); transform: scale(0); animation: rip .6s ease-out forwards; }}
@keyframes rip {{ to {{ transform: scale(4); opacity: 0; }} }}

/* ---------- phone dock (built by fx.py, only visible on small screens) ---------- */
#cs-dock {{ display: none; }}
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
.kpi:after {{ content: ''; position: absolute; right: -30px; top: -30px; width: 90px; height: 90px; border-radius: 50%; background: radial-gradient(circle, rgba(247,147,26,.16), transparent 70%); transition: transform .5s; }}
.kpi:hover:after {{ transform: scale(2.2); }}
.tone-good:after {{ background: radial-gradient(circle, rgba(52,211,153,.18), transparent 70%); }} .tone-bad:after {{ background: radial-gradient(circle, rgba(255,71,87,.2), transparent 70%); }}
h3 {{ position: relative; padding-left: 16px !important; }} h3:before {{ content: ''; position: absolute; left: 0; top: .25em; bottom: .25em; width: 4px; border-radius: 3px; background: linear-gradient(180deg, {GOLD}, {BTC}); box-shadow: 0 0 10px rgba(247,147,26,.6); }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
/* ---------- phone / tablet ---------- */
@media (max-width: 820px) {{
  [data-testid="stTopNavLinkContainer"], [data-testid="stHeader"] nav {{ display: none !important; }}
  html {{ font-size: 15px; }}
  [data-testid="stHeader"] {{ height: 3rem; }} [data-testid="stHeader"] img {{ height: 30px; }}
  .block-container {{ padding: 4.2rem .85rem 7.5rem !important; }}
  .hero {{ padding: 22px 18px; border-radius: 18px; margin-bottom: 16px; }}
  .hero h1 {{ font-size: 1.35rem; line-height: 1.2; }} .hero p {{ font-size: .92rem; }}
  .hero:after {{ width: 220px; height: 220px; right: -70px; top: -70px; }}
  .kicker {{ font-size: .62rem; }} .badge {{ font-size: .66rem; padding: 3px 8px; margin: 10px 6px 0 0; }}
  h3 {{ font-size: 1.02rem !important; }}
  .kpi {{ min-height: 86px; padding: 12px 14px; }} .kpi .v {{ font-size: 1.4rem; }} .kpi .l {{ font-size: .6rem; }}
  .card {{ min-height: 0; padding: 14px 16px; }}
  .step {{ flex: 1 1 100%; }}
  .verdict {{ padding: 16px 18px; }} .verdict .t {{ font-size: 1.05rem; }}
  .callout {{ padding: 11px 14px; font-size: .88rem; }}
  .chain.big .blk {{ width: 15px; height: 15px; }}
  .explain {{ gap: 8px; }} .explain > div {{ padding: 10px 13px; }} .explain span {{ font-size: .84rem; line-height: 1.45; }}
  .stButton>button, .stDownloadButton>button {{ min-height: 46px; width: 100%; }}
  div[data-testid="stMetric"] {{ padding: 8px 10px; }}
  iframe[height="560"] {{ height: 470px !important; }} iframe[height="700"] {{ height: 560px !important; }}
  .stTabs [data-baseweb="tab"] {{ padding: 8px 10px; font-size: .74rem; }}
  .js-plotly-plot {{ max-width: 100%; }}

  #cs-dock {{ display: block; position: fixed; z-index: 99990; left: 10px; right: 10px; bottom: calc(10px + env(safe-area-inset-bottom)); height: 64px; border-radius: 22px;
      background: rgba(14,14,19,.82); backdrop-filter: blur(16px) saturate(150%); border: 1px solid rgba(247,147,26,.3); box-shadow: 0 10px 34px rgba(0,0,0,.6), inset 0 0 22px rgba(247,147,26,.06); }}
  #cs-dock .row {{ position: relative; display: grid; grid-template-columns: repeat(7, 1fr); height: 100%; }}
  #cs-dock a {{ position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; color: #8f8fa0; text-decoration: none; -webkit-tap-highlight-color: transparent; }}
  #cs-dock .hx {{ width: 26px; height: 26px; clip-path: polygon(25% 5%, 75% 5%, 100% 50%, 75% 95%, 25% 95%, 0 50%); background: #2c2c38; display: flex; align-items: center; justify-content: center;
      font: 700 11px 'JetBrains Mono', monospace; color: #cfcfdc; transition: transform .4s cubic-bezier(.3,1.6,.5,1), background .3s, color .3s; }}
  #cs-dock .lb {{ font: 600 8.5px 'Unbounded', sans-serif; letter-spacing: .02em; transition: color .3s; }}
  #cs-dock a.on .hx {{ background: linear-gradient(135deg, #ffb340, #f7931a); color: #1a0f00; transform: translateY(-5px) scale(1.22); box-shadow: 0 0 16px rgba(247,147,26,.8); }}
  #cs-dock a.on .lb {{ color: #ffc857; transform: translateY(-2px); }}
  #cs-dock a:active .hx {{ transform: scale(.82); }}
  #cs-dock .bar {{ position: absolute; z-index: 1; top: -1px; left: 0; width: calc(100% / 7); height: 3px; display: flex; justify-content: center; transition: transform .45s cubic-bezier(.34,1.45,.5,1); }}
  #cs-dock .bar:before {{ content: ''; width: 46%; height: 100%; border-radius: 3px; background: linear-gradient(90deg, #ffc857, #f7931a); box-shadow: 0 0 12px #f7931a, 0 6px 18px rgba(247,147,26,.55); }}
}}
@media (hover: none) {{
  .card:hover, .kpi:hover, .step:hover {{ background-image: none; box-shadow: none; }}
  .card:hover:before {{ left: -60%; }} .blk:hover {{ transform: none; box-shadow: none; }}
}}
@media (max-width: 420px) {{ .hero h1 {{ font-size: 1.2rem; }} #cs-dock .lb {{ font-size: 7.5px; }} }}
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
