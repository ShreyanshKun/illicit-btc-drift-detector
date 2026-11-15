"""Global front-end effects injected into the Streamlit page (no extra packages).

Streamlit renders components.html in a same-origin iframe, so a tiny script can reach
window.parent.document and add: a constellation background, a cursor spotlight,
3-D card tilt, count-up numbers and a scroll-progress bar. Everything is wrapped in
try/catch and runs once (guarded by an element id), so failures never break the app.
"""
import streamlit.components.v1 as components

JS = r"""
<script>
(function(){
try{
const P = window.parent, D = P.document;
if (D.getElementById('cs-fx')) return;
const root = D.createElement('div'); root.id = 'cs-fx'; D.body.appendChild(root);

/* ---- constellation background ---- */
const cv = D.createElement('canvas'); cv.id = 'cs-bg';
cv.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;opacity:.9';
root.appendChild(cv);
const ctx = cv.getContext('2d'); let W, H, dpr = Math.min(P.devicePixelRatio || 1, 2);
const TOUCH = P.matchMedia('(hover: none)').matches, SMALL = P.innerWidth < 820;
const mouse = {x:-999, y:-999}; const N = SMALL ? 26 : 70, pts = [];
function size(){ W = P.innerWidth; H = P.innerHeight; cv.width = W*dpr; cv.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); }
size(); P.addEventListener('resize', size);
for (let i=0;i<N;i++) pts.push({x:Math.random()*W, y:Math.random()*H, vx:(Math.random()-.5)*.28, vy:(Math.random()-.5)*.28, bad:Math.random()<.08});
function tick(){
  ctx.clearRect(0,0,W,H);
  for (const p of pts){ p.x+=p.vx; p.y+=p.vy; if(p.x<0||p.x>W)p.vx*=-1; if(p.y<0||p.y>H)p.vy*=-1;
    const dx=p.x-mouse.x, dy=p.y-mouse.y, d=Math.hypot(dx,dy); if(d<140){ p.x+=dx/d*0.9; p.y+=dy/d*0.9; } }
  for (let i=0;i<N;i++){ const a=pts[i];
    for (let j=i+1;j<N;j++){ const b=pts[j], d=Math.hypot(a.x-b.x,a.y-b.y);
      if(d<150){ ctx.strokeStyle=`rgba(247,147,26,${(1-d/150)*0.16})`; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); } }
    ctx.fillStyle = a.bad ? 'rgba(255,71,87,.75)' : 'rgba(255,200,87,.45)'; ctx.beginPath(); ctx.arc(a.x,a.y,a.bad?2.2:1.5,0,6.283); ctx.fill(); }
  P.requestAnimationFrame(tick);
}
tick();

/* ---- cursor spotlight ---- */
const sp = D.createElement('div'); sp.id='cs-spot';
sp.style.cssText = 'position:fixed;left:0;top:0;width:520px;height:520px;margin:-260px 0 0 -260px;border-radius:50%;pointer-events:none;z-index:2;mix-blend-mode:screen;'+
  'background:radial-gradient(circle,rgba(247,147,26,.10) 0%,rgba(247,147,26,.04) 40%,transparent 70%);transition:opacity .3s';
if (!TOUCH) root.appendChild(sp);
if (!TOUCH) D.addEventListener('mousemove', e=>{ mouse.x=e.clientX; mouse.y=e.clientY; sp.style.transform=`translate(${e.clientX}px,${e.clientY}px)`; });

/* ---- 3-D tilt with glare on cards ---- */
const SEL = '.card,.kpi,.step,.verdict';
if (!TOUCH) D.addEventListener('mousemove', e=>{
  const el = e.target.closest ? e.target.closest(SEL) : null;
  if (D.__tilted && D.__tilted !== el){ D.__tilted.style.transform=''; D.__tilted.style.removeProperty('--gx'); D.__tilted = null; }
  if (!el) return; D.__tilted = el;
  const r = el.getBoundingClientRect(), px=(e.clientX-r.left)/r.width, py=(e.clientY-r.top)/r.height;
  el.style.transform = `perspective(700px) rotateX(${(0.5-py)*7}deg) rotateY(${(px-0.5)*9}deg) translateZ(0)`;
  el.style.setProperty('--gx', (px*100)+'%'); el.style.setProperty('--gy', (py*100)+'%');
});

/* ---- count-up numbers ---- */
function countUp(el){
  if (el.dataset.cu) return; el.dataset.cu='1';
  const txt = el.textContent.trim(); const m = txt.match(/^([^0-9\-]*)([0-9][0-9,]*\.?[0-9]*)(.*)$/); if(!m) return;
  const target = parseFloat(m[2].replace(/,/g,'')); const dec = (m[2].split('.')[1]||'').length; const comma = m[2].includes(',');
  const t0 = performance.now(), dur = 1100;
  (function step(now){ const k = Math.min(1,(now-t0)/dur), e = 1-Math.pow(1-k,3), v = target*e;
    let s = v.toFixed(dec); if (comma) s = Number(s).toLocaleString('en-US',{minimumFractionDigits:dec,maximumFractionDigits:dec});
    el.textContent = m[1]+s+m[3]; if (k<1) P.requestAnimationFrame(step); })(t0);
}
new P.MutationObserver(()=>{ D.querySelectorAll('.kpi .v').forEach(countUp); }).observe(D.body,{childList:true,subtree:true});
D.querySelectorAll('.kpi .v').forEach(countUp);

/* ---- click ripple on nav links and buttons ---- */
D.addEventListener('click', e=>{
  const el = e.target.closest && e.target.closest('[data-testid="stTopNavLink"],[data-testid="stHeader"] nav a,.stButton>button');
  if(!el) return; const r = el.getBoundingClientRect(), d = Math.max(r.width,r.height), s = D.createElement('span');
  s.className='cs-rip'; s.style.cssText=`width:${d}px;height:${d}px;left:${e.clientX-r.left-d/2}px;top:${e.clientY-r.top-d/2}px`;
  if(getComputedStyle(el).position==='static') el.style.position='relative'; el.style.overflow='hidden'; el.appendChild(s); setTimeout(()=>s.remove(),650);
});

/* ---- "lens" nav: a liquid highlight that glides to the hovered / active link ---- */
function setupLens(){
  const links = [...D.querySelectorAll('[data-testid="stTopNavLink"]')]; if (links.length < 2) return;
  let box = links[0]; while (box && !links.every(l => box.contains(l))) box = box.parentElement; if (!box || box.dataset.lens) return; box.dataset.lens = '1'; box.classList.add('cs-nav');
  const trail = D.createElement('div'), lens = D.createElement('div'); trail.className = 'cs-trail'; lens.className = 'cs-lens';
  box.prepend(lens); box.prepend(trail);
  const active = () => links.find(l => l.getAttribute('aria-current') === 'page') || links[0];
  function go(el, mode){
    const bx = box.getBoundingClientRect(), r = el.getBoundingClientRect(), x = r.left - bx.left, w = r.width;
    for (const e of [lens, trail]){ e.style.transform = `translateX(${x}px)`; e.style.width = w + 'px'; e.style.opacity = e === trail ? (mode==='hover' ? .9 : .5) : 1; }
    box.dataset.mode = mode; links.forEach(l => l.classList.toggle('cs-under', l === el));
  }
  links.forEach(l => { l.addEventListener('mouseenter', () => go(l, l === active() ? 'solid' : 'hover')); });
  box.addEventListener('mouseleave', () => go(active(), 'solid'));
  P.addEventListener('resize', () => go(active(), 'solid'));
  go(active(), 'solid'); setInterval(() => { if (box.dataset.mode === 'solid') go(active(), 'solid'); }, 600);
}

/* ---- phone dock: bottom tab bar of hexagon badges (shown by CSS under 820px) ---- */
const PAGES = [['Home','overview'],['Scan','scanner'],['3D','universe'],['Lab','model-lab'],['Drift','drift-radar'],['Why','explainer'],['Guide','field-guide']];
function setupDock(){
  if (D.getElementById('cs-dock')) return;
  const dock = D.createElement('div'); dock.id = 'cs-dock';
  dock.innerHTML = '<div class="row"><div class="bar"></div>' + PAGES.map((p,i) => `<a href="#" data-p="${p[1]}"><span class="hx">${i+1}</span><span class="lb">${p[0]}</span></a>`).join('') + '</div>';
  D.body.appendChild(dock);
  const cur = () => { const seg = P.location.pathname.split('/').filter(Boolean).pop() || 'overview'; return PAGES.findIndex(p => p[1] === seg) < 0 ? 0 : PAGES.findIndex(p => p[1] === seg); };
  function sync(){ const i = cur(); dock.querySelectorAll('a').forEach((a,k) => a.classList.toggle('on', k === i)); dock.querySelector('.bar').style.transform = `translateX(${i*100}%)`; }
  dock.addEventListener('click', e => {
    const a = e.target.closest('a'); if (!a) return; e.preventDefault();
    const slug = a.dataset.p; try{ P.navigator.vibrate && P.navigator.vibrate(8); }catch(_){}
    const known = PAGES.map(q => q[1]), base = P.location.pathname.replace(/\/+$/, '').split('/').filter((x, i, arr) => !(i === arr.length - 1 && known.includes(x))).join('/');
    const real = [...D.querySelectorAll('a[href]')].find(l => { if (l.closest('#cs-dock')) return false; const q = l.pathname.replace(/\/+$/, '');
      return slug === 'overview' ? (q === base) : q.endsWith('/' + slug); });             // Streamlit's own link keeps the session alive
    if (real){ real.click(); } else { P.location.href = (slug === 'overview' ? (base || '/') : base + '/' + slug); }
    setTimeout(sync, 150);
  });
  sync(); setInterval(sync, 500);
}
new P.MutationObserver(() => { setupLens(); setupDock(); }).observe(D.body, {childList:true, subtree:true});
setupLens(); setupDock();

/* ---- scroll progress ---- */
const bar = D.createElement('div'); bar.id='cs-prog';
bar.style.cssText='position:fixed;left:0;top:0;height:3px;width:0;z-index:9999;background:linear-gradient(90deg,#f7931a,#ffc857,#ff4757);box-shadow:0 0 12px rgba(247,147,26,.8)';
root.appendChild(bar);
function prog(){ const sc = D.querySelector('[data-testid="stMain"]') || D.scrollingElement; if(!sc) return;
  const h = sc.scrollHeight - sc.clientHeight; bar.style.width = (h>0 ? sc.scrollTop/h*100 : 0)+'%'; }
(D.querySelector('[data-testid="stMain"]')||P).addEventListener('scroll', prog, true); P.addEventListener('scroll', prog, true);
}catch(err){ console.log('fx disabled', err); }
})();
</script>
"""


def inject():
    """Call once per script run; the script itself de-duplicates."""
    components.html(JS, height=0)
