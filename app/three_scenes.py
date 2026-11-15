"""Three.js scenes embedded in the Streamlit app (rendered inside components.html iframes).

1. tunnel_html():   "Drift tunnel" hero. A fly-through of 49 glowing rings, one per time step,
                    coloured by how well the XGBoost model coped. You can see step 43 break.
2. universe_html(): "Transaction universe". Real transactions as an orbitable 3-D point cloud
                    (PCA of the 165 features). Colour by truth / model verdict / period, hover
                    for details and scrub through time.

Scenes load three.js r160 from a CDN and need WebGL. Data is injected as JSON.
"""
import json

IMPORTMAP = """<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin><link rel="modulepreload" href="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js">
<style>#ld{position:absolute;inset:0;z-index:50;background:#0b0b0e;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;transition:opacity .6s;font:600 12px 'JetBrains Mono',monospace;letter-spacing:.2em;color:#f7931a}
#ld i{width:42px;height:42px;border:3px solid #2a2a35;border-top-color:#f7931a;border-radius:50%;animation:sp .8s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}</style>
<script>window.csReady=function(){var l=document.getElementById('ld');if(l){l.style.opacity=0;setTimeout(function(){l.remove()},650);}};</script>"""
LOADER = '<div id="ld"><i></i>LOADING 3D SCENE</div>'


FONTS = "<link href=\"https://fonts.googleapis.com/css2?family=Unbounded:wght@500;600&family=Manrope:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap\" rel=\"stylesheet\">"

# ----------------------------------------------------------------------------- tunnel
TUNNEL = """<!doctype html><html><head><meta charset="utf-8">__FONTS__ __IMPORTMAP__
<style>
html,body{margin:0;height:100%;background:#0b0b0e;overflow:hidden;font-family:'Manrope',sans-serif;color:#f2f2f5}
#c{position:absolute;inset:0;width:100%;height:100%;display:block}
.shade{position:absolute;inset:0;background:linear-gradient(90deg,rgba(11,11,14,.92) 0%,rgba(11,11,14,.55) 42%,rgba(11,11,14,0) 70%);pointer-events:none}
.copy{position:absolute;left:44px;top:50%;transform:translateY(-52%);max-width:600px;pointer-events:none}
.kick{font-family:'JetBrains Mono';font-size:12px;letter-spacing:.2em;color:#f7931a;font-weight:700;text-transform:uppercase}
h1{font-family:'Unbounded';font-size:34px;line-height:1.15;margin:12px 0 14px;letter-spacing:-.02em;
 background:linear-gradient(120deg,#fff 30%,#ffc857 70%,#f7931a);-webkit-background-clip:text;background-clip:text;color:transparent}
p{font-size:17px;line-height:1.5;color:#c9c9d4;margin:0 0 18px}
.badges span{display:inline-block;font-family:'JetBrains Mono';font-size:12px;padding:5px 11px;margin:0 8px 8px 0;border:1px solid #3a3a48;border-radius:6px;background:rgba(0,0,0,.4)}
.hud{position:absolute;right:28px;bottom:22px;text-align:right;font-family:'JetBrains Mono'}
.hud .s{font-size:34px;font-weight:700;letter-spacing:-.02em}
.hud .t{font-size:12px;letter-spacing:.14em;margin-top:2px}
.legend{position:absolute;left:44px;bottom:20px;font-size:12px;color:#9a9aa8;display:flex;gap:16px;font-family:'JetBrains Mono'}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.flash{position:absolute;inset:0;background:radial-gradient(circle at 50% 50%,rgba(255,71,87,.45),transparent 62%);opacity:0;pointer-events:none;transition:opacity .25s}
.hint{position:absolute;right:28px;top:20px;font-family:'JetBrains Mono';font-size:11px;color:#6d6d7c;letter-spacing:.12em}
</style></head><body>
__LOADER__<canvas id="c"></canvas><div class="shade"></div><div class="flash" id="flash"></div>
<div class="copy"><div class="kick">Intro to Data Science · Elliptic dataset</div>
<h1>Can a machine spot criminal Bitcoin transactions, and does it keep working when criminals change tactics?</h1>
<p>Fly through 49 blocks of time. Each ring is two weeks of activity, glowing by how well the model coped.</p>
<div class="badges"><span>203,769 transactions</span><span>165 features</span><span>leak-free temporal test</span></div></div>
<div class="legend"><span><i style="background:#5a5a78"></i>training</span><span><i style="background:#34d399"></i>healthy</span><span><i style="background:#ffc857"></i>degraded</span><span><i style="background:#ff4757"></i>failing</span></div>
<div class="hud"><div class="s" id="step">STEP 1</div><div class="t" id="phase">TRAINING</div></div>
<div class="hint">MOVE YOUR MOUSE TO STEER</div>
<script type="module">
import * as THREE from 'three';
const HEALTH = __HEALTH__;            // per step: 0 training, 1 healthy, 2 degraded, 3 failing, 4 too few cases
const COL = {0:0x6a6a90,1:0x34d399,2:0xffc857,3:0xff4757,4:0x8b8b9c};
const GAP = 2.4, N = HEALTH.length, L = N*GAP, R = 3.2;
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.5));
const scene = new THREE.Scene(); scene.fog = new THREE.FogExp2(0x0b0b0e, 0.042);
const camera = new THREE.PerspectiveCamera(72, 1, 0.1, 200);
const rings = [];
for (let i=0;i<N;i++){
  const g = new THREE.Group(); g.position.z = -i*GAP - 4;
  const c = COL[HEALTH[i]];
  const broken = (i===42);
  const arc = broken ? Math.PI*1.55 : Math.PI*2;
  const core = new THREE.Mesh(new THREE.TorusGeometry(R, broken?0.07:0.045, 10, 96, arc), new THREE.MeshBasicMaterial({color:c, transparent:true, opacity:HEALTH[i]===0?0.5:0.95}));
  const glow = new THREE.Mesh(new THREE.TorusGeometry(R, 0.22, 10, 96, arc), new THREE.MeshBasicMaterial({color:c, transparent:true, opacity:HEALTH[i]===0?0.05:0.14, blending:THREE.AdditiveBlending, depthWrite:false}));
  g.add(core); g.add(glow);
  if (broken){ for (let k=0;k<26;k++){ const s=new THREE.Mesh(new THREE.BoxGeometry(0.12,0.12,0.5*Math.random()+0.1), new THREE.MeshBasicMaterial({color:0xff4757,transparent:true,opacity:.85}));
      const a=Math.PI*1.55+Math.random()*Math.PI*0.45; const r=R+(Math.random()-.3)*1.6; s.position.set(Math.cos(a)*r,Math.sin(a)*r,(Math.random()-.5)*2.4); s.rotation.set(Math.random()*3,Math.random()*3,Math.random()*3); s.userData.v=Math.random(); g.add(s);} }
  scene.add(g); rings.push(g);
}
// faint longitudinal ribs
const rib = new THREE.BufferGeometry(); const rp=[];
for (let k=0;k<20;k++){ const a=k/20*Math.PI*2; rp.push(Math.cos(a)*R,Math.sin(a)*R,-2, Math.cos(a)*R,Math.sin(a)*R,-L-8); }
rib.setAttribute('position', new THREE.Float32BufferAttribute(rp,3));
scene.add(new THREE.LineSegments(rib, new THREE.LineBasicMaterial({color:0x2a2a3a, transparent:true, opacity:.55})));
// stream of transactions
const P = 3200, pos = new Float32Array(P*3), col = new Float32Array(P*3), spd = new Float32Array(P);
for (let i=0;i<P;i++){ const a=Math.random()*Math.PI*2, r=Math.sqrt(Math.random())*R*0.93;
  pos.set([Math.cos(a)*r, Math.sin(a)*r, -Math.random()*(L+10)], i*3); spd[i]=0.4+Math.random()*1.2;
  const bad = Math.random()<0.07; const c=new THREE.Color(bad?0xff4757:(Math.random()<0.5?0x7dd3fc:0xe9e9f2)); col.set([c.r,c.g,c.b], i*3); }
const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pos,3)); pg.setAttribute('color', new THREE.BufferAttribute(col,3));
const pts = new THREE.Points(pg, new THREE.PointsMaterial({size:0.075, vertexColors:true, transparent:true, opacity:.9, depthWrite:false, blending:THREE.AdditiveBlending}));
scene.add(pts);
function resize(){ const w=canvas.clientWidth,h=canvas.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize); resize();
let mx=0,my=0; window.addEventListener('mousemove',e=>{mx=(e.clientX/innerWidth-.5)*2; my=(e.clientY/innerHeight-.5)*2;});
const stepEl=document.getElementById('step'), phaseEl=document.getElementById('phase'), flash=document.getElementById('flash');
const NAMES={0:'TRAINING',1:'MODEL HEALTHY',2:'MODEL DEGRADED',3:'MODEL FAILING',4:'TOO FEW CASES'};
let t0=performance.now(), lastStep=0;
function frame(now){
  const t=(now-t0)/1000, speed=2.6; const z=-((t*speed)%(L+6));
  camera.position.set(mx*0.55+Math.sin(t*0.3)*0.25, -my*0.4+Math.cos(t*0.25)*0.2, z);
  camera.rotation.set(-my*0.05, -mx*0.12, Math.sin(t*0.2)*0.03);
  const idx=Math.max(0,Math.min(N-1,Math.floor((-z-4+GAP*0.5)/GAP)));
  if (idx!==lastStep){ lastStep=idx; stepEl.textContent='STEP '+(idx+1); const h=HEALTH[idx]; phaseEl.textContent=NAMES[h]; phaseEl.style.color='#'+COL[h].toString(16).padStart(6,'0');
     stepEl.style.color=(idx>=42&&idx<=44)?'#ff4757':'#f2f2f5'; if(idx===42){flash.style.opacity=1; setTimeout(()=>flash.style.opacity=0,500);} }
  const arr=pg.attributes.position.array; for(let i=0;i<P;i++){ arr[i*3+2]+=spd[i]*0.016*2.2; if(arr[i*3+2]>2) arr[i*3+2]-= (L+12); } pg.attributes.position.needsUpdate=true;
  rings[42].children.forEach((m,k)=>{ if(k>1){ m.rotation.x+=0.01*m.userData.v; m.position.x+=Math.sin(t*2+k)*0.003; } });
  rings.forEach((g,i)=>{ if(HEALTH[i]!==0){ g.children[1].material.opacity=0.12+0.07*Math.sin(t*2.2+i*0.5);} });
  renderer.render(scene,camera); requestAnimationFrame(frame);
}
requestAnimationFrame(frame); window.csReady();
</script></body></html>"""


def tunnel_html(health):
    return TUNNEL.replace("__FONTS__", FONTS).replace("__IMPORTMAP__", IMPORTMAP).replace("__LOADER__", LOADER).replace("__HEALTH__", json.dumps(health))


# ----------------------------------------------------------------------------- universe
UNIVERSE = """<!doctype html><html><head><meta charset="utf-8">__FONTS__ __IMPORTMAP__
<style>
html,body{margin:0;height:100%;background:#0b0b0e;overflow:hidden;font-family:'Manrope',sans-serif;color:#f2f2f5}
#c{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab}
.bar{position:absolute;left:16px;right:16px;top:14px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;pointer-events:none}
.bar > *{pointer-events:auto}
.seg{display:flex;border:1px solid #34344a;border-radius:10px;overflow:hidden;background:rgba(11,11,14,.78);backdrop-filter:blur(8px)}
.seg button{background:transparent;border:0;color:#b9b9c6;padding:8px 14px;font-family:'Unbounded';font-weight:600;font-size:13px;cursor:pointer}
.seg button.on{background:linear-gradient(135deg,#f7931a,#c46f08);color:#1a0f00}
.panel{border:1px solid #34344a;border-radius:10px;background:rgba(11,11,14,.78);backdrop-filter:blur(8px);padding:7px 12px;font-family:'JetBrains Mono';font-size:12px;display:flex;gap:10px;align-items:center}
input[type=range]{accent-color:#f7931a;width:230px}
button.pl{background:#f7931a;border:0;border-radius:6px;color:#1a0f00;font-weight:700;padding:4px 10px;cursor:pointer;font-family:'JetBrains Mono'}
.legend{position:absolute;left:16px;bottom:14px;display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#c9c9d4;font-family:'JetBrains Mono';background:rgba(11,11,14,.72);padding:8px 12px;border-radius:10px;border:1px solid #2a2a35}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.stats{position:absolute;right:16px;bottom:14px;text-align:right;font-family:'JetBrains Mono';font-size:12px;line-height:1.6;color:#c9c9d4;background:rgba(11,11,14,.72);padding:8px 12px;border-radius:10px;border:1px solid #2a2a35}
.stats b{color:#f2f2f5}
#tip{position:absolute;pointer-events:none;display:none;background:rgba(11,11,14,.94);border:1px solid #f7931a;border-radius:8px;padding:8px 11px;font-family:'JetBrains Mono';font-size:12px;line-height:1.5;white-space:nowrap}
</style></head><body>
__LOADER__<canvas id="c"></canvas>
<div class="bar">
 <div class="seg" id="modes"><button data-m="truth" class="on">Truth</button><button data-m="verdict">Model verdict</button><button data-m="period">Period</button></div>
 <div class="panel"><span>time travel</span><input id="rng" type="range" min="1" max="49" value="49"><b id="rv">≤ 49</b><button class="pl" id="play">▶ play</button></div>
 <div class="panel"><label><input type="checkbox" id="rot" checked> auto-rotate</label></div>
</div>
<div class="legend" id="legend"></div><div class="stats" id="stats"></div><div id="tip"></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__; const N = D.n, THR = D.thr;
const canvas=document.getElementById('c');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
const scene=new THREE.Scene(); scene.fog=new THREE.FogExp2(0x0b0b0e,0.012);
const camera=new THREE.PerspectiveCamera(55,1,0.1,400); camera.position.set(17,10,20);
const controls=new OrbitControls(camera,canvas); controls.enableDamping=true; controls.dampingFactor=0.06; controls.autoRotate=true; controls.autoRotateSpeed=0.7; controls.minDistance=6; controls.maxDistance=60;
const grid=new THREE.GridHelper(40,20,0x3a3a50,0x1c1c28); grid.position.y=-12; scene.add(grid);
const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(24,24,24)), new THREE.LineBasicMaterial({color:0x2a2a3a,transparent:true,opacity:.7})); scene.add(box);
const pos=new Float32Array(N*3), colA=new Float32Array(N*3), sizeA=new Float32Array(N), stepA=new Float32Array(N);
for(let i=0;i<N;i++){ pos.set([D.x[i],D.y[i],D.z[i]],i*3); stepA[i]=D.step[i]; }
const geo=new THREE.BufferGeometry(); geo.setAttribute('position',new THREE.BufferAttribute(pos,3)); geo.setAttribute('color',new THREE.BufferAttribute(colA,3));
geo.setAttribute('size',new THREE.BufferAttribute(sizeA,1)); geo.setAttribute('step',new THREE.BufferAttribute(stepA,1));
const mat=new THREE.ShaderMaterial({transparent:true,depthWrite:false,uniforms:{uMax:{value:49},uPx:{value:renderer.getPixelRatio()}},
 vertexShader:`attribute float size;attribute float step;attribute vec3 color;uniform float uMax;uniform float uPx;varying vec3 vC;varying float vOn;
  void main(){vC=color;vOn=step<=uMax+0.01?1.0:0.0;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=size*uPx*(260.0/-mv.z);gl_Position=projectionMatrix*mv;}`,
 fragmentShader:`varying vec3 vC;varying float vOn;void main(){if(vOn<0.5)discard;float d=length(gl_PointCoord-0.5);if(d>0.5)discard;float a=smoothstep(0.5,0.05,d);gl_FragColor=vec4(vC,a*0.92);}`});
const points=new THREE.Points(geo,mat); scene.add(points);
const C={licit:new THREE.Color(0x2dd4bf).multiplyScalar(0.55), ill:new THREE.Color(0xff4757), caught:new THREE.Color(0xffc857), missed:new THREE.Color(0xff4757), tn:new THREE.Color(0x2dd4bf).multiplyScalar(0.45), fa:new THREE.Color(0xb79bff),
  p1:new THREE.Color(0x6a7ab8), p2:new THREE.Color(0xffc857), p3:new THREE.Color(0xf7931a), p4:new THREE.Color(0xff4757)};
const LEG={truth:[['#2dd4bf','licit'],['#ff4757','illicit']],
 verdict:[['#2dd4bf','licit, judged licit'],['#ffc857','illicit, caught'],['#ff4757','illicit, MISSED'],['#b79bff','licit, false alarm']],
 period:[['#6a7ab8','steps 1-27 (train)'],['#ffc857','28-34 (validation)'],['#f7931a','35-42 (test)'],['#ff4757','43-49 (test, after drift)']]};
let mode='truth';
function paint(){
  for(let i=0;i<N;i++){ const s=D.step[i], y=D.label[i], p=D.p[i]; let c,sz;
    if(mode==='truth'){ c=y?C.ill:C.licit; sz=y?1.25:0.8; }
    else if(mode==='verdict'){ const flag=p>=THR; if(y&&flag){c=C.caught;sz=1.2;} else if(y&&!flag){c=C.missed;sz=1.7;} else if(!y&&flag){c=C.fa;sz=1.3;} else {c=C.tn;sz=0.7;} }
    else { c= s<=27?C.p1: s<=34?C.p2: s<=42?C.p3: C.p4; sz=y?1.2:0.8; }
    colA.set([c.r,c.g,c.b],i*3); sizeA[i]=sz; }
  geo.attributes.color.needsUpdate=true; geo.attributes.size.needsUpdate=true;
  document.getElementById('legend').innerHTML=LEG[mode].map(l=>`<span><i style="background:${l[0]}"></i>${l[1]}</span>`).join('');
  stats();
}
function stats(){ const m=+rng.value; let n=0,ill=0,caught=0,miss=0,tn=0,fa=0;
  for(let i=0;i<N;i++){ if(D.step[i]>m)continue; n++; const y=D.label[i],f=D.p[i]>=THR; if(y){ill++; if(D.step[i]>27){ if(f)caught++; else miss++; }} }
  document.getElementById('stats').innerHTML=`showing <b>${n}</b> sampled transactions (steps 1-${m})<br>illicit: <b>${ill}</b>`+(m>27?` · after training: caught <b>${caught}</b>, missed <b>${miss}</b>`:'')+`<br>PCA variance in 3 axes: <b>${(D.explained.reduce((a,b)=>a+b,0)*100).toFixed(0)}%</b>`; }
const rng=document.getElementById('rng'), rv=document.getElementById('rv');
rng.oninput=()=>{mat.uniforms.uMax.value=+rng.value; rv.textContent='≤ '+rng.value; stats();};
document.querySelectorAll('#modes button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#modes button').forEach(x=>x.classList.remove('on'));b.classList.add('on');mode=b.dataset.m;paint();});
document.getElementById('rot').onchange=e=>controls.autoRotate=e.target.checked;
let playing=false, timer=null; const pb=document.getElementById('play');
pb.onclick=()=>{ if(playing){clearInterval(timer);playing=false;pb.textContent='▶ play';return;} playing=true;pb.textContent='❚❚ pause'; if(+rng.value>=49)rng.value=1;
  timer=setInterval(()=>{ let v=+rng.value+1; if(v>49){clearInterval(timer);playing=false;pb.textContent='▶ play';return;} rng.value=v; rng.oninput(); },380); rng.oninput(); };
const ray=new THREE.Raycaster(); ray.params.Points.threshold=0.45; const mouse=new THREE.Vector2(); const tip=document.getElementById('tip');
canvas.addEventListener('mousemove',e=>{ const r=canvas.getBoundingClientRect(); mouse.set(((e.clientX-r.left)/r.width)*2-1,-((e.clientY-r.top)/r.height)*2+1); ray.setFromCamera(mouse,camera);
  const hits=ray.intersectObject(points).filter(h=>D.step[h.index]<=+rng.value); if(hits.length){ const i=hits[0].index; const y=D.label[i];
   tip.style.display='block'; tip.style.left=Math.min(e.clientX-r.left+14,r.width-230)+'px'; tip.style.top=(e.clientY-r.top+14)+'px';
   tip.innerHTML=`<b>step ${D.step[i]}</b> · ${y?'<span style="color:#ff4757">illicit</span>':'<span style="color:#2dd4bf">licit</span>'}<br>model P(illicit): <b>${(D.p[i]*100).toFixed(1)}%</b>`+(D.step[i]<=27?'<br><span style="color:#9a9aa8">in training data</span>':'<br>'+((D.p[i]>=THR)===!!y?'<span style="color:#34d399">model correct</span>':'<span style="color:#ffc857">model wrong</span>')); } else tip.style.display='none'; });
canvas.addEventListener('pointerdown',()=>canvas.style.cursor='grabbing'); addEventListener('pointerup',()=>canvas.style.cursor='grab');
function resize(){const w=canvas.clientWidth,h=canvas.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
addEventListener('resize',resize); resize(); paint(); window.csReady();
(function loop(){controls.update();renderer.render(scene,camera);requestAnimationFrame(loop);})();
</script></body></html>"""


def universe_html(data):
    return UNIVERSE.replace("__FONTS__", FONTS).replace("__IMPORTMAP__", IMPORTMAP).replace("__LOADER__", LOADER).replace("__DATA__", json.dumps(data, separators=(",", ":")))
