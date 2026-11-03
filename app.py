"""Industrial Robot Cloud Simulator - RoboDK-style offline programming (OLP).

EDGE COMPUTING DESIGN
  Python only (1) scans your assets/ folder, (2) builds a small JSON payload, (3) serves the page.
  All heavy work - kinematics, inverse kinematics, motion, 3D drawing, STL loading and webcam
  gesture tracking - runs in JavaScript inside the user's own browser (laptop or phone).
  Moving a slider never triggers a Streamlit rerun, so there is no server delay.

REPO LAYOUT (unchanged):  assets/robots/<ROBOT>/base_link.stl, link_1.stl ... link_6.stl
                          assets/robot_tools/<welding_guns|grippers|welding_torches>/*.stl
Only requirement: streamlit  (numpy / ikpy are no longer needed).
"""
import base64, json, re
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

SIM_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robot OLP Simulator</title>
<meta name="description" content="Offline robot programming simulator in the browser: jog, teach, simulate, check and export programs. For students and engineers.">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TransformControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
<style>
:root{--bg:#12151a;--p:#1a1e25;--p2:#222831;--line:#2d3541;--fg:#e7ebf1;--dim:#8f9bad;--ac:#ff9800;--ac2:#4da3ff;--ok:#4caf50;--bad:#f44336;--warn:#ffc107}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);font:13px/1.4 "Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;display:flex;flex-direction:column;overflow:hidden;user-select:none}
button,select,input{font:inherit;color:inherit}
button{cursor:pointer}
#menubar{display:flex;align-items:center;gap:4px;height:40px;padding:0 10px;background:var(--p);border-bottom:1px solid var(--line);flex:none}
.brand{font-weight:800;letter-spacing:.04em;color:var(--ac);margin-right:10px;white-space:nowrap}
.brand small{color:var(--dim);font-weight:500;letter-spacing:0;margin-left:6px}
.menu{position:relative}
.menu>button,.mb{background:none;border:0;padding:7px 10px;border-radius:6px;color:var(--fg)}
.menu>button:hover,.mb:hover{background:var(--p2)}
.drop{display:none;position:absolute;top:100%;left:0;min-width:210px;background:var(--p2);border:1px solid var(--line);border-radius:8px;padding:4px;z-index:50;box-shadow:0 10px 30px #0008}
.menu:hover .drop,.menu:focus-within .drop{display:block}
.drop button{display:flex;justify-content:space-between;gap:20px;width:100%;text-align:left;background:none;border:0;padding:7px 10px;border-radius:6px}
.drop button:hover{background:#2e3745}.drop kbd{color:var(--dim);font-size:11px}
.sp{flex:1}
#robotsel{background:var(--p2);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
#main{flex:1;display:grid;grid-template-columns:230px 1fr 360px;min-height:0}
aside{background:var(--p);overflow:auto;min-height:0}
#tree{border-right:1px solid var(--line);padding:8px}
#side{border-left:1px solid var(--line);display:flex;flex-direction:column}
.h{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:12px 4px 6px}
.node{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;cursor:pointer}
.node:hover{background:var(--p2)}
.node.sel{background:#2a3342;outline:1px solid #3b4a63}
.node .eye{margin-left:auto;opacity:.9}
.node.off{opacity:.45}
.node i{font-style:normal;width:18px;text-align:center}
.card{background:var(--p2);border:1px solid var(--line);border-radius:10px;padding:10px;margin:8px 0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px}.kv span:nth-child(odd){color:var(--dim)}
#view{position:relative;min-width:0;min-height:0;background:#0d0f13}
#cv{position:absolute;inset:0}
#vtools{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap;z-index:5}
.vb{background:#12161cdd;border:1px solid var(--line);border-radius:7px;padding:5px 9px;backdrop-filter:blur(4px)}
.vb:hover,.vb.on{border-color:var(--ac);color:var(--ac)}
#hud{position:absolute;left:10px;bottom:10px;z-index:5;display:flex;flex-direction:column;gap:6px;max-width:calc(100% - 20px)}
#toast{background:#12161cee;border:1px solid var(--line);border-left:4px solid var(--ac2);border-radius:8px;padding:8px 12px;max-width:520px;display:none}
#toast.ok{border-left-color:var(--ok)}#toast.bad{border-left-color:var(--bad)}#toast.warn{border-left-color:var(--warn)}
#flags{display:flex;gap:6px;flex-wrap:wrap}
.flag{padding:3px 9px;border-radius:999px;font-size:11px;background:#12161cdd;border:1px solid var(--line);color:var(--dim)}
.flag.ok{color:var(--ok);border-color:#2f5d33}.flag.bad{color:var(--bad);border-color:#7a2a26}.flag.warn{color:var(--warn);border-color:#6d5a14}
#ar{position:absolute;right:10px;bottom:10px;width:280px;height:210px;border:2px solid var(--ac);border-radius:8px;overflow:hidden;display:none;z-index:6;background:#000}
#ar video,#ar canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
#arhud{position:absolute;top:0;left:0;right:0;background:#000b;color:#fff;font-size:11px;text-align:center;padding:4px;z-index:3}
#tabs{display:flex;border-bottom:1px solid var(--line);flex:none}
#tabs button{flex:1;background:none;border:0;border-bottom:2px solid transparent;padding:10px 4px;color:var(--dim)}
#tabs button.on{color:var(--ac);border-bottom-color:var(--ac)}
#pane{flex:1;overflow:auto;padding:10px}
.seg{display:flex;gap:4px;margin:6px 0}
.seg button{flex:1;background:var(--p2);border:1px solid var(--line);border-radius:6px;padding:6px 4px;color:var(--dim)}
.seg button.on{background:var(--ac);border-color:var(--ac);color:#111;font-weight:700}
.btn{background:var(--p2);border:1px solid var(--line);border-radius:7px;padding:7px 10px}
.btn:hover{border-color:var(--ac)}
.btn.pri{background:var(--ac);border-color:var(--ac);color:#111;font-weight:700}
.btn.ok{background:var(--ok);border-color:var(--ok);color:#fff;font-weight:700}
.btn.bad{background:#3a2020;border-color:#7a2a26}
.btn:disabled{opacity:.4;cursor:not-allowed}
.row{display:flex;gap:6px;align-items:center;margin:5px 0;flex-wrap:wrap}
.row>.btn{flex:1}
.jr{display:grid;grid-template-columns:28px 1fr 64px;gap:6px;align-items:center;margin:6px 0}
.jr b{color:var(--ac2);font-family:monospace}
.jr input[type=range]{width:100%;accent-color:var(--ac)}
.num{background:#0e1116;border:1px solid var(--line);border-radius:5px;padding:3px 5px;width:100%;font-family:monospace;color:#8be9d0}
.bar{height:4px;background:#0e1116;border-radius:3px;margin-top:2px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.lbl{font-size:11px;color:var(--dim);display:block;margin-bottom:2px}
select.s{width:100%;background:var(--p2);border:1px solid var(--line);border-radius:6px;padding:6px}
table.prog{width:100%;border-collapse:collapse}
table.prog td{padding:5px 4px;border-bottom:1px solid var(--line);vertical-align:middle}
table.prog tr{cursor:pointer}table.prog tr:hover{background:#222a35}
table.prog tr.cur{background:#3a2f12;outline:1px solid var(--ac)}
table.prog tr.err td:nth-child(2){color:var(--bad)}
table.prog tr.wrn td:nth-child(2){color:var(--warn)}
.mini{background:none;border:1px solid var(--line);border-radius:5px;padding:0 6px;color:var(--dim)}
.mini:hover{color:var(--fg);border-color:var(--ac)}
.tl{height:8px;background:#0e1116;border-radius:4px;overflow:hidden;margin:8px 0}.tl i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--ac2),var(--ac))}
details{background:var(--p2);border:1px solid var(--line);border-radius:8px;margin:6px 0;padding:6px 10px}
summary{cursor:pointer;font-weight:600}
details p{margin:6px 0;color:#c4ccd9}
.spot{display:flex;align-items:center;gap:8px;padding:4px 0}
.dot{width:12px;height:12px;border-radius:50%;background:#555}.dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
.note{color:var(--dim);font-size:12px}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;background:#2b3442;color:#cdd6e3}
#status{display:flex;gap:16px;align-items:center;height:28px;padding:0 12px;background:var(--p);border-top:1px solid var(--line);font:12px monospace;color:var(--dim);flex:none;overflow:hidden;white-space:nowrap}
#status b{color:#8be9d0;font-weight:500}
#bsheet{display:none}
@media (max-width:1000px){
  #main{grid-template-columns:1fr;grid-template-rows:55% 45%}
  #tree{display:none}#side{border-left:0;border-top:1px solid var(--line)}
  #status{font-size:11px;gap:10px}.brand small{display:none}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>

<div id="menubar">
  <div class="brand">ROBOT OLP<small>offline programming simulator</small></div>
  <div class="menu"><button>File</button><div class="drop">
    <button data-a="new">New station</button>
    <button data-a="open">Open station…</button>
    <button data-a="save">Save station (.json)</button>
    <button data-a="exrapid">Export program: ABB RAPID</button>
    <button data-a="exkrl">Export program: KUKA KRL</button>
    <button data-a="extxt">Export program: plain text</button>
  </div></div>
  <div class="menu"><button>Program</button><div class="drop">
    <button data-a="rec">Record target <kbd>R</kbd></button>
    <button data-a="play">Run / pause <kbd>Space</kbd></button>
    <button data-a="stop">Stop <kbd>Esc</kbd></button>
    <button data-a="check">Check program</button>
    <button data-a="auto">Auto-teach the weld mission</button>
  </div></div>
  <div class="menu"><button>View</button><div class="drop">
    <button data-a="v-iso">Isometric <kbd>1</kbd></button>
    <button data-a="v-front">Front <kbd>2</kbd></button>
    <button data-a="v-side">Side <kbd>3</kbd></button>
    <button data-a="v-top">Top <kbd>4</kbd></button>
    <button data-a="v-fit">Fit to robot <kbd>F</kbd></button>
  </div></div>
  <div class="menu"><button>Help</button><div class="drop">
    <button data-a="help">Quick start</button>
  </div></div>
  <div class="sp"></div>
  <label class="note" for="robotsel">Robot</label>
  <select id="robotsel" aria-label="Robot model"></select>
</div>

<div id="main">
  <aside id="tree" aria-label="Station tree"></aside>
  <div id="view">
    <canvas id="cv"></canvas>
    <div id="vtools">
      <button class="vb" data-v="iso">Iso</button><button class="vb" data-v="front">Front</button><button class="vb" data-v="side">Side</button><button class="vb" data-v="top">Top</button><button class="vb" data-v="fit">Fit</button>
      <button class="vb" id="b-home" title="Move to the home position (H)">Home</button>
    </div>
    <div id="hud"><div id="toast" role="status"></div><div id="flags"></div></div>
    <div id="ar"><div id="arhud">PLACE HAND INSIDE THE BOX</div><video id="cam" autoplay playsinline muted></video><canvas id="arc" width="280" height="210"></canvas></div>
  </div>
  <aside id="side">
    <div id="tabs"></div>
    <div id="pane"></div>
  </aside>
</div>
<div id="status"></div>
<input type="file" id="fopen" accept=".json" hidden>
<input type="file" id="fstl" accept=".stl" hidden>

<script>
(() => {
'use strict';
const T = THREE, D2R = Math.PI / 180, R2D = 180 / Math.PI, PI = Math.PI;
(T.Object3D.DefaultUp || T.Object3D.DEFAULT_UP).set(0, 0, 1);

/* ============================================================
   1. ROBOT REGISTRY (educational kinematic models, metres and radians)
   t = link translation, o = link orientation (XYZ euler), a = joint axis
   ============================================================ */
const L = (t, o, a) => ({ t, o, a });
const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1], O = [0, 0, 0];
const ROBOTS = {
  ABB_6700: { name: 'ABB 6700', color: 0xe8e9ec, accent: 0xd8321e, reach: 2.65,
    links: [L([0, 0, .78], O, Z), L([.32, 0, 0], O, Y), L([0, 0, 1.1], O, Y), L([1.142, 0, .2], O, X), L([.2, 0, 0], O, Y), L([.2, 0, 0], O, X)],
    lim: [[-PI, PI], [-1.047, 1.483], [-1.3962, 1.3962], [-6.108, 6.108], [-2.181, 2.181], [-6.108, 6.108]] },
  ABB_4400: { name: 'ABB 4400', color: 0xe8e9ec, accent: 0xd8321e, reach: 1.95,
    links: [L([0, 0, 0], O, Z), L([.2, 0, .6], O, Y), L([0, 0, .88], O, Y), L([2.15, 0, .15], O, X), L([-1.0, 0, 0], O, Y), L([.14, 0, 0], O, X)],
    lim: [[-PI, PI], [-1.221, 1.658], [-1.3962, 1.3962], [-5.235, 5.235], [-2.094, 2.094], [-6.981, 6.981]] },
  ABB_6600: { name: 'ABB 6600', color: 0xe8e9ec, accent: 0xd8321e, reach: 2.55,
    links: [L([0, 0, .22], O, Z), L([.3, 0, .5], O, Y), L([0, -.2, 1.145], O, Y), L([1.145, .2, .2], O, X), L([.65, 0, 0], O, Y), L([.18, 0, 0], O, X)],
    lim: [[-PI, PI], [-1.134, 1.483], [-1.3962, 1.3962], [-5.235, 5.235], [-2.094, 2.094], [-5.235, 5.235]] },
  KUKA_KR150: { name: 'KUKA KR150', color: 0xff6a00, accent: 0x333a44, reach: 3.1,
    links: [L([0, 0, .55], O, Z), L([.35, 0, 0], [0, -1.5708, 0], Y), L([1.3, 0, -.05], [0, 1.5708, 0], Y), L([2.4, 0, .1], O, X), L([-1.0, 0, 0], O, Y), L([.21, 0, 0], O, X)],
    lim: [[-3.228, 3.228], [-.785, 1.483], [-1.3962, 1.3962], [-6.108, 6.108], [-2.181, 2.181], [-6.108, 6.108]] },
  Yaskawa_3500: { name: 'Yaskawa 3500', color: 0x1f5fe0, accent: 0xdfe4ec, reach: 2.45,
    links: [L([0, 0, .5], O, Z), L([.16, 0, 0], O, Y), L([0, 0, .9], O, Y), L([0, 0, .21], O, X), L([1.0, 0, 0], [0, -1.5708, 0], Y), L([0, 0, -.17], O, X)],
    lim: [[-3.141, 3.141], [-1.57, 1.308], [-1.3962, 1.3962], [-3.141, 3.141], [-2.268, 2.268], [-6.283, 6.283]] }
};
/* ---- Optional payload injected by app.py (Streamlit). null when opened standalone. ---- */
const PAYLOAD = /*__PAYLOAD__*/null;
if (PAYLOAD && PAYLOAD.robots) for (const k in PAYLOAD.robots) {
  const r = PAYLOAD.robots[k], base = ROBOTS[k] || {};
  ROBOTS[k] = Object.assign({ name: k, color: 0xe8e9ec, accent: 0x333a44, reach: 2.5, links: base.links, lim: base.lim }, base, r);
}
const TOOLLIB = (PAYLOAD && PAYLOAD.tools) || [];
const E1_LIM = [-2 * PI, 2 * PI];
const HOME = [0, 0, 0, 0, 0.5, 0];

/* ============================================================
   2. STATE
   ============================================================ */
const defaults = () => ({
  robot: 'ABB_6700', q: HOME.slice(), e1: 0,
  base: { x: 0, y: 0, z: 0, yaw: 0 },
  tool: { type: 'gun', tcp: [0, 0, 300], rot: [0, 0, 0], mrot: [0, 0, 0], mscale: 0.001, stl: '' },
  jig: { type: 'table', x: 1.6, y: 0, z: 0.55, rx: 0, ry: 0, scale: 0.001, stl: '' },
  prog: [], speedJ: 90, speedL: 500, loop: false
});
let ST = defaults();
const VIS = { robot: true, tool: true, jig: true, tcp: true, spots: true, env: false, targets: true, grid: true };
let mode = 'joint', tab = 'jog', selNode = 'robot', toolOpen = 0, welded = [false, false, false, false], missionLog = [];
let issues = [], cycleTime = 0, ikErr = { p: 0, r: 0 }, flagsNow = {};

/* ============================================================
   3. SCENE
   ============================================================ */
const cvs = document.getElementById('cv'), viewEl = document.getElementById('view');
const renderer = new T.WebGLRenderer({ canvas: cvs, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new T.Scene(); scene.background = new T.Color(0x0d0f13);
const camera = new T.PerspectiveCamera(42, 1, 0.02, 200); camera.position.set(4.6, -4.6, 3.2);
const controls = new T.OrbitControls(camera, cvs); controls.target.set(1, 0, 0.9); controls.enableDamping = true; controls.dampingFactor = 0.12;
scene.add(new T.HemisphereLight(0xffffff, 0x3a3f4a, 0.95));
const dl = new T.DirectionalLight(0xffffff, 0.85); dl.position.set(5, -4, 9); scene.add(dl);
const dl2 = new T.DirectionalLight(0x9fc0ff, 0.35); dl2.position.set(-6, 5, 4); scene.add(dl2);
const grid = new T.GridHelper(16, 32, 0x4a5465, 0x232a35); grid.rotation.x = PI / 2; scene.add(grid);
scene.add(new T.AxesHelper(0.5));
const stlLoader = new T.STLLoader();

const robotRoot = new T.Group(); scene.add(robotRoot);
const frames = []; for (let i = 0; i < 7; i++) { const f = new T.Object3D(); f.matrixAutoUpdate = false; robotRoot.add(f); frames.push(f); }   // siblings: each matrix is absolute in robot coordinates
const toolRoot = new T.Object3D(); toolRoot.matrixAutoUpdate = false; frames[6].add(toolRoot);
const toolModel = new T.Group(); toolRoot.add(toolModel);
const tcpFrame = new T.Object3D(); toolRoot.add(tcpFrame);
const tcpAxes = new T.AxesHelper(0.14); tcpFrame.add(tcpAxes);
const jigRoot = new T.Group(); scene.add(jigRoot);
const jigStatic = new T.Group(), jigRot = new T.Group(); jigRoot.add(jigStatic, jigRot);
const spotGroup = new T.Group(); jigRot.add(spotGroup);
const envPts = new T.Points(new T.BufferGeometry(), new T.PointsMaterial({ color: 0x4da3ff, size: 0.025, transparent: true, opacity: 0.35, depthWrite: false })); robotRoot.add(envPts); envPts.visible = false;
const targetGroup = new T.Group(); scene.add(targetGroup);
const traceMax = 4000, tracePos = new Float32Array(traceMax * 3);
const traceGeo = new T.BufferGeometry(); traceGeo.setAttribute('position', new T.BufferAttribute(tracePos, 3)); traceGeo.setDrawRange(0, 0);
const traceLine = new T.Line(traceGeo, new T.LineBasicMaterial({ color: 0xff9800 })); traceLine.frustumCulled = false; scene.add(traceLine);
let traceN = 0;

const gizmo = new T.TransformControls(camera, cvs); gizmo.setSize ? gizmo.setSize(0.7) : (gizmo.size = 0.7);
const gizmoTarget = new T.Object3D(); scene.add(gizmoTarget); gizmo.attach(gizmoTarget);
scene.add(gizmo.isObject3D ? gizmo : gizmo.getHelper());
gizmo.visible = false; gizmo.enabled = false;
gizmo.addEventListener('dragging-changed', e => { controls.enabled = !e.value; if (!e.value) snapGizmo(); });
gizmo.addEventListener('objectChange', () => { if (mode !== 'tcp' || runner.on) return; gizmoTarget.updateMatrixWorld(true); solveToTarget(gizmoTarget.matrixWorld, {}); updateAll(false); });

/* ---------- helpers ---------- */
const mats = {};
const M = (c, o = {}) => { const k = c + JSON.stringify(o); return mats[k] || (mats[k] = new T.MeshStandardMaterial(Object.assign({ color: c, roughness: 0.45, metalness: 0.25 }, o))); };
function box(parent, sx, sy, sz, x, y, z, mat) { const m = new T.Mesh(new T.BoxGeometry(sx, sy, sz), mat); m.position.set(x, y, z); parent.add(m); return m; }
function cylAxis(parent, r, len, axis, pos, mat, seg = 28) { const g = new T.CylinderGeometry(r, r, len, seg); const m = new T.Mesh(g, mat); m.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), new T.Vector3(...axis).normalize()); m.position.set(...pos); parent.add(m); return m; }
function clearGroup(g) { while (g.children.length) { const c = g.children.pop(); c.traverse(o => { if (o.geometry) o.geometry.dispose(); }); } }
function labelSprite(text, color = '#ffd27a') {
  const c = document.createElement('canvas'); c.width = 128; c.height = 48; const x = c.getContext('2d');
  x.font = 'bold 30px Segoe UI, Arial'; x.fillStyle = 'rgba(10,12,16,.75)'; x.fillRect(0, 0, 128, 48); x.fillStyle = color; x.textAlign = 'center'; x.textBaseline = 'middle'; x.fillText(text, 64, 26);
  const s = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), depthTest: false, transparent: true })); s.scale.set(0.2, 0.075, 1); s.renderOrder = 5; return s;
}

/* ---------- build robot visuals ---------- */
const GEOCACHE = {};
let cur = ROBOTS[ST.robot], S = [];   // S[i] static matrix of link i
function buildRobot() {
  cur = ROBOTS[ST.robot];
  frames[6].remove(toolRoot); frames.forEach(f => clearGroup(f)); frames[6].add(toolRoot);
  S = cur.links.map(l => new T.Matrix4().makeTranslation(...l.t).multiply(new T.Matrix4().makeRotationFromEuler(new T.Euler(l.o[0], l.o[1], l.o[2], 'XYZ'))));
  const body = M(cur.color, { roughness: 0.35 }), joint = M(cur.accent, { roughness: 0.4, metalness: 0.5 }), dark = M(0x2a2f38);
  const th = [0.42, 0.3, 0.26, 0.2, 0.15, 0.1];
  if (cur.meshes && cur.meshes.some(Boolean)) {      // real CAD meshes from the user's repo (loaded in the browser = edge computing)
    const sc = cur.mscale || 0.001, mc = M(cur.color, { roughness: 0.4, metalness: 0.3 });
    cur.meshes.forEach((src, i) => {
      if (!src) return;
      const done = g => { const m = new T.Mesh(g, i === 0 ? M(0x4a5260) : mc); m.scale.setScalar(sc);
        const off = (cur.moff && cur.moff[i]) || [0, 0, 0], rot = (cur.mrot && cur.mrot[i]) || [0, 0, 0];
        m.position.set(off[0], off[1], off[2]); m.rotation.set(rot[0] * D2R, rot[1] * D2R, rot[2] * D2R); frames[i].add(m); };
      const key = ST.robot + ':' + src;
      if (GEOCACHE[key]) done(GEOCACHE[key]);
      else stlLoader.load(src, g => { GEOCACHE[key] = g; if (cur === ROBOTS[ST.robot]) done(g); }, undefined, () => toast('Mesh not found: ' + src, 'bad'));
    });
    cylAxis(frames[6], 0.001, 0.001, X, [0, 0, 0], dark);
    return;
  }
  // base
  cylAxis(frames[0], 0.42, 0.06, Z, [0, 0, 0.03], dark, 40);
  // The origin of the roll joint A4 is only a point on its axis, so the forearm is drawn straight from A3 to A5 (t4 + t5 along the roll axis).
  const vecs = cur.links.map(l => l.t.slice()); vecs[2] = [0, 1, 2].map(k => cur.links[3].t[k] + cur.links[4].t[k]); vecs[3] = null;
  for (let i = 0; i <= 5; i++) {
    const v = vecs[i]; if (!v) continue;
    const pts2 = [[0, 0, 0], [v[0], 0, 0], [v[0], v[1], 0], [v[0], v[1], v[2]]];
    for (let k = 0; k < 3; k++) {
      const a = pts2[k], b = pts2[k + 1], d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], len = Math.hypot(...d); if (len < 1e-4) continue;
      const t = th[i] * (i === 0 ? 0.9 : 1), cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2, cz = (a[2] + b[2]) / 2;
      box(frames[i], Math.abs(d[0]) > 0 ? len + t * 0.5 : t, Math.abs(d[1]) > 0 ? len + t * 0.5 : t, Math.abs(d[2]) > 0 ? len + t * 0.5 : t, cx, cy, cz, body);
    }
  }
  for (let i = 1; i <= 6; i++) {
    const ax = cur.links[i - 1].a, r = [0.2, 0.17, 0.15, 0.12, 0.1, 0.07][i - 1], len = r * 1.7;
    if (i === 4) { const e = vecs[2]; cylAxis(frames[3], r, len, ax, [e[0] - 0.12, e[1], e[2]], joint); continue; }   // roll housing sits at the end of the forearm
    cylAxis(frames[i], r, len, ax, [0, 0, 0], joint);
  }
  cylAxis(frames[6], 0.07, 0.03, X, [0.02, 0, 0], dark);
}
function toolMountMatrix() { return new T.Matrix4().makeRotationY(PI / 2); }   // tool +Z points along flange +X
let TCPL = new T.Matrix4();
function buildTool() {
  clearGroup(toolModel); const t = ST.tool; toolRoot.matrix.copy(toolMountMatrix());
  const steel = M(0xaab2bd, { metalness: 0.7, roughness: 0.3 }), dk = M(0x39414d), red = M(0xd8321e), cu = M(0xc27a45, { metalness: 0.8, roughness: 0.3 });
  toolModel.userData = {};
  if (t.type === 'gun') {
    box(toolModel, .1, .12, .1, 0, 0, .05, dk); box(toolModel, .035, .05, .22, .085, 0, .2, steel); box(toolModel, .035, .05, .22, -.085, 0, .2, steel); box(toolModel, .2, .05, .04, 0, 0, .09, steel);
    const e1 = new T.Group(), e2 = new T.Group(); toolModel.add(e1, e2);
    cylAxis(e1, .014, .05, X, [-.045, 0, .3], cu, 16); cylAxis(e2, .014, .05, X, [.045, 0, .3], cu, 16);
    toolModel.userData.e1 = e1; toolModel.userData.e2 = e2;
  } else if (t.type === 'gripper') {
    box(toolModel, .1, .12, .07, 0, 0, .035, dk); box(toolModel, .15, .03, .03, 0, 0, .08, steel);
    const f1 = new T.Group(), f2 = new T.Group(); toolModel.add(f1, f2);
    box(f1, .02, .035, .1, 0, 0, .13, red); box(f2, .02, .035, .1, 0, 0, .13, red);
    toolModel.userData.f1 = f1; toolModel.userData.f2 = f2;
  } else if (t.type === 'torch') {
    box(toolModel, .08, .08, .06, 0, 0, .03, dk); cylAxis(toolModel, .022, .22, Z, [0, 0, .17], steel); cylAxis(toolModel, .03, .08, Z, [0, 0, .3], dk); cylAxis(toolModel, .008, .06, Z, [0, 0, .35], cu, 12);
  } else if (t.type === 'lib' && t.url) {
    stlLoader.load(t.url, g => { g.center(); const m = new T.Mesh(g, M(0xdadfe6, { roughness: 0.4 })); m.scale.setScalar(t.mscale || 0.001);
      m.rotation.set(t.mrot[0] * D2R, t.mrot[1] * D2R, t.mrot[2] * D2R); toolModel.add(m); }, undefined, () => toast('Tool mesh not found', 'bad'));
  } else if (t.type === 'stl' && t.stl) {
    try {
      const g = stlLoader.parse(b64ToBuf(t.stl)); g.center(); const m = new T.Mesh(g, M(0xdadfe6, { roughness: 0.4 })); m.scale.setScalar(t.mscale || 0.001);
      m.rotation.set(t.mrot[0] * D2R, t.mrot[1] * D2R, t.mrot[2] * D2R); toolModel.add(m);
    } catch (e) { toast('Could not read the tool STL file', 'bad'); }
  }
  tcpFrame.position.set(t.tcp[0] / 1000, t.tcp[1] / 1000, t.tcp[2] / 1000);
  tcpFrame.rotation.set(t.rot[0] * D2R, t.rot[1] * D2R, t.rot[2] * D2R, 'XYZ'); tcpFrame.updateMatrix();
  TCPL = toolMountMatrix().multiply(new T.Matrix4().compose(tcpFrame.position.clone(), new T.Quaternion().setFromEuler(tcpFrame.rotation), new T.Vector3(1, 1, 1)));
  toolRoot.visible = VIS.tool; tcpAxes.visible = VIS.tcp; toolRoot.updateMatrixWorld(true);
}
function workAxisWorld(q = ST.q) { const m = robotWorld().clone().multiply(fkAll(q, [])[6]).multiply(toolMountMatrix()); return toolWorkAxisLocal().transformDirection(m); }
function toolWorkAxisLocal() { return ST.tool.type === 'gun' ? new T.Vector3(1, 0, 0) : new T.Vector3(0, 0, 1); }
function animateTool() {
  const u = toolModel.userData;
  if (u.e1) { const g = 0.025 * (1 - toolOpen); u.e1.position.x = g; u.e2.position.x = -g; }
  if (u.f1) { const g = 0.012 + 0.036 * (1 - toolOpen); u.f1.position.x = -g; u.f2.position.x = g; }
}
function b64ToBuf(b) { const s = atob(b), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a.buffer; }
function bufToB64(buf) { let s = ''; const a = new Uint8Array(buf); for (let i = 0; i < a.length; i += 8192) s += String.fromCharCode.apply(null, a.subarray(i, i + 8192)); return btoa(s); }

/* ---------- jig / positioner ---------- */
const SPOTS = [[0.28, 0.17], [0.28, -0.17], [-0.28, -0.17], [-0.28, 0.17]];
const spotMeshes = [];
function buildJig() {
  clearGroup(jigStatic); clearGroup(jigRot); spotGroup.children.length = 0; spotMeshes.length = 0; jigRot.add(spotGroup);
  const j = ST.jig; jigRoot.position.set(j.x, j.y, j.z);
  if (j.type === 'table') {
    cylAxis(jigStatic, 0.26, j.z - 0.06, Z, [0, 0, -(j.z - 0.06) / 2 - 0.06], M(0x3b4452), 32);
    cylAxis(jigStatic, 0.34, 0.06, Z, [0, 0, -0.03], M(0x59687d), 36);
    box(jigRot, 0.8, 0.55, 0.03, 0, 0, 0.015, M(0x6b7b92));
    box(jigRot, 0.7, 0.45, 0.004, 0, 0, 0.032, M(0xb9c2cf, { metalness: 0.6, roughness: 0.35 }));
    box(jigRot, 0.05, 0.05, 0.12, -0.36, 0.25, 0.09, M(0xd8321e));
    SPOTS.forEach((p, i) => {
      const ring = new T.Mesh(new T.RingGeometry(0.014, 0.024, 28), new T.MeshBasicMaterial({ color: 0xff9800, side: T.DoubleSide }));
      ring.position.set(p[0], p[1], 0.036); spotGroup.add(ring); spotMeshes.push(ring);
      const s = labelSprite(String(i + 1)); s.position.set(p[0], p[1], 0.11); spotGroup.add(s);
    });
    paintSpots();
  } else if (j.type === 'stl' && j.stl) {
    try {
      const g = stlLoader.parse(b64ToBuf(j.stl)); g.center(); const m = new T.Mesh(g, M(0x00bcd4, { transparent: true, opacity: 0.75 }));
      m.scale.setScalar(j.scale || 0.001); m.rotation.x = j.rx * D2R; m.rotation.y = j.ry * D2R; jigRot.add(m);
    } catch (e) { toast('Could not read the jig STL file', 'bad'); }
  }
  jigRoot.visible = VIS.jig; spotGroup.visible = VIS.spots && j.type === 'table';
}
function paintSpots() { spotMeshes.forEach((m, i) => { m.material.color.set(welded[i] ? 0x4caf50 : 0xff9800); m.scale.setScalar(welded[i] ? 1.3 : 1); }); }

/* ============================================================
   4. KINEMATICS: forward, inverse (damped least squares)
   ============================================================ */
const _ax = [X, Y, Z].map(a => new T.Vector3(...a));
function fkAll(q, out) {
  const F = out || [];
  F[0] = F[0] || new T.Matrix4(); F[0].identity();
  const r = new T.Matrix4();
  for (let i = 0; i < 6; i++) {
    F[i + 1] = F[i + 1] || new T.Matrix4();
    F[i + 1].copy(F[i]).multiply(S[i]).multiply(r.makeRotationAxis(new T.Vector3(...cur.links[i].a), q[i]));
  }
  return F;
}
const _F = [];
function tcpLocal(q) { return fkAll(q, _F)[6].clone().multiply(TCPL); }
function robotWorld() { robotRoot.updateMatrixWorld(true); return robotRoot.matrixWorld; }
function tcpWorld(q) { return robotWorld().clone().multiply(tcpLocal(q)); }
function rotVec(qa, qb) { const d = qa.clone().multiply(qb.clone().invert()); if (d.w < 0) { d.x = -d.x; d.y = -d.y; d.z = -d.z; d.w = -d.w; } const l = Math.hypot(d.x, d.y, d.z); if (l < 1e-9) return new T.Vector3(0, 0, 0); return new T.Vector3(d.x, d.y, d.z).multiplyScalar(2 * Math.atan2(l, d.w) / l); }
function inv6(A) {
  const n = 6, a = A.map((r, i) => r.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    if (Math.abs(a[p][c]) < 1e-12) return null; [a[c], a[p]] = [a[p], a[c]];
    const d = a[c][c]; for (let k = 0; k < 2 * n; k++) a[c][k] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = a[r][c]; if (f) for (let k = 0; k < 2 * n; k++) a[r][k] -= f * a[c][k]; }
  }
  return a.map(r => r.slice(n));
}
const clampQ = (q) => { let hit = false; for (let i = 0; i < 6; i++) { const [lo, hi] = cur.lim[i]; if (q[i] < lo) { q[i] = lo; hit = true; } else if (q[i] > hi) { q[i] = hi; hit = true; } } return hit; };
/* Solve so that the TCP reaches targetWorld (Matrix4). opts: orient (default true), lock (joint indices), iter */
function solveToTarget(targetWorld, opts = {}, q = ST.q) {
  const wr = opts.orient === false ? 0 : 0.35, lock = opts.lock || [], iters = opts.iter || 30, lam = 0.05;
  const Tt = robotWorld().clone().invert().multiply(targetWorld);
  const tp = new T.Vector3().setFromMatrixPosition(Tt), tq = new T.Quaternion().setFromRotationMatrix(Tt);
  let pe = 0, re = 0, limHit = false;
  const poseOf = qq => { const m = tcpLocal(qq); return [new T.Vector3().setFromMatrixPosition(m), new T.Quaternion().setFromRotationMatrix(m)]; };
  for (let it = 0; it < iters; it++) {
    const [cp, cq] = poseOf(q), ep = tp.clone().sub(cp), er = rotVec(tq, cq);
    pe = ep.length(); re = er.length();
    if (pe < 2e-4 && (wr === 0 || re < 1e-3)) break;
    const e = [ep.x, ep.y, ep.z, er.x * wr, er.y * wr, er.z * wr], J = Array.from({ length: 6 }, () => new Array(6).fill(0));
    for (let j = 0; j < 6; j++) {
      if (lock.includes(j)) continue; const h = 1e-4; q[j] += h; const [p2, q2] = poseOf(q); q[j] -= h;
      const dp = p2.sub(cp).multiplyScalar(1 / h), dr = rotVec(q2, cq).multiplyScalar(wr / h);
      J[0][j] = dp.x; J[1][j] = dp.y; J[2][j] = dp.z; J[3][j] = dr.x; J[4][j] = dr.y; J[5][j] = dr.z;
    }
    const A = Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, k) => { let s = 0; for (let j = 0; j < 6; j++) s += J[i][j] * J[k][j]; return s + (i === k ? lam * lam : 0); }));
    const Ai = inv6(A); if (!Ai) break;
    const y = Ai.map(r => r.reduce((s, v, k) => s + v * e[k], 0));
    for (let j = 0; j < 6; j++) { if (lock.includes(j)) continue; let d = 0; for (let i = 0; i < 6; i++) d += J[i][j] * y[i]; q[j] += Math.max(-0.2, Math.min(0.2, d)); }
    if (clampQ(q)) limHit = true;
  }
  const [cp, cq] = poseOf(q); pe = tp.distanceTo(cp); re = rotVec(tq, cq).length();
  ikErr = { p: pe, r: wr ? re : 0, lim: limHit };
  return ikErr;
}
function currentTCP() { return tcpWorld(ST.q); }
function snapGizmo() { const m = currentTCP(); m.decompose(gizmoTarget.position, gizmoTarget.quaternion, gizmoTarget.scale); gizmoTarget.updateMatrixWorld(true); }
function wristSingular() { return Math.abs(ST.q[4]) < 6 * D2R; }
function reachRadius() { const m = tcpLocal(ST.q), p = new T.Vector3().setFromMatrixPosition(m); const o = new T.Vector3().setFromMatrixPosition(S[0]); return p.distanceTo(o); }

/* reach envelope */
let envKey = '';
function buildEnvelope() {
  if (envKey === ST.robot) return; envKey = ST.robot; const n = 7000, pos = new Float32Array(n * 3), q = new Array(6), F = []; let maxR = 0;
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < 6; i++) { const [lo, hi] = cur.lim[i]; q[i] = lo + Math.random() * (hi - lo); if (i > 3) q[i] = 0; }
    const m = fkAll(q, F)[4]; pos[k * 3] = m.elements[12]; pos[k * 3 + 1] = m.elements[13]; pos[k * 3 + 2] = m.elements[14];
    maxR = Math.max(maxR, Math.hypot(m.elements[12], m.elements[13], m.elements[14] - cur.links[0].t[2]));
  }
  envPts.geometry.setAttribute('position', new T.BufferAttribute(pos, 3)); envPts.geometry.computeBoundingSphere(); cur.estReach = maxR;
}

/* ============================================================
   5. PROGRAM: targets, runner, analysis, export
   ============================================================ */
const KIND = { MoveJ: 'Joint move', MoveL: 'Linear move', Weld: 'Spot weld', Open: 'Open tool', Close: 'Close tool', Wait: 'Wait', Jig: 'Rotate jig' };
const ICON = { MoveJ: '⤳', MoveL: '→', Weld: '⚡', Open: '◌', Close: '●', Wait: '⏱', Jig: '⟳' };
function addInstr(kind, extra = {}) {
  const n = ST.prog.filter(p => p.q).length + 1, ins = Object.assign({ k: kind, sp: 100 }, extra);
  if (['MoveJ', 'MoveL', 'Weld'].includes(kind)) { ins.q = ST.q.slice(); ins.e1 = ST.e1; ins.name = (kind === 'Weld' ? 'W' : 'T') + n; }
  if (kind === 'Wait') ins.t = extra.t || 1;
  if (kind === 'Jig') ins.e1 = extra.e1 != null ? extra.e1 : ST.e1;
  ST.prog.push(ins); refreshTargets(); renderPane(); saveLocal(); return ins;
}
function refreshTargets() {
  clearGroup(targetGroup); const pts = [];
  ST.prog.forEach((p, i) => {
    if (!p.q) return; const m = tcpWorld(p.q), pos = new T.Vector3().setFromMatrixPosition(m); pts.push(pos);
    const ax = new T.AxesHelper(0.07); m.decompose(ax.position, ax.quaternion, ax.scale); ax.scale.set(1, 1, 1); targetGroup.add(ax);
    const s = labelSprite(p.name || '', p.k === 'Weld' ? '#7dffa0' : '#ffd27a'); s.position.copy(pos).add(new T.Vector3(0, 0, 0.1)); targetGroup.add(s);
  });
  if (pts.length > 1) { const g = new T.BufferGeometry().setFromPoints(pts), l = new T.Line(g, new T.LineDashedMaterial({ color: 0x4da3ff, dashSize: 0.05, gapSize: 0.03 })); l.computeLineDistances(); targetGroup.add(l); }
  targetGroup.visible = VIS.targets;
}
const runner = { on: false, paused: false, i: 0, t: 0, dur: 0, ph: 'move', plan: null, issueNow: null };
const smooth = u => u * u * (3 - 2 * u);
function planMove(ins, fromQ, fromE1) {
  const sp = Math.max(1, ins.sp || 100) / 100, jv = ST.speedJ * D2R * sp, lv = ST.speedL / 1000 * sp;
  const toQ = ins.q, toE = ins.e1 != null ? ins.e1 : fromE1; let dur;
  const pa = tcpWorld(fromQ), pb = tcpWorld(toQ), A = new T.Vector3().setFromMatrixPosition(pa), B = new T.Vector3().setFromMatrixPosition(pb);
  const qa = new T.Quaternion().setFromRotationMatrix(pa), qb = new T.Quaternion().setFromRotationMatrix(pb);
  if (ins.k === 'MoveJ') { let d = Math.abs(toE - fromE1); for (let i = 0; i < 6; i++) d = Math.max(d, Math.abs(toQ[i] - fromQ[i])); dur = d / jv; }
  else { dur = Math.max(A.distanceTo(B) / lv, (qa.angleTo(qb)) / jv, Math.abs(toE - fromE1) / jv); }
  return { dur: Math.max(0.08, dur), fromQ: fromQ.slice(), toQ: toQ.slice(), fromE1, toE, A, B, qa, qb, linear: ins.k !== 'MoveJ' };
}
function stepMove(pl, u, q) {
  const s = smooth(u);
  if (!pl.linear) { for (let i = 0; i < 6; i++) q[i] = pl.fromQ[i] + (pl.toQ[i] - pl.fromQ[i]) * s; return { p: 0, r: 0 }; }
  const pos = pl.A.clone().lerp(pl.B, s), quat = pl.qa.clone().slerp(pl.qb, s), m = new T.Matrix4().compose(pos, quat, new T.Vector3(1, 1, 1));
  return solveToTarget(m, { iter: 10 }, q);
}
function startRun() {
  if (!ST.prog.length) { toast('Record at least one step first (Program tab, or press R).', 'warn'); return; }
  if (!ST.prog.some(p => p.q)) { toast('The program has no robot targets yet.', 'warn'); return; }
  runner.on = true; runner.paused = false; runner.i = -1; runner.t = 0; runner.dur = 0; runner.ph = 'next'; traceN = 0; traceGeo.setDrawRange(0, 0);
  welded = [false, false, false, false]; missionLog = []; paintSpots(); toolOpen = 1; setMode('joint', true); renderPane(); toast('Running program…', 'ok');
}
function stopRun(msg) { runner.on = false; runner.paused = false; renderPane(); if (msg) toast(msg, 'ok'); }
function advance(dt) {
  if (!runner.on || runner.paused) return;
  const spd = +document.getElementById('simspd')?.value || 1; dt *= spd; runner.t += dt;
  for (let guard = 0; guard < 5; guard++) {
    if (runner.ph === 'next') {
      runner.i++; if (runner.i >= ST.prog.length) { if (ST.loop) { runner.i = 0; welded = [false, false, false, false]; paintSpots(); } else { stopRun('Program finished'); return; } }
      const ins = ST.prog[runner.i]; runner.t = 0; runner.ph = 'move'; runner.act = null;
      if (ins.q) { runner.plan = planMove(ins, ST.q, ST.e1); runner.dur = runner.plan.dur; }
      else if (ins.k === 'Wait') runner.dur = ins.t; else if (ins.k === 'Jig') { runner.plan = { fromE1: ST.e1, toE: ins.e1 }; runner.dur = Math.max(0.1, Math.abs(ins.e1 - ST.e1) / (60 * D2R * Math.max(0.01, ins.sp / 100))); }
      else runner.dur = 0.45;
      renderProgRows(); if (!ins.q && ins.k !== 'Wait' && ins.k !== 'Jig') { runner.fromOpen = toolOpen; }
    }
    const ins = ST.prog[runner.i], u = Math.min(1, runner.t / runner.dur);
    if (runner.ph === 'move') {
      if (ins.q) { const e = stepMove(runner.plan, u, ST.q); ST.e1 = runner.plan.fromE1 + (runner.plan.toE - runner.plan.fromE1) * smooth(u); if (e.p > 0.006 || (e.lim && runner.plan.linear)) runner.issueNow = 'Step ' + (runner.i + 1) + ': the straight path cannot be followed here'; }
      else if (ins.k === 'Jig') ST.e1 = runner.plan.fromE1 + (runner.plan.toE - runner.plan.fromE1) * smooth(u);
      else if (ins.k === 'Open') toolOpen = runner.fromOpen + (1 - runner.fromOpen) * u; else if (ins.k === 'Close') toolOpen = runner.fromOpen + (0 - runner.fromOpen) * u;
      if (u >= 1) { if (ins.k === 'Weld') { runner.ph = 'weld'; runner.t = 0; runner.dur = 0.9; runner.fromOpen = toolOpen; } else { runner.ph = 'next'; } }
      else break;
    } else if (runner.ph === 'weld') {
      const uu = Math.min(1, runner.t / runner.dur); toolOpen = uu < 0.3 ? runner.fromOpen * (1 - uu / 0.3) : uu < 0.7 ? 0 : (uu - 0.7) / 0.3;
      if (!runner.welded && uu >= 0.4) { runner.welded = true; tryWeld(true); }
      if (uu >= 1) { runner.welded = false; toolOpen = 1; runner.ph = 'next'; } else break;
    }
    if (runner.ph !== 'next') break;
  }
  if (runner.on && traceN < traceMax) { const p = new T.Vector3().setFromMatrixPosition(tcpWorld(ST.q)); tracePos.set([p.x, p.y, p.z], traceN * 3); traceN++; traceGeo.setDrawRange(0, traceN); traceGeo.attributes.position.needsUpdate = true; }
}
function tryWeld(fromRun) {
  if (ST.jig.type !== 'table') { toast('Spot weld done (the mission needs the default jig)', 'ok'); return; }
  const tw = tcpWorld(ST.q), tp = new T.Vector3().setFromMatrixPosition(tw); jigRoot.updateMatrixWorld(true);
  let best = -1, bd = 1e9; spotMeshes.forEach((m, i) => { const w = new T.Vector3().setFromMatrixPosition(m.matrixWorld), d = w.distanceTo(tp); if (d < bd) { bd = d; best = i; } });
  const axis = workAxisWorld(), tilt = Math.acos(Math.max(-1, Math.min(1, axis.dot(new T.Vector3(0, 0, -1))))) * R2D;
  const mm = (bd * 1000).toFixed(0);
  if (bd < 0.02 && tilt < 25) { if (!welded[best]) { welded[best] = true; paintSpots(); } missionLog.push('Spot ' + (best + 1) + ' welded: ' + mm + ' mm off, tool tilt ' + tilt.toFixed(0) + '°'); toast('Weld OK on spot ' + (best + 1) + ': ' + mm + ' mm from the centre, tool tilt ' + tilt.toFixed(0) + '°.' + (welded.every(Boolean) ? ' Mission complete!' : ''), 'ok'); }
  else { const why = bd >= 0.02 ? 'the tool is ' + mm + ' mm from the nearest spot (limit 20 mm)' : 'the tool is tilted ' + tilt.toFixed(0) + '° from vertical (limit 25°)'; missionLog.push('Missed: ' + why); toast('Weld missed: ' + why + '.', 'bad'); }
  if (!fromRun) renderPane(); else if (tab === 'learn') renderPane();
}
function analyze() {
  const save = { q: ST.q.slice(), e1: ST.e1 }, q = ST.q.slice(), out = []; let e1 = ST.e1, total = 0, ok = true;
  const saveErr = ikErr;
  ST.prog.forEach((ins, i) => {
    const row = { lvl: 0, msg: '' };
    if (ins.q) {
      for (let j = 0; j < 6; j++) { const [lo, hi] = cur.lim[j]; if (ins.q[j] < lo - 1e-6 || ins.q[j] > hi + 1e-6) { row.lvl = 2; row.msg = 'Joint A' + (j + 1) + ' outside its limit'; } }
      const pl = planMove(ins, q, e1); total += pl.dur; let worst = 0, sing = false, lim = false; const n = Math.max(8, Math.ceil(pl.dur * 20)), qq = q.slice();
      for (let k = 1; k <= n; k++) { const e = stepMove(pl, k / n, qq); worst = Math.max(worst, e.p); if (e.lim && pl.linear) lim = true; if (pl.linear && Math.abs(qq[4]) < 4 * D2R) sing = true; }
      if (pl.linear && worst > 0.004 && !row.lvl) { row.lvl = 2; row.msg = 'Straight path not reachable (' + (worst * 1000).toFixed(0) + ' mm error)'; }
      else if (sing && !row.lvl) { row.lvl = 1; row.msg = 'Passes near a wrist singularity (A5 ≈ 0)'; }
      else if (lim && !row.lvl) { row.lvl = 1; row.msg = 'A joint reaches its limit on the way'; }
      for (let j = 0; j < 6; j++) q[j] = ins.q[j]; e1 = ins.e1 != null ? ins.e1 : e1;
    } else if (ins.k === 'Wait') total += ins.t; else if (ins.k === 'Jig') { total += Math.max(0.1, Math.abs(ins.e1 - e1) / (60 * D2R * Math.max(0.01, ins.sp / 100))); e1 = ins.e1; } else total += ins.k === 'Weld' ? 0.9 : 0.45;
    if (ins.k === 'Weld') total += 0.9;
    out.push(row);
  });
  ST.q = save.q; ST.e1 = save.e1; ikErr = saveErr; issues = out; cycleTime = total; renderProgRows(); updateAll(true);
  const bad = out.filter(r => r.lvl === 2).length, warn = out.filter(r => r.lvl === 1).length;
  toast(bad ? bad + ' step(s) cannot be reached. See the red rows.' : warn ? 'Program is reachable, with ' + warn + ' warning(s).' : 'Program is reachable. Estimated cycle time ' + cycleTime.toFixed(1) + ' s.', bad ? 'bad' : warn ? 'warn' : 'ok');
}
function autoTeach() {
  if (ST.jig.type !== 'table') { toast('Auto-teach needs the default jig (Layout tab).', 'warn'); return; }
  stopRun(); ST.prog = []; welded = [false, false, false, false]; missionLog = []; paintSpots();
  const q = HOME.slice(); ST.q = q; ST.e1 = 0; jigRoot.updateMatrixWorld(true);
  const down = alignDownMatrix(tcpWorld(q)); let failed = 0;
  const poseAt = (p, dz) => { const m = down.clone(); m.setPosition(p.x, p.y, p.z + dz); return m; };
  const spotsW = spotMeshes.map(m => new T.Vector3().setFromMatrixPosition(m.matrixWorld));
  // approach in a joint move first, then linear moves
  let first = true;
  spotsW.forEach((p, i) => {
    const hover = solveToTarget(poseAt(p, 0.12), { iter: 80 }, q);
    if (hover.p > 0.003) failed++;
    addInstr(first ? 'MoveJ' : 'MoveL', { q: q.slice(), e1: 0, name: 'A' + (i + 1) }); first = false;
    const hit = solveToTarget(poseAt(p, 0), { iter: 60 }, q); if (hit.p > 0.003) failed++;
    ST.prog.push({ k: 'Weld', sp: 60, q: q.slice(), e1: 0, name: 'W' + (i + 1) });
    solveToTarget(poseAt(p, 0.12), { iter: 40 }, q); ST.prog.push({ k: 'MoveL', sp: 100, q: q.slice(), e1: 0, name: 'R' + (i + 1) });
  });
  ST.q = HOME.slice(); refreshTargets(); updateAll(true); renderPane(); saveLocal();
  toast(failed ? 'Auto-teach finished, but ' + failed + ' point(s) are out of reach for this robot or layout. Try the ABB 6700 or move the jig closer.' : 'Program taught: 4 spots × (approach, weld, retreat). Press Run, or Check program first.', failed ? 'warn' : 'ok');
}
function alignDownMatrix(m) {
  const pos = new T.Vector3(), q = new T.Quaternion(), s = new T.Vector3(); m.decompose(pos, q, s);
  const ax = workAxisWorld(ST.q);   // current work axis (world)
  const rot = new T.Quaternion().setFromUnitVectors(ax, new T.Vector3(0, 0, -1)); return new T.Matrix4().compose(pos, rot.multiply(q), new T.Vector3(1, 1, 1));
}
/* exports */
function fmt(n, d = 2) { return n.toFixed(d); }
function exportProgram(kind) {
  const L = [], pr = ST.prog, rb = robotRoot.matrixWorld.clone().invert(), nm = ST.robot;
  const pose = q => { const m = rb.clone().multiply(tcpWorld(q)); const p = new T.Vector3().setFromMatrixPosition(m), qu = new T.Quaternion().setFromRotationMatrix(m), e = new T.Euler().setFromRotationMatrix(m, 'ZYX'); return { p, qu, e }; };
  const sp = i => Math.max(5, Math.round(ST.speedL * (pr[i].sp || 100) / 100));
  if (kind === 'rapid') {
    L.push('MODULE OlpExport', '  ! Exported from Robot OLP Simulator for ' + nm + '. Teaching export: check configuration data, tool data and speeds before running on a real robot.', '  PERS tooldata tTool := [TRUE,[[' + ST.tool.tcp.map(v => fmt(v, 1)).join(',') + '],[1,0,0,0]],[1,[0,0,100],[1,0,0,0],0,0,0]];', '  PROC main()');
    pr.forEach((p, i) => {
      if (p.k === 'MoveJ') L.push('    MoveAbsJ [[' + p.q.map(v => fmt(v * R2D, 2)).join(',') + '],[' + fmt((p.e1 || 0) * R2D, 2) + ',9E9,9E9,9E9,9E9,9E9]],v' + Math.round(ST.speedJ * (p.sp / 100)) + ',fine,tTool; ! ' + p.name);
      else if (p.k === 'MoveL' || p.k === 'Weld') { const o = pose(p.q); L.push('    MoveL [[' + [o.p.x, o.p.y, o.p.z].map(v => fmt(v * 1000, 1)).join(',') + '],[' + [o.qu.w, o.qu.x, o.qu.y, o.qu.z].map(v => fmt(v, 5)).join(',') + '],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]],v' + sp(i) + ',fine,tTool; ! ' + p.name); if (p.k === 'Weld') L.push('    SetDO doWeld,1; WaitTime 0.9; SetDO doWeld,0;'); }
      else if (p.k === 'Open') L.push('    SetDO doTool,1;'); else if (p.k === 'Close') L.push('    SetDO doTool,0;');
      else if (p.k === 'Wait') L.push('    WaitTime ' + p.t + ';'); else if (p.k === 'Jig') L.push('    ! rotate positioner axis E1 to ' + fmt(p.e1 * R2D, 1) + ' deg');
    });
    L.push('  ENDPROC', 'ENDMODULE'); return L.join('\n');
  }
  if (kind === 'krl') {
    L.push('&ACCESS RVP', 'DEF OLP_EXPORT()', '; Exported from Robot OLP Simulator for ' + nm + '. Teaching export: check tool/base data, $VEL and the joint order before running on a real robot.', '$TOOL={X ' + fmt(ST.tool.tcp[0], 1) + ',Y ' + fmt(ST.tool.tcp[1], 1) + ',Z ' + fmt(ST.tool.tcp[2], 1) + ',A 0,B 0,C 0}', '$BASE={X 0,Y 0,Z 0,A 0,B 0,C 0}');
    pr.forEach((p, i) => {
      if (p.k === 'MoveJ') L.push('$VEL_AXIS[1]=' + Math.round(p.sp) + ' ; ' + p.name, 'PTP {' + p.q.map((v, k) => 'A' + (k + 1) + ' ' + fmt(v * R2D, 2)).join(',') + ',E1 ' + fmt((p.e1 || 0) * R2D, 2) + '}');
      else if (p.k === 'MoveL' || p.k === 'Weld') { const o = pose(p.q); L.push('$VEL.CP=' + fmt(sp(i) / 1000, 3), 'LIN {X ' + fmt(o.p.x * 1000, 1) + ',Y ' + fmt(o.p.y * 1000, 1) + ',Z ' + fmt(o.p.z * 1000, 1) + ',A ' + fmt(o.e.z * R2D, 2) + ',B ' + fmt(o.e.y * R2D, 2) + ',C ' + fmt(o.e.x * R2D, 2) + '} ; ' + p.name); if (p.k === 'Weld') L.push('$OUT[1]=TRUE', 'WAIT SEC 0.9', '$OUT[1]=FALSE'); }
      else if (p.k === 'Open') L.push('$OUT[2]=TRUE'); else if (p.k === 'Close') L.push('$OUT[2]=FALSE'); else if (p.k === 'Wait') L.push('WAIT SEC ' + p.t); else if (p.k === 'Jig') L.push('; rotate positioner axis E1 to ' + fmt(p.e1 * R2D, 1) + ' deg');
    });
    L.push('END'); return L.join('\n');
  }
  L.push('Robot OLP Simulator program', 'Robot: ' + nm, 'Tool: ' + ST.tool.type + ', TCP (mm) ' + ST.tool.tcp.join(', '), 'Estimated cycle time: ' + cycleTime.toFixed(1) + ' s', '');
  pr.forEach((p, i) => L.push((i + 1) + '. ' + KIND[p.k] + (p.name ? ' ' + p.name : '') + (p.q ? '  joints(deg) ' + p.q.map(v => fmt(v * R2D, 1)).join(' ') : '') + (p.k === 'Wait' ? ' ' + p.t + ' s' : '') + (p.k === 'Jig' ? ' to ' + fmt(p.e1 * R2D, 0) + '°' : '') + '  speed ' + p.sp + '%'));
  return L.join('\n');
}
function download(name, text, type = 'text/plain') { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500); }

/* ============================================================
   6. UI
   ============================================================ */
const $ = s => document.querySelector(s), pane = $('#pane');
let toastTimer;
function toast(msg, kind = '') { const t = $('#toast'); t.textContent = msg; t.className = kind; t.style.display = 'block'; clearTimeout(toastTimer); toastTimer = setTimeout(() => t.style.display = 'none', 6000); }
const TABS = [['jog', 'Jog'], ['prog', 'Program'], ['layout', 'Tool & Layout'], ['learn', 'Learn']];
function renderTabs() { $('#tabs').innerHTML = TABS.map(([id, n]) => '<button data-tab="' + id + '" class="' + (tab === id ? 'on' : '') + '">' + n + '</button>').join(''); }
$('#tabs').onclick = e => { const b = e.target.closest('button'); if (!b) return; tab = b.dataset.tab; renderTabs(); renderPane(); };

/* ---- tree ---- */
const NODES = [['robot', '🤖', 'Robot'], ['tool', '🔧', 'Tool'], ['jig', '⚙', 'Jig and positioner'], ['tcp', '⌖', 'TCP frame'], ['spots', '◎', 'Weld spots'], ['targets', '➤', 'Targets and path'], ['env', '⬡', 'Reach envelope'], ['grid', '▦', 'Floor grid']];
function renderTree() {
  const r = cur;
  $('#tree').innerHTML = '<div class="h">Station</div>' + NODES.map(([id, ic, n]) => '<div class="node' + (selNode === id ? ' sel' : '') + (VIS[id] ? '' : ' off') + '" data-n="' + id + '"><i>' + ic + '</i>' + n + '<span class="eye" data-eye="' + id + '" title="Show or hide">' + (VIS[id] ? '👁' : '—') + '</span></div>').join('') +
    '<div class="h">Selected robot</div><div class="card"><div class="kv"><span>Model</span><span>' + r.name + '</span><span>Axes</span><span>6 + E1 positioner</span><span>Reach</span><span>' + (r.estReach ? r.estReach.toFixed(2) + ' m (sampled)' : '~' + r.reach + ' m') + '</span><span>Base</span><span>' + ST.base.x.toFixed(2) + ', ' + ST.base.y.toFixed(2) + ' m</span></div></div>' +
    '<div class="h">Programs</div><div class="node sel" data-go="prog"><i>▶</i>Main program<span class="eye pill">' + ST.prog.length + '</span></div>' +
    '<p class="note" style="padding:6px">Educational models. Link shapes are simplified; use for learning, not for cell sign-off.</p>';
}
$('#tree').onclick = e => {
  const eye = e.target.closest('[data-eye]'); if (eye) { const id = eye.dataset.eye; VIS[id] = !VIS[id]; applyVis(); renderTree(); return; }
  const g = e.target.closest('[data-go]'); if (g) { tab = g.dataset.go; renderTabs(); renderPane(); return; }
  const n = e.target.closest('[data-n]'); if (n) { selNode = n.dataset.n; if (selNode === 'tool') tab = 'layout'; else if (selNode === 'jig') tab = 'layout'; else if (selNode === 'spots') tab = 'learn'; else if (selNode === 'robot') tab = 'jog'; renderTabs(); renderPane(); renderTree(); }
};
function applyVis() {
  robotRoot.children.forEach(c => { if (frames.includes(c)) c.visible = VIS.robot; });
  toolRoot.visible = VIS.tool; tcpAxes.visible = VIS.tcp; jigRoot.visible = VIS.jig; spotGroup.visible = VIS.spots && ST.jig.type === 'table'; targetGroup.visible = VIS.targets; grid.visible = VIS.grid;
  if (VIS.env) buildEnvelope(); envPts.visible = VIS.env;
}

/* ---- panes ---- */
function renderPane() {
  if (tab === 'jog') paneJog(); else if (tab === 'prog') paneProg(); else if (tab === 'layout') paneLayout(); else paneLearn();
  syncUI();
}
function seg(items, cur, attr) { return '<div class="seg">' + items.map(([v, n]) => '<button data-' + attr + '="' + v + '" class="' + (cur === v ? 'on' : '') + '">' + n + '</button>').join('') + '</div>'; }
function paneJog() {
  let h = seg([['joint', 'Joint'], ['tcp', '⌖ TCP'], ['gesture', '✋ Gesture']], mode, 'mode');
  if (mode === 'joint') {
    h += '<div class="note">Drag a slider, type an angle, or use ± buttons (step <select id="jstep" class="num" style="width:62px"><option>1</option><option selected>5</option><option>15</option></select>°).</div>';
    for (let i = 0; i < 6; i++) h += '<div class="jr"><b>A' + (i + 1) + '</b><div><input type="range" data-j="' + i + '" min="' + (cur.lim[i][0] * R2D).toFixed(1) + '" max="' + (cur.lim[i][1] * R2D).toFixed(1) + '" step="0.1"><div class="bar"><i id="bar' + i + '"></i></div></div><input class="num" data-jn="' + i + '" type="number" step="0.5"></div>';
    h += '<div class="jr"><b style="color:var(--ac)">E1</b><div><input type="range" data-j="e1" min="-360" max="360" step="0.5"></div><input class="num" data-jn="e1" type="number" step="1"></div>';
    h += '<div class="row"><button class="btn" data-step="-1">− step (selected)</button><button class="btn" data-step="1">+ step</button></div>';
  } else if (mode === 'tcp') {
    h += '<div class="note">Drag the arrows or rings on the 3D tool point, or type a pose. Position in mm (world frame), angles in degrees (Z-Y-X).</div>';
    h += '<div class="grid3">' + ['X', 'Y', 'Z'].map((a, i) => '<div><span class="lbl">' + a + ' mm</span><input class="num" data-p="' + i + '" type="number" step="10"></div>').join('') + ['Rx', 'Ry', 'Rz'].map((a, i) => '<div><span class="lbl">' + a + ' °</span><input class="num" data-p="' + (i + 3) + '" type="number" step="5"></div>').join('') + '</div>';
    h += seg([['translate', 'Move'], ['rotate', 'Rotate']], gizmo.mode, 'gm');
    h += '<div class="row"><button class="btn" id="b-down">Point tool down</button><button class="btn" id="b-reach">Go to weld spot 1</button></div>';
  } else {
    h += '<div class="note">Hold your palm inside the box for 2 seconds. Move your palm to move the tool, roll your hand to turn the last joint, pinch thumb and index finger to record a step. Needs a webcam and an internet connection for the hand-tracking model. Video stays in your browser.</div>';
    h += '<div class="row"><button class="btn pri" id="b-cam">Start camera</button></div>';
  }
  h += '<div class="row"><button class="btn" id="b-rec">● Record step (R)</button><button class="btn" id="b-weldnow">⚡ Try a weld here</button></div>';
  h += '<div class="row"><button class="btn" id="b-open">Open tool</button><button class="btn" id="b-close">Close tool</button></div>';
  pane.innerHTML = h;
}
function paneProg() {
  const sp = ST.speedJ, lv = ST.speedL;
  let h = '<div class="row"><button class="btn ' + (runner.on && !runner.paused ? '' : 'ok') + '" id="p-play">' + (runner.on ? (runner.paused ? '▶ Resume' : '❚❚ Pause') : '▶ Run') + '</button><button class="btn bad" id="p-stop">■ Stop</button><button class="btn" id="p-check">✔ Check</button></div>';
  h += '<div class="tl"><i id="tlbar"></i></div><div class="note" id="progInfo"></div>';
  h += '<div class="row"><button class="btn" data-add="MoveJ">+ MoveJ</button><button class="btn" data-add="MoveL">+ MoveL</button><button class="btn" data-add="Weld">+ Weld</button></div>';
  h += '<div class="row"><button class="btn" data-add="Open">+ Open</button><button class="btn" data-add="Close">+ Close</button><button class="btn" data-add="Wait">+ Wait</button><button class="btn" data-add="Jig">+ Jig</button></div>';
  h += '<table class="prog"><tbody id="progRows"></tbody></table>';
  h += '<div class="h">Speeds</div><div class="grid3"><div><span class="lbl">Joint °/s</span><input class="num" id="s-j" type="number" value="' + sp + '"></div><div><span class="lbl">Linear mm/s</span><input class="num" id="s-l" type="number" value="' + lv + '"></div><div><span class="lbl">Sim speed</span><select id="simspd" class="num"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div></div>';
  h += '<label class="note"><input type="checkbox" id="p-loop"' + (ST.loop ? ' checked' : '') + '> Loop the program</label>';
  h += '<div class="h">Export and share</div><div class="row"><button class="btn" data-ex="rapid">ABB RAPID</button><button class="btn" data-ex="krl">KUKA KRL</button><button class="btn" data-ex="txt">Text</button><button class="btn" data-a2="save">Save station</button></div>';
  h += '<p class="note">Exports are teaching formats: configuration data, tool and speed data must be checked before use on a real robot.</p>';
  pane.innerHTML = h; renderProgRows(); const s = $('#simspd'); if (s) s.value = String(window.__spd || 1);
}
function renderProgRows() {
  const tb = $('#progRows'); if (!tb) return;
  tb.innerHTML = ST.prog.map((p, i) => {
    const is = issues[i], cls = (runner.on && runner.i === i ? 'cur ' : '') + (is && is.lvl === 2 ? 'err' : is && is.lvl === 1 ? 'wrn' : '');
    const d = p.k === 'Wait' ? p.t + ' s' : p.k === 'Jig' ? (p.e1 * R2D).toFixed(0) + '°' : p.name || '';
    return '<tr class="' + cls + '" data-i="' + i + '" title="' + (is && is.msg ? is.msg : 'Double-click to move the robot to this step') + '"><td>' + (i + 1) + '</td><td>' + ICON[p.k] + ' ' + p.k + (is && is.msg ? ' ⚠' : '') + '</td><td>' + d + '</td><td><input class="num" style="width:60px" data-sp="' + i + '" type="number" min="5" max="200" value="' + p.sp + '" title="Speed %"></td><td style="white-space:nowrap"><button class="mini" data-up="' + i + '">↑</button> <button class="mini" data-dn="' + i + '">↓</button> ' + (p.q ? '<button class="mini" data-upd="' + i + '" title="Overwrite with the current pose">⌖</button> ' : '') + '<button class="mini" data-del="' + i + '">✕</button></td></tr>';
  }).join('') || '<tr><td class="note" style="padding:14px">No steps yet. Jog the robot, then press <b>Record</b>, or use <b>Auto-teach</b> (Program menu) to see an example.</td></tr>';
  const info = $('#progInfo'); if (info) info.textContent = ST.prog.length + ' steps' + (cycleTime ? ' · estimated cycle time ' + cycleTime.toFixed(1) + ' s' : '') + (issues.some(r => r.lvl === 2) ? ' · unreachable steps found' : '');
}
function paneLayout() {
  const t = ST.tool, j = ST.jig, b = ST.base;
  let h = '<div class="h" style="margin-top:0">Tool</div><select class="s" id="t-type">' + [['none', 'None'], ['gun', 'Spot welding gun'], ['gripper', 'Gripper'], ['torch', 'Welding torch'], ['stl', 'Upload STL tool…']].concat(TOOLLIB.map((x, i) => ['lib' + i, '📦 ' + x.name])).map(([v, n]) => '<option value="' + v + '"' + ((t.type === v || (t.type === 'lib' && v === 'lib' + t.libi)) ? ' selected' : '') + '>' + n + '</option>').join('') + '</select>';
  if (t.type === 'stl') h += '<div class="row"><button class="btn" id="t-file">Choose STL file</button></div><div class="grid3"><div><span class="lbl">Scale</span><input class="num" id="t-sc" type="number" step="0.0005" value="' + t.mscale + '"></div></div><div class="grid3">' + ['X', 'Y', 'Z'].map((a, i) => '<div><span class="lbl">Model rot ' + a + ' °</span><input class="num" data-mr="' + i + '" type="number" step="90" value="' + t.mrot[i] + '"></div>').join('') + '</div>';
  h += '<div class="h">Tool centre point (TCP)</div><div class="grid3">' + ['X', 'Y', 'Z'].map((a, i) => '<div><span class="lbl">' + a + ' mm</span><input class="num" data-tcp="' + i + '" type="number" step="5" value="' + t.tcp[i] + '"></div>').join('') + ['Rx', 'Ry', 'Rz'].map((a, i) => '<div><span class="lbl">' + a + ' °</span><input class="num" data-tr="' + i + '" type="number" step="5" value="' + t.rot[i] + '"></div>').join('') + '</div>';
  h += '<p class="note">The TCP is the point the robot moves. For the spot gun it sits between the electrode tips.</p>';
  h += '<div class="h">Jig / positioner (axis E1)</div><select class="s" id="j-type">' + [['table', 'Rotary table with 4 weld spots'], ['stl', 'Upload STL jig…'], ['none', 'None']].map(([v, n]) => '<option value="' + v + '"' + (j.type === v ? ' selected' : '') + '>' + n + '</option>').join('') + '</select>';
  if (j.type === 'stl') h += '<div class="row"><button class="btn" id="j-file">Choose STL file</button></div><div class="grid3"><div><span class="lbl">Scale</span><input class="num" id="j-sc" type="number" step="0.0005" value="' + j.scale + '"></div><div><span class="lbl">Rot X °</span><input class="num" id="j-rx" type="number" step="90" value="' + j.rx + '"></div><div><span class="lbl">Rot Y °</span><input class="num" id="j-ry" type="number" step="90" value="' + j.ry + '"></div></div>';
  h += '<div class="grid3">' + ['X', 'Y', 'Z'].map((a, i) => '<div><span class="lbl">Jig ' + a + ' m</span><input class="num" data-jp="' + i + '" type="number" step="0.05" value="' + [j.x, j.y, j.z][i] + '"></div>').join('') + '</div>';
  h += '<div class="h">Robot base position</div><div class="grid3"><div><span class="lbl">X m</span><input class="num" data-bp="x" type="number" step="0.1" value="' + b.x + '"></div><div><span class="lbl">Y m</span><input class="num" data-bp="y" type="number" step="0.1" value="' + b.y + '"></div><div><span class="lbl">Yaw °</span><input class="num" data-bp="yaw" type="number" step="15" value="' + b.yaw + '"></div></div>';
  h += '<div class="row"><button class="btn bad" id="l-reset">Reset tool and jig</button></div>';
  pane.innerHTML = h;
}
const LEARN = [
  ['What is offline programming?', 'You teach and test the robot on a computer, not on the factory floor. The robot keeps producing while you work, and mistakes show up as a red row here instead of a crash in the cell.'],
  ['Joint move or linear move?', 'A joint move (MoveJ) turns every joint at the same pace: fast and always possible, but the tool follows a curved path. A linear move (MoveL) keeps the tool on a straight line, so the controller must solve all joints at every instant. Use MoveL near the part and MoveJ in free space.'],
  ['Tool centre point (TCP)', 'The TCP is the working point of the tool, like the electrode tips of a weld gun. All positions you record and all moves are about the TCP, so a wrong TCP means every point is wrong.'],
  ['Reach and the envelope', 'Switch on "Reach envelope" in the station tree. The blue cloud is every place the wrist can reach. A target outside it, or in the empty zone close to the base, cannot be programmed. Move the jig or the robot base until the points are inside.'],
  ['Singularity', 'When joint A5 is close to 0°, joints A4 and A6 line up and can trade rotation without moving the tool. The solver then needs huge joint speeds. The program check flags this. Keep A5 away from 0° near the weld.'],
  ['Joint limits', 'Each axis has a stop. The bar under each slider fills towards the limit and turns red near it. A straight line can be impossible just because one joint would need to pass its limit.'],
  ['Why the tool must point down', 'A spot gun must meet the sheet at the right angle, and the electrodes must squeeze along the sheet normal. In the mission a weld counts only if the work axis is within 25° of straight down.']
];
function paneLearn() {
  let h = '<div class="h" style="margin-top:0">Mission: spot-weld 4 points</div><div class="card">';
  if (ST.jig.type !== 'table') h += '<p class="note">The mission needs the default rotary table. Choose it in Tool & Layout.</p>';
  else { h += '<div class="note">Teach the robot to weld all four marked spots. A weld counts when the tool centre is within 20 mm of the spot and the tool points down within 25°.</div>' + welded.map((w, i) => '<div class="spot"><span class="dot' + (w ? ' on' : '') + '"></span>Spot ' + (i + 1) + (w ? ' welded' : '') + '</div>').join('') + '<div class="row"><button class="btn" id="m-reset">Reset mission</button><button class="btn pri" id="m-auto">Show me (auto-teach)</button></div>';
    if (missionLog.length) h += '<div class="note" style="margin-top:6px">' + missionLog.slice(-4).map(x => '• ' + x).join('<br>') + '</div>'; }
  h += '</div><div class="h">Steps to try</div><ol class="note" style="padding-left:18px;margin:0"><li>Choose a robot and the <b>Spot welding gun</b> (Tool & Layout).</li><li>In <b>Jog → TCP</b>, press <b>Point tool down</b>.</li><li>Drag the tool over spot 1 and press <b>Try a weld here</b> to test your aim.</li><li>Record <b>MoveJ</b> above the spot, then <b>Weld</b> on it, then a retreat <b>MoveL</b>.</li><li>Repeat for the other spots, rotate the jig with E1 if needed, and press <b>Check</b>.</li><li>Run it, then export the program.</li></ol>';
  h += '<div class="h">Key ideas</div>' + LEARN.map(([t, d]) => '<details><summary>' + t + '</summary><p>' + d + '</p></details>').join('');
  pane.innerHTML = h;
}

/* ---- pane events (delegated) ---- */
pane.addEventListener('input', e => {
  const t = e.target;
  if (t.dataset.j != null) { if (runner.on) return; if (t.dataset.j === 'e1') ST.e1 = +t.value * D2R; else ST.q[+t.dataset.j] = +t.value * D2R; updateAll(true); }
  else if (t.dataset.jn != null) { if (runner.on || t.value === '') return; const v = +t.value * D2R; if (t.dataset.jn === 'e1') ST.e1 = v; else { const i = +t.dataset.jn; ST.q[i] = Math.max(cur.lim[i][0], Math.min(cur.lim[i][1], v)); } updateAll(true, t); }
  else if (t.dataset.p != null) { if (runner.on || t.value === '') return; applyPoseInputs(); }
  else if (t.dataset.sp != null) { ST.prog[+t.dataset.sp].sp = Math.max(5, Math.min(200, +t.value || 100)); saveLocal(); }
  else if (t.id === 's-j') { ST.speedJ = Math.max(5, +t.value || 90); } else if (t.id === 's-l') { ST.speedL = Math.max(5, +t.value || 500); }
  else if (t.id === 'simspd') window.__spd = +t.value;
});
pane.addEventListener('change', e => {
  const t = e.target;
  if (t.id === 'p-loop') { ST.loop = t.checked; saveLocal(); }
  else if (t.id === 't-type' && /^lib\d+$/.test(t.value)) { const x = TOOLLIB[+t.value.slice(3)]; ST.tool = Object.assign(ST.tool, { type: 'lib', libi: +t.value.slice(3), url: x.url, tcp: (x.tcp || [0, 0, 200]).slice(), rot: (x.rot || [0, 0, 0]).slice(), mscale: x.scale || 0.001, mrot: (x.mrot || [0, 0, 0]).slice() }); buildTool(); refreshTargets(); updateAll(true); renderPane(); saveLocal(); }
  else if (t.id === 't-type') { ST.tool.type = t.value; const d = { gun: [0, 0, 300], gripper: [0, 0, 160], torch: [0, 0, 350], none: [0, 0, 0], stl: ST.tool.tcp }; ST.tool.tcp = d[t.value].slice(); if (t.value === 'stl' && !ST.tool.stl) $('#fstl').dataset.for = 'tool', $('#fstl').click(); buildTool(); refreshTargets(); updateAll(true); renderPane(); saveLocal(); }
  else if (t.dataset.tcp != null) { ST.tool.tcp[+t.dataset.tcp] = +t.value || 0; buildTool(); refreshTargets(); updateAll(true); saveLocal(); }
  else if (t.dataset.tr != null) { ST.tool.rot[+t.dataset.tr] = +t.value || 0; buildTool(); refreshTargets(); updateAll(true); saveLocal(); }
  else if (t.dataset.mr != null) { ST.tool.mrot[+t.dataset.mr] = +t.value || 0; buildTool(); saveLocal(); }
  else if (t.id === 't-sc') { ST.tool.mscale = +t.value || 0.001; buildTool(); saveLocal(); }
  else if (t.id === 'j-type') { ST.jig.type = t.value; if (t.value === 'stl' && !ST.jig.stl) $('#fstl').dataset.for = 'jig', $('#fstl').click(); buildJig(); refreshTargets(); updateAll(true); renderPane(); saveLocal(); }
  else if (t.dataset.jp != null) { ST.jig[['x', 'y', 'z'][+t.dataset.jp]] = +t.value || 0; buildJig(); updateAll(true); saveLocal(); }
  else if (t.id === 'j-sc') { ST.jig.scale = +t.value || 0.001; buildJig(); saveLocal(); } else if (t.id === 'j-rx') { ST.jig.rx = +t.value || 0; buildJig(); saveLocal(); } else if (t.id === 'j-ry') { ST.jig.ry = +t.value || 0; buildJig(); saveLocal(); }
  else if (t.dataset.bp != null) { ST.base[t.dataset.bp] = +t.value || 0; applyBase(); refreshTargets(); updateAll(true); renderTree(); saveLocal(); }
});
pane.addEventListener('click', e => {
  const t = e.target.closest('button,tr'); if (!t) return;
  if (t.dataset.mode) { setMode(t.dataset.mode); return; }
  if (t.dataset.gm) { gizmo.setMode(t.dataset.gm); renderPane(); return; }
  if (t.dataset.add) { addInstr(t.dataset.add, t.dataset.add === 'Wait' ? { t: 1 } : {}); return; }
  if (t.dataset.ex) { download('program.' + ({ rapid: 'mod', krl: 'src', txt: 'txt' })[t.dataset.ex], exportProgram(t.dataset.ex === 'txt' ? 'text' : t.dataset.ex)); return; }
  if (t.dataset.a2 === 'save') { action('save'); return; }
  if (t.dataset.del != null) { ST.prog.splice(+t.dataset.del, 1); issues = []; cycleTime = 0; refreshTargets(); renderPane(); saveLocal(); return; }
  if (t.dataset.up != null) { const i = +t.dataset.up; if (i > 0) { [ST.prog[i - 1], ST.prog[i]] = [ST.prog[i], ST.prog[i - 1]]; issues = []; refreshTargets(); renderPane(); saveLocal(); } return; }
  if (t.dataset.dn != null) { const i = +t.dataset.dn; if (i < ST.prog.length - 1) { [ST.prog[i + 1], ST.prog[i]] = [ST.prog[i], ST.prog[i + 1]]; issues = []; refreshTargets(); renderPane(); saveLocal(); } return; }
  if (t.dataset.upd != null) { const p = ST.prog[+t.dataset.upd]; p.q = ST.q.slice(); p.e1 = ST.e1; issues = []; refreshTargets(); renderPane(); saveLocal(); toast('Step ' + (+t.dataset.upd + 1) + ' updated to the current pose', 'ok'); return; }
  if (t.dataset.step) { const k = +document.querySelector('#jstep')?.value || 5, sel = window.__selJ ?? 0; if (!runner.on) { ST.q[sel] = Math.max(cur.lim[sel][0], Math.min(cur.lim[sel][1], ST.q[sel] + (+t.dataset.step) * k * D2R)); updateAll(true); } return; }
  if (t.tagName === 'TR' && t.dataset.i != null) return;
  switch (t.id) {
    case 'b-rec': addInstr('MoveJ'); toast('Step recorded as MoveJ. Change the type in the Program tab if needed.', 'ok'); break;
    case 'b-weldnow': tryWeld(false); break;
    case 'b-open': toolOpen = 1; break; case 'b-close': toolOpen = 0; break;
    case 'b-down': pointDown(); break;
    case 'b-reach': goToSpot(0); break;
    case 'b-cam': startCam(); break;
    case 'p-play': if (!runner.on) startRun(); else { runner.paused = !runner.paused; renderPane(); } break;
    case 'p-stop': stopRun('Stopped'); break;
    case 'p-check': analyze(); break;
    case 't-file': $('#fstl').dataset.for = 'tool'; $('#fstl').click(); break;
    case 'j-file': $('#fstl').dataset.for = 'jig'; $('#fstl').click(); break;
    case 'l-reset': { const d = defaults(); ST.tool = d.tool; ST.jig = d.jig; buildTool(); buildJig(); refreshTargets(); updateAll(true); renderPane(); saveLocal(); break; }
    case 'm-reset': welded = [false, false, false, false]; missionLog = []; paintSpots(); renderPane(); break;
    case 'm-auto': autoTeach(); tab = 'prog'; renderTabs(); renderPane(); break;
  }
});
pane.addEventListener('dblclick', e => { const tr = e.target.closest('tr[data-i]'); if (!tr || runner.on) return; const p = ST.prog[+tr.dataset.i]; if (p.q) { ST.q = p.q.slice(); ST.e1 = p.e1 || 0; updateAll(true); } });
pane.addEventListener('focusin', e => { const t = e.target; if (t.dataset && t.dataset.jn != null && t.dataset.jn !== 'e1') window.__selJ = +t.dataset.jn; if (t.dataset && t.dataset.j != null && t.dataset.j !== 'e1') window.__selJ = +t.dataset.j; });
function pointDown() { const m = alignDownMatrix(currentTCP()); solveToTarget(m, { iter: 60 }); updateAll(true); }
function goToSpot(i) {
  if (ST.jig.type !== 'table') { toast('Choose the default jig first', 'warn'); return; } jigRoot.updateMatrixWorld(true);
  const p = new T.Vector3().setFromMatrixPosition(spotMeshes[i].matrixWorld), m = alignDownMatrix(currentTCP()); m.setPosition(p.x, p.y, p.z + 0.1);
  const e = solveToTarget(m, { iter: 80 }); updateAll(true); if (e.p > 0.004) toast('That point is ' + (e.p * 1000).toFixed(0) + ' mm out of reach for this robot. Try another robot or move the jig closer.', 'warn');
}
function applyPoseInputs() {
  const v = [...pane.querySelectorAll('[data-p]')].map(i => parseFloat(i.value) || 0);
  const m = new T.Matrix4().compose(new T.Vector3(v[0] / 1000, v[1] / 1000, v[2] / 1000), new T.Quaternion().setFromEuler(new T.Euler(v[3] * D2R, v[4] * D2R, v[5] * D2R, 'ZYX')), new T.Vector3(1, 1, 1));
  solveToTarget(m, {}); updateAll(false, document.activeElement); snapGizmo();
}
function setMode(m, silent) {
  if (m === mode && !silent) return; mode = m; gizmo.visible = gizmo.enabled = m === 'tcp'; if (m !== 'gesture') stopCam(); if (m === 'tcp') snapGizmo();
  if (tab === 'jog') renderPane(); updateAll(true);
}

/* ---- live sync ---- */
function updateAll(doGizmo, skipEl) {
  ST.q.forEach((v, i) => { }); const F = fkAll(ST.q, []);
  for (let i = 0; i < 7; i++) frames[i].matrix.copy(F[i]);
  jigRot.rotation.z = ST.e1; animateTool(); robotRoot.updateMatrixWorld(true); jigRoot.updateMatrixWorld(true);
  if (doGizmo && mode === 'tcp') snapGizmo();
  syncUI(skipEl);
}
function eulerZYX(m) { return new T.Euler().setFromRotationMatrix(m, 'ZYX'); }
function syncUI(skip) {
  const m = currentTCP(), p = new T.Vector3().setFromMatrixPosition(m), e = eulerZYX(m);
  pane.querySelectorAll('[data-j]').forEach(el => { if (el.dataset.j === 'e1') el.value = (ST.e1 * R2D).toFixed(1); else el.value = (ST.q[+el.dataset.j] * R2D).toFixed(1); });
  pane.querySelectorAll('[data-jn]').forEach(el => { if (el !== skip) el.value = (el.dataset.jn === 'e1' ? ST.e1 : ST.q[+el.dataset.jn]) * R2D === 0 ? '0' : ((el.dataset.jn === 'e1' ? ST.e1 : ST.q[+el.dataset.jn]) * R2D).toFixed(1); });
  for (let i = 0; i < 6; i++) { const b = document.getElementById('bar' + i); if (b) { const [lo, hi] = cur.lim[i], f = (ST.q[i] - lo) / (hi - lo), d = Math.abs(f - 0.5) * 2; b.style.width = (f * 100) + '%'; b.style.background = d > 0.93 ? '#f44336' : d > 0.8 ? '#ffc107' : '#4caf50'; } }
  pane.querySelectorAll('[data-p]').forEach(el => { if (el === skip) return; const i = +el.dataset.p, v = i < 3 ? [p.x, p.y, p.z][i] * 1000 : [e.x, e.y, e.z][i - 3] * R2D; el.value = v.toFixed(i < 3 ? 1 : 1); });
  const flags = [];
  const nearLim = ST.q.some((q, i) => { const [lo, hi] = cur.lim[i]; return Math.abs((q - lo) / (hi - lo) - 0.5) * 2 > 0.93; });
  flags.push(nearLim ? ['bad', 'Joint limit'] : ['ok', 'Limits OK']);
  flags.push(wristSingular() ? ['warn', 'Wrist singularity'] : ['ok', 'No singularity']);
  if (ikErr.p > 0.004 && mode === 'tcp') flags.push(['bad', 'Pose unreachable (' + (ikErr.p * 1000).toFixed(0) + ' mm)']);
  if (runner.issueNow) flags.push(['bad', runner.issueNow]);
  $('#flags').innerHTML = flags.map(f => '<span class="flag ' + f[0] + '">' + f[1] + '</span>').join('');
  const wa = workAxisWorld(), tilt = Math.acos(Math.max(-1, Math.min(1, wa.dot(new T.Vector3(0, 0, -1))))) * R2D;
  $('#status').innerHTML = 'TCP <b>X ' + (p.x * 1000).toFixed(0) + ' Y ' + (p.y * 1000).toFixed(0) + ' Z ' + (p.z * 1000).toFixed(0) + '</b> mm · <b>' + (e.x * R2D).toFixed(0) + '/' + (e.y * R2D).toFixed(0) + '/' + (e.z * R2D).toFixed(0) + '</b>° · Joints <b>' + ST.q.map(v => (v * R2D).toFixed(0)).join(' ') + '</b> · E1 <b>' + (ST.e1 * R2D).toFixed(0) + '</b> · Tool tilt <b>' + tilt.toFixed(0) + '°</b>' + (runner.on ? ' · <b>Running step ' + (runner.i + 1) + '/' + ST.prog.length + '</b>' : '') + ' · Welded <b>' + welded.filter(Boolean).length + '/4</b>';
}
function applyBase() { robotRoot.position.set(ST.base.x, ST.base.y, ST.base.z); robotRoot.rotation.set(0, 0, ST.base.yaw * D2R); robotRoot.updateMatrixWorld(true); }

/* ---- views ---- */
function view(n) {
  const c = new T.Vector3(ST.base.x + 1.0, ST.base.y, 0.9), d = 5.2;
  const P = { iso: [3.6, -3.6, 2.8], front: [0, -d + 0, 1.2], side: [d, 0, 1.2], top: [0.01, -0.01, d + 1] }[n] || [4.6, -4.6, 3.4];
  if (n === 'fit') { camera.position.set(c.x + 3.0, c.y - 3.0, 2.4); } else camera.position.set(c.x + P[0], c.y + P[1], P[2]);
  controls.target.copy(c); controls.update();
}
document.getElementById('vtools').onclick = e => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.v) view(b.dataset.v); else if (b.id === 'b-home') goHome(); };
function goHome() { if (runner.on) return; ST.q = HOME.slice(); ST.e1 = 0; updateAll(true); }

/* ---- menus / file ---- */
function action(a) {
  if (a === 'new') { if (confirm('Start a new station? Unsaved work is lost.')) { stopRun(); ST = defaults(); welded = [false, false, false, false]; missionLog = []; issues = []; cycleTime = 0; rebuildAll(); } }
  else if (a === 'open') $('#fopen').click();
  else if (a === 'save') download('robot-station.json', JSON.stringify(ST), 'application/json');
  else if (a === 'exrapid') download('program.mod', exportProgram('rapid')); else if (a === 'exkrl') download('program.src', exportProgram('krl')); else if (a === 'extxt') download('program.txt', exportProgram('text'));
  else if (a === 'rec') addInstr('MoveJ'); else if (a === 'play') { if (!runner.on) startRun(); else { runner.paused = !runner.paused; if (tab === 'prog') renderPane(); } }
  else if (a === 'stop') stopRun('Stopped'); else if (a === 'check') { tab = 'prog'; renderTabs(); renderPane(); analyze(); } else if (a === 'auto') { autoTeach(); tab = 'prog'; renderTabs(); renderPane(); }
  else if (a.startsWith('v-')) view(a.slice(2));
  else if (a === 'help') { tab = 'learn'; renderTabs(); renderPane(); toast('Start with the Learn tab: choose the spot gun, press "Point tool down", aim at spot 1, record, and press Run.', 'ok'); }
}
document.getElementById('menubar').addEventListener('click', e => { const b = e.target.closest('[data-a]'); if (b) { action(b.dataset.a); document.activeElement.blur(); } });
$('#fopen').onchange = e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader(); r.onload = () => { try { const d = JSON.parse(r.result); ST = Object.assign(defaults(), d); if (!ROBOTS[ST.robot]) ST.robot = 'ABB_6700'; rebuildAll(); toast('Station loaded', 'ok'); } catch (err) { toast('That file is not a valid station', 'bad'); } }; r.readAsText(f); e.target.value = ''; };
$('#fstl').onchange = e => { const f = e.target.files[0], who = e.target.dataset.for; if (!f) return; const r = new FileReader(); r.onload = () => { const b = bufToB64(r.result); try { stlLoader.parse(r.result); } catch (err) { toast('Could not read that STL file', 'bad'); return; } if (who === 'jig') { ST.jig.stl = b; ST.jig.type = 'stl'; buildJig(); } else { ST.tool.stl = b; ST.tool.type = 'stl'; buildTool(); } refreshTargets(); renderPane(); saveLocal(); toast('Model loaded. Adjust scale and rotation if it looks wrong (millimetre models need scale 0.001).', 'ok'); }; r.readAsArrayBuffer(f); e.target.value = ''; };
const sel = $('#robotsel'); sel.innerHTML = Object.entries(ROBOTS).map(([k, v]) => '<option value="' + k + '">' + v.name + '</option>').join('');
sel.onchange = () => { if (runner.on) { sel.value = ST.robot; return; } ST.robot = sel.value; ST.q = HOME.slice(); buildRobot(); envKey = ''; if (VIS.env) buildEnvelope(); refreshTargets(); renderTree(); renderPane(); updateAll(true); saveLocal(); toast(ROBOTS[ST.robot].name + ' selected. Recorded steps keep their joint values, so run Check again.', 'ok'); issues = []; };
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return; const typing = /input|textarea|select/i.test(document.activeElement.tagName); if (typing) return;
  if (e.code === 'Space') { e.preventDefault(); action('play'); } else if (e.key === 'Escape') action('stop'); else if (e.key.toLowerCase() === 'r') action('rec'); else if (e.key.toLowerCase() === 'h') goHome();
  else if (e.key === '1') view('iso'); else if (e.key === '2') view('front'); else if (e.key === '3') view('side'); else if (e.key === '4') view('top'); else if (e.key.toLowerCase() === 'f') view('fit');
  else if (e.key === 'Delete' || e.key === 'Backspace') { const i = ST.prog.length - 1; if (i >= 0 && tab === 'prog') { ST.prog.pop(); refreshTargets(); renderPane(); } }
});
function saveLocal() { try { const s = JSON.stringify(ST); if (s.length < 3e6) localStorage.setItem('olp.station', s); } catch (e) { } }
function loadLocal() { try { const s = localStorage.getItem('olp.station'); if (s) { const d = JSON.parse(s); if (ROBOTS[d.robot]) ST = Object.assign(defaults(), d); } } catch (e) { } }
function rebuildAll() { sel.value = ST.robot; applyBase(); buildRobot(); buildTool(); buildJig(); envKey = ''; applyVis(); refreshTargets(); renderTree(); renderTabs(); renderPane(); updateAll(true); }

/* ============================================================
   7. GESTURE CONTROL (MediaPipe Hands, runs in the browser)
   ============================================================ */
let mpHands = null, mpCam = null, camOn = false;
const arCv = document.getElementById('arc'), arCtx = arCv.getContext('2d'), arBox = [0.33, 0.67, 0.3, 0.7];
let g = { inBox: false, t0: 0, eng: false, hBase: null, rBase: null, roll: null, pinch: false, lastPinch: 0 };
function startCam() {
  if (typeof Hands === 'undefined' || typeof Camera === 'undefined') { toast('The hand-tracking library could not be loaded. Check your internet connection.', 'bad'); return; }
  $('#ar').style.display = 'block'; camOn = true; g = { inBox: false, t0: 0, eng: false, hBase: null, rBase: null, roll: null, pinch: false, lastPinch: 0 };
  if (!mpHands) { mpHands = new Hands({ locateFile: f => 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/' + f }); mpHands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 }); mpHands.onResults(onHand); }
  if (!mpCam) mpCam = new Camera($('#cam'), { onFrame: async () => { if (camOn) await mpHands.send({ image: $('#cam') }); }, width: 320, height: 240 });
  mpCam.start().catch(err => toast('Camera not available: ' + err, 'bad'));
}
function stopCam() { camOn = false; $('#ar').style.display = 'none'; if (mpCam) { try { mpCam.stop(); } catch (e) { } mpCam = null; } g.eng = false; }
function onHand(res) {
  arCtx.clearRect(0, 0, 280, 210); if (mode !== 'gesture' || runner.on) return;
  const hud = $('#arhud'), W = 280, H = 210; arCtx.lineWidth = 2; arCtx.strokeStyle = g.eng ? '#4caf50' : g.inBox ? '#ffeb3b' : '#ff9800'; arCtx.strokeRect(arBox[0] * W, arBox[2] * H, (arBox[1] - arBox[0]) * W, (arBox[3] - arBox[2]) * H);
  if (!res.multiHandLandmarks || !res.multiHandLandmarks.length) { if (g.eng || g.inBox) { g.eng = false; g.inBox = false; hud.textContent = 'PLACE HAND INSIDE THE BOX'; } return; }
  const h = res.multiHandLandmarks[0], pc = h[9], inside = pc.x >= arBox[0] && pc.x <= arBox[1] && pc.y >= arBox[2] && pc.y <= arBox[3], now = Date.now();
  const raw = new T.Vector3(ST.base.x + 0.8 + (1 - pc.z) * 0.9, -(pc.x - 0.5) * 1.6, 0.3 + (1 - pc.y) * 1.0);
  if (!g.eng) {
    if (inside) { if (!g.inBox) { g.inBox = true; g.t0 = now; } const s = (now - g.t0) / 1000; if (s >= 2) { g.eng = true; hud.textContent = 'TRACKING: MOVE YOUR PALM'; g.hBase = raw.clone(); g.rBase = new T.Vector3().setFromMatrixPosition(currentTCP()); gizmoTarget.position.copy(g.rBase); } else hud.textContent = 'HOLD STILL (' + (2 - s).toFixed(1) + ' s)'; }
    else { g.inBox = false; hud.textContent = 'PLACE YOUR PALM IN THE CENTRE BOX'; }
  } else if (pc.x < 0.05 || pc.x > 0.95 || pc.y < 0.05 || pc.y > 0.95) { g.eng = false; g.inBox = false; hud.textContent = 'HAND LOST: PLACE IT IN THE BOX AGAIN'; }
  if (g.eng) {
    const target = g.rBase.clone().add(raw.clone().sub(g.hBase)); gizmoTarget.position.lerp(target, 0.12); gizmoTarget.updateMatrixWorld(true);
    const m = currentTCP(); m.setPosition(gizmoTarget.position); solveToTarget(m, { orient: false, lock: [5], iter: 12 });
    const roll = Math.atan2(h[17].y - h[5].y, h[17].x - h[5].x); if (g.roll != null) { let d = roll - g.roll; if (d > PI) d -= 2 * PI; if (d < -PI) d += 2 * PI; if (Math.abs(d) > 0.005) ST.q[5] = Math.max(cur.lim[5][0], Math.min(cur.lim[5][1], ST.q[5] + d * 0.45)); } g.roll = roll;
    const dd = Math.hypot(h[4].x - h[8].x, h[4].y - h[8].y, h[4].z - h[8].z);
    if (dd < 0.045) { if (!g.pinch && now - g.lastPinch > 1200) { g.pinch = true; g.lastPinch = now; addInstr('MoveJ'); toast('Step recorded by pinch gesture', 'ok'); } } else g.pinch = false;
    updateAll(false);
  }
  arCtx.fillStyle = g.eng ? '#00ffcc' : '#ff9800'; arCtx.beginPath(); arCtx.arc(pc.x * W, pc.y * H, 6, 0, 6.3); arCtx.fill();
}

/* ============================================================
   8. MAIN LOOP
   ============================================================ */
function resize() { const w = viewEl.clientWidth, h = viewEl.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(viewEl); addEventListener('resize', resize);
let last = performance.now(), acc = 0;
function loop(now) {
  requestAnimationFrame(loop); const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (runner.on) { runner.issueNow = null; advance(dt); updateAll(false); const tl = $('#tlbar'); if (tl) tl.style.width = ((runner.i + Math.min(1, runner.t / Math.max(runner.dur, 0.01))) / Math.max(1, ST.prog.length) * 100) + '%'; const rows = document.querySelectorAll('#progRows tr'); rows.forEach((r, i) => r.classList.toggle('cur', i === runner.i)); }
  else { animateTool(); }
  controls.update(); renderer.render(scene, camera);
}

/* ---------- start ---------- */
loadLocal(); view('iso'); rebuildAll(); resize(); requestAnimationFrame(loop);
window.__olp = { advance, ST: () => ST, T, scene, camera, renderer, updateAll, solveToTarget, autoTeach, startRun, analyze, runner, tcpWorld, ROBOTS, buildRobot, rebuildAll, tryWeld, action, setMode, view, welded: () => welded, issues: () => issues, cycle: () => cycleTime };
})();
</script>
</body>
</html>
'''

ROOT = Path(__file__).parent
ROBOT_DIR = ROOT / "assets" / "robots"
TOOL_DIR = ROOT / "assets" / "robot_tools"
GITHUB_RAW = "https://raw.githubusercontent.com/RashidiA/Industrial-Robot-Cloud-Simulator/main/"

st.set_page_config(page_title="Industrial Robot Cloud Simulator", page_icon="🤖",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding:0.4rem 0.6rem 0 0.6rem;max-width:100%}"
            "header[data-testid=stHeader]{display:none}footer{display:none}</style>",
            unsafe_allow_html=True)

# Per-robot mesh placement (metres / degrees). Same hack as the earlier app: the link_4 mesh
# is shifted 1.0 m back along its local X for every robot except the Yaskawa. Tune if a model looks off.
MESH_FIXES = {
    "ABB_6700": {"moff": {3: [-1.0, 0, 0]}},
    "ABB_4400": {"moff": {3: [-1.0, 0, 0]}},
    "ABB_6600": {"moff": {3: [-1.0, 0, 0]}},
    "KUKA_KR150": {"moff": {3: [-1.0, 0, 0]}},
    "Yaskawa_3500": {},
}

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def link_index(stem):
    n = stem.lower()
    if "base" in n:
        return 0
    m = re.search(r"(?:link|l|j|axis|a)[_\- ]?(\d)", n) or re.search(r"(\d)", n)
    return int(m.group(1)) if m else None

@st.cache_data(show_spinner=False)
def scan_assets():
    robots, tools, warn = {}, [], []
    if ROBOT_DIR.exists():
        for d in sorted(p for p in ROBOT_DIR.iterdir() if p.is_dir()):
            slots = [None] * 7
            for f in sorted(list(d.glob("*.stl")) + list(d.glob("*.STL"))):
                i = link_index(f.stem)
                if i is None or i > 6:
                    warn.append(f"{d.name}/{f.name}: cannot tell which link this is")
                else:
                    slots[i] = f.relative_to(ROOT).as_posix()
            robots[d.name] = slots
    if TOOL_DIR.exists():
        for f in sorted(list(TOOL_DIR.rglob("*.stl")) + list(TOOL_DIR.rglob("*.STL"))):
            kind = f.parent.name.replace("_", " ").title().rstrip("s")
            tools.append({"name": f"{kind}: {f.stem.replace('_', ' ')}", "path": f.relative_to(ROOT).as_posix()})
    return robots, tools, warn

def src_for(rel, mode):
    if rel is None:
        return None
    if mode.startswith("Embed"):
        return "data:application/octet-stream;base64," + base64.b64encode((ROOT / rel).read_bytes()).decode()
    return GITHUB_RAW + rel          # browser downloads straight from GitHub (CORS allowed)

robots, tool_files, warnings = scan_assets()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("How the browser gets your STL files",
                    ["GitHub direct (fast, recommended)", "Embed in page (works offline, slower start)"],
                    help="GitHub direct needs the files to be pushed to the main branch. "
                         "Embed works even for files not on GitHub yet, but makes the page bigger.")
    st.caption("Heavy computing runs in your own browser (edge computing); the server only delivers files.")
    st.write(f"Robots found: **{len(robots)}**  |  Tool files: **{len(tool_files)}**")
    for r, slots in robots.items():
        st.write(f"- {r}: {sum(s is not None for s in slots)}/7 meshes")
    for w in warnings:
        st.warning(w)

payload = {"robots": {}, "tools": []}
keys = {norm(k): k for k in MESH_FIXES}
for folder, slots in robots.items():
    key = keys.get(norm(folder))
    if not key:
        continue
    fix = dict(MESH_FIXES[key])
    if isinstance(fix.get("moff"), dict):
        fix["moff"] = [fix["moff"].get(i, [0, 0, 0]) for i in range(7)]
    payload["robots"][key] = {"meshes": [src_for(s, mode) for s in slots], **fix}
for t in tool_files:
    payload["tools"].append({"name": t["name"], "url": src_for(t["path"], mode)})

components.html(SIM_HTML.replace("/*__PAYLOAD__*/null", json.dumps(payload)), height=900, scrolling=False)
