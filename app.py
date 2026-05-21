import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import os
import json
import base64
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from scipy.spatial.transform import Rotation as R

# --- 1. SYSTEM INITIALIZATION ---
st.set_page_config(page_title="Multi-Robot OLP Pro-Simulator", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

if 'j_angles' not in st.session_state:
    st.session_state.j_angles = [0.0] * 8 
if 'program' not in st.session_state:
    st.session_state.program = []

# --- 2. MULTI-ROBOT KINEMATICS REGISTRY ---
ROBOT_REGISTRY = {
    "ABB_6700": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.78],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.32, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 1.1],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [1.142, 0.0, 0.2],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.2, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.2, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.78, 0.5, 1.28, 0.4, 0.2, 0.2, 0.1]
    },
    "ABB_4400": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.20, 0.0, 0.6],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 0.88],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [2.15, 0.0, 0.15],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [-1.0, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.14, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.68, 0.4, 0.88, 0.3, 0.15, 0.15, 0.08]
    },
    "ABB_6600": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.22],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.30, 0.0, 0.5],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, -0.2, 1.145], "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [1.145, 0.2, 0.20], "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.65, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.18, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.715, 0.45, 1.145, 0.38, 0.2, 0.2, 0.1]
    },
    "KUKA_KR150": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.55],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.35, 0.0, 0.0],   "orient": [0.0, -1.5708, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [1.3, 0.0, -0.05],  "orient": [0.0, 1.5708, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [2.40, 0.0, 0.1],   "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [-1.0, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.21, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.75, 0.5, 1.25, 0.35, 0.23, 0.21, 0.09]
    },
    "Yaskawa_3500": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.50],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.16, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 0.9],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [1.0, 0.0, 0.2],    "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.0, 0.0, 0.0],    "orient": [0.0, -1.5708, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.0, 0.0, -0.17],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.70, 0.45, 1.15, 0.35, 0.18, 0.18, 0.10]
    }
}

# --- 3. HARDWARE PROFILE INITIALIZATION LAYER ---
robot_folder_path = os.path.join(BASE_DIR, "assets", "robots")
available_profiles = list(ROBOT_REGISTRY.keys())
if os.path.exists(robot_folder_path):
    scanned_folders = [f for f in os.listdir(robot_folder_path) if os.path.isdir(os.path.join(robot_folder_path, f))]
    if scanned_folders:
        available_profiles = sorted(list(set(scanned_folders + available_profiles)))

# Read selection parameters from query parameters
query_params = st.query_params
if "profile" in query_params:
    st.session_state.robot_profile_selection = query_params.get("profile")

selected_profile = st.session_state.get("robot_profile_selection", "ABB_6700")
if selected_profile not in available_profiles:
    selected_profile = available_profiles[0]

active_cfg = ROBOT_REGISTRY.get(selected_profile, ROBOT_REGISTRY["ABB_6700"])

# --- 4. KINEMATICS ENGINE ---
@st.cache_resource
def build_robot_chain(profile_name, hardware_config):
    links_data = hardware_config["links"]
    return Chain(name=profile_name, links=[
        OriginLink(),
        URDFLink(name=links_data[0]["name"], origin_translation=links_data[0]["trans"], origin_orientation=links_data[0]["orient"], rotation=links_data[0]["rot"]),
        URDFLink(name=links_data[1]["name"], origin_translation=links_data[1]["trans"], origin_orientation=links_data[1]["orient"], rotation=links_data[1]["rot"]),
        URDFLink(name=links_data[2]["name"], origin_translation=links_data[2]["trans"], origin_orientation=links_data[2]["orient"], rotation=links_data[2]["rot"]),
        URDFLink(name=links_data[3]["name"], origin_translation=links_data[3]["trans"], origin_orientation=links_data[3]["orient"], rotation=links_data[3]["rot"]),
        URDFLink(name=links_data[4]["name"], origin_translation=links_data[4]["trans"], origin_orientation=links_data[4]["orient"], rotation=links_data[4]["rot"]),
        URDFLink(name=links_data[5]["name"], origin_translation=links_data[5]["trans"], origin_orientation=links_data[5]["orient"], rotation=links_data[5]["rot"]),
    ], active_links_mask=[False, True, True, True, True, True, True])

robot_chain = build_robot_chain(selected_profile, active_cfg)

@st.cache_data(show_spinner=False)
def get_file_base64_cached(filepath, file_hash=""):
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            return ""
    return ""

# --- 5. EVENT PARAMETER INTERCEPT LAYER ---
if "event" in query_params:
    event_type = query_params.get("event")
    if event_type == "sync_sequence":
        try:
            raw_program = json.loads(query_params.get("program_data", "[]"))
            st.session_state.program = raw_program
            if len(raw_program) > 0:
                st.session_state.j_angles = raw_program[-1]["angles"]
        except Exception as e:
            pass
    elif event_type == "clear_sequence":
        st.session_state.program = []
        st.session_state.j_angles = [0.0] * 8
    
    # Direct cleanup after structural interceptions
    st.query_params.clear()
    st.query_params.profile = selected_profile

# --- 6. VIRTUAL EMBEDDED VIEWPORT (OPTION 2 FIXED) ---
def build_embedded_viewport(payload):
    json_stream = json.dumps(payload)
    
    html_source = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body { margin: 0; background-color: #111111; overflow: hidden; font-family: sans-serif; user-select: none; color: white; }
            #canvas-container { width: 100vw; height: 100vh; position: absolute; top:0; left:0; z-index:1; }
            
            .control-panel { position: absolute; background: rgba(20, 20, 20, 0.85); border: 1px solid #333; border-radius: 8px; padding: 12px; z-index: 10; box-shadow: 0 4px 20px rgba(0,0,0,0.6); backdrop-filter: blur(5px); }
            #hardware-panel { top: 10px; left: 10px; width: 260px; }
            #jog-pendant { top: 10px; right: 10px; width: 240px; border: 1px solid #ff9800; }
            
            .panel-header { font-size: 11px; font-weight: bold; text-transform: uppercase; color: #ff9800; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
            .section-title { font-size: 11px; font-weight: bold; color: #00ffcc; margin-top: 10px; margin-bottom: 6px; text-transform: uppercase; border-bottom: 1px solid #222; padding-bottom: 2px;}
            
            .jog-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
            .jog-label { font-size: 12px; font-weight: bold; font-family: monospace; color: #bbb; }
            .jog-btn { background: #222; border: 1px solid #444; color: white; width: 45px; height: 26px; font-size: 14px; font-weight: bold; cursor: pointer; border-radius: 4px; }
            .jog-btn:active { background: #ff9800; color: black; border-color: #ff9800; }
            .val-display { font-family: monospace; font-size: 11px; color: #00ffcc; width: 60px; text-align: center; }
            
            .ui-select { background: #222; color: white; border: 1px solid #444; padding: 6px; width: 100%; border-radius: 4px; font-size: 12px; margin-bottom: 8px; font-weight: bold; }
            .ui-slider-group { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; }
            .ui-slider-label { font-size: 11px; color: #aaa; display: flex; justify-content: space-between; }
            .ui-slider { width: 100%; accent-color: #ff9800; margin: 2px 0; }
            
            .file-upload-btn { position: relative; display: flex; align-items: center; justify-content: center; background: #2a2a2a; border: 1px dashed #555; padding: 6px; border-radius: 4px; font-size: 11px; cursor: pointer; text-align: center; margin-bottom: 6px; transition: all 0.2s; }
            .file-upload-btn:hover { background: #3a3a3a; border-color: #00ffcc; }
            .file-upload-btn input[type="file"] { position: absolute; left: 0; top: 0; opacity: 0; width: 100%; height: 100%; cursor: pointer; }

            .btn-action { width: 100%; border: none; font-weight: bold; height: 30px; border-radius: 4px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 4px; }
            .btn-danger { background: #d32f2f; color: white; }
            .btn-danger:hover { background: #f44336; }
            #btn-save-step { background: #ff9800; color: black; }
            #btn-run-sim { background: #4caf50; color: white; }
            #btn-clear-seq { background: #f44336; color: white; }
            
            .toggle-panel-btn { background: none; border: none; color: #ff9800; cursor: pointer; font-size: 14px; padding: 0 5px; }
            .collapsed { height: 18px !important; overflow: hidden; padding-bottom: 0 !important; width: 180px !important; }
        </style>
    </head>
    <body>
        <div id="hardware-panel" class="control-panel">
            <div class="panel-header">
                <span>⚙️ HARDWARE SETUP</span>
                <button class="toggle-panel-btn" onclick="togglePanel('hardware-panel')">⌃</button>
            </div>
            <div class="panel-content">
                <label class="ui-slider-label"><span style="color:#ff9800; font-weight:bold;">Active Robot Profile</span></label>
                <select id="profile-selector" class="ui-select"></select>
                
                <div class="section-title">🔫 Tooling (Welding Gun)</div>
                <div class="file-upload-btn">
                    <span>📁 Load Gun STL File</span>
                    <input type="file" id="file-gun" accept=".stl">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>Gun Offset X (TCP)</span><span id="lbl-g-off-x">0.00m</span></div>
                    <input type="range" id="sld-g-off-x" min="-0.5" max="0.5" step="0.01" class="ui-slider">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>Gun Twist (Y-Rot)</span><span id="lbl-g-rot-z">180°</span></div>
                    <input type="range" id="sld-g-rot-z" min="-180" max="180" step="90" class="ui-slider">
                </div>

                <div class="section-title">🏗️ Rotary Positioner Jig</div>
                <div class="file-upload-btn">
                    <span>📁 Load Jig STL File</span>
                    <input type="file" id="file-jig" accept=".stl">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>Base Position X</span><span id="lbl-jx">1.60m</span></div>
                    <input type="range" id="sld-jx" min="0.5" max="3.0" step="0.05" class="ui-slider">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>Base Position Y</span><span id="lbl-jy">0.00m</span></div>
                    <input type="range" id="sld-jy" min="-1.5" max="1.5" step="0.05" class="ui-slider">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>Elevation Level Z</span><span id="lbl-jz">0.55m</span></div>
                    <input type="range" id="sld-jz" min="0.0" max="1.5" step="0.01" class="ui-slider">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>CAD Alignment Rot X</span><span id="lbl-jrot-x">0°</span></div>
                    <input type="range" id="sld-jrot-x" min="-180" max="180" step="90" class="ui-slider">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>CAD Alignment Rot Y</span><span id="lbl-jrot-y">0°</span></div>
                    <input type="range" id="sld-jrot-y" min="-180" max="180" step="90" class="ui-slider">
                </div>
                <div class="ui-slider-group">
                    <div class="ui-slider-label"><span>Geometry Scale</span><span id="lbl-jscale">0.001</span></div>
                    <input type="range" id="sld-jscale" min="0.0001" max="0.01" step="0.0001" class="ui-slider">
                </div>

                <button class="btn-action btn-danger" id="btn-hardware-reset" style="margin-top:10px;">🔴 RESET SYSTEM</button>
            </div>
        </div>

        <div id="jog-pendant" class="control-panel">
            <div class="panel-header">
                <span>⚡ INSTANT JOG PENDANT</span>
                <button class="toggle-panel-btn" onclick="togglePanel('jog-pendant')">⌃</button>
            </div>
            <div class="panel-content">
                <div id="jog-rows-container"></div>
                <div class="section-title">Simulation Execution</div>
                <div class="jog-row" style="margin-bottom: 8px;">
                    <div class="jog-label" style="font-size: 11px; color: #aaa;">SPEED</div>
                    <input type="range" id="sld-speed" min="5" max="100" value="50" step="5" style="flex-grow: 1; margin: 0 8px; accent-color: #ff9800;">
                    <div class="val-display" id="val-speed" style="width: 35px; color: #ff9800; font-weight: bold;">50%</div>
                </div>
                <button class="btn-action" id="btn-save-step">💾 RECORD STEP POSITION</button>
                <button class="btn-action" id="btn-run-sim">▶️ RUN SIMULATION</button>
                <button class="btn-action" id="btn-clear-seq">🗑️ CLEAR SEQUENCE</button>
                <div style="font-size: 12px; font-family: monospace; text-align: center; color: #aaa; margin-top: 6px;" id="lbl-steps">Steps: 0</div>
            </div>
        </div>

        <div id="canvas-container"></div>

        <script>
            const data = __PAYLOAD_OBJECT__;
            const dh = data.dhConfig;

            let localJointAngles = [...data.initialAngles];
            let lastComputedTransforms = [];
            let embeddedTrajectory = [...data.trajectory];
            
            const J_STEP = 5 * (Math.PI / 180);

            let gunOffset = data.gunOffset;
            let gunRotZ = data.gunRotZ;
            let jigX = data.jigX;
            let jigY = data.jigY;
            let jigZ = data.jigZ;
            let jigRotX = data.rotX;
            let jigRotY = data.rotY;
            let jigScale = data.jigScale;

            document.getElementById('sld-g-off-x').value = gunOffset;
            document.getElementById('lbl-g-off-x').innerText = gunOffset.toFixed(2) + "m";
            document.getElementById('sld-g-rot-z').value = Math.round(gunRotZ * (180.0 / Math.PI));
            document.getElementById('lbl-g-rot-z').innerText = Math.round(gunRotZ * (180.0 / Math.PI)) + "°";
            document.getElementById('sld-jx').value = jigX;
            document.getElementById('lbl-jx').innerText = jigX.toFixed(2) + "m";
            document.getElementById('sld-jy').value = jigY;
            document.getElementById('lbl-jy').innerText = jigY.toFixed(2) + "m";
            document.getElementById('sld-jz').value = jigZ;
            document.getElementById('lbl-jz').innerText = jigZ.toFixed(2) + "m";
            document.getElementById('sld-jrot-x').value = Math.round(jigRotX * (180.0 / Math.PI));
            document.getElementById('lbl-jrot-x').innerText = Math.round(jigRotX * (180.0 / Math.PI)) + "°";
            document.getElementById('sld-jrot-y').value = Math.round(jigRotY * (180.0 / Math.PI));
            document.getElementById('lbl-jrot-y').innerText = Math.round(jigRotY * (180.0 / Math.PI)) + "°";
            document.getElementById('sld-jscale').value = jigScale;
            document.getElementById('lbl-jscale').innerText = jigScale.toFixed(4);

            function togglePanel(id) {
                const el = document.getElementById(id);
                el.classList.toggle('collapsed');
                el.querySelector('.toggle-panel-btn').innerText = el.classList.contains('collapsed') ? '⌄' : '⌃';
            }

            // --- ESCAPE IFRAME AND REWRITE PARENT LOCATION ---
            const sel = document.getElementById('profile-selector');
            data.availableProfiles.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p;
                opt.innerText = p;
                if(p === data.profileName) opt.selected = true;
                sel.appendChild(opt);
            });
            
            sel.addEventListener('change', (e) => {
                // Modified handler explicitly targeting top parent frame URL context
                const parentUrl = new URL(window.parent.location.href);
                parentUrl.searchParams.set("profile", e.target.value);
                window.parent.location.href = parentUrl.toString();
            });

            THREE.Object3D.DefaultUp.set(0, 0, 1);
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
            camera.position.set(4.0, -4.0, 3.0);

            const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(0.8, 0, 0.8);

            scene.add(new THREE.AmbientLight(0x777777));
            const light1 = new THREE.DirectionalLight(0xffffff, 0.9); light1.position.set(5, 5, 10); scene.add(light1);
            const light2 = new THREE.DirectionalLight(0xffffff, 0.4); light2.position.set(-5, -5, 5); scene.add(light2);

            const grid = new THREE.GridHelper(15, 30, 0x555555, 0x252525);
            grid.rotation.x = Math.PI / 2;
            scene.add(grid);

            const loader = new THREE.STLLoader();
            const links = [];
            
            let gunMesh = new THREE.Group();
            let jigMesh = new THREE.Group();
            let internalJigContent = new THREE.Group();
            
            scene.add(gunMesh);
            jigMesh.add(internalJigContent);
            scene.add(jigMesh);

            function base64ToArrayBuffer(base64Str) {
                const binaryString = window.atob(base64Str);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) { bytes[i] = binaryString.charCodeAt(i); }
                return bytes.buffer;
            }

            let targetColor = 0xcccccc; 
            if (data.profileName === "Yaskawa_3500") { targetColor = 0x0055ff; } 
            else if (data.profileName === "KUKA_KR150") { targetColor = 0xff6600; }

            for(let i=0; i<7; i++) {
                let mesh;
                let currentLinkColor = (i === 0) ? 0x222222 : targetColor;

                if(data.linkGeometries && data.linkGeometries[i] && data.linkGeometries[i].length > 0) {
                    const geometry = loader.parse(base64ToArrayBuffer(data.linkGeometries[i]));
                    mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: currentLinkColor, roughness: 0.4 }));
                } else {
                    const h = data.fallbackHeights[i];
                    const geometry = new THREE.CylinderGeometry(0.18, 0.22, h, 24);
                    geometry.rotateX(Math.PI / 2); 
                    if(i === 1 || i === 2 || i === 3) { geometry.translate(0, 0, h / 2); }
                    mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: currentLinkColor, roughness: 0.4 }));
                }
                scene.add(mesh);
                links.push(mesh);
            }

            function parseAndApplyGunMesh(buffer) {
                gunMesh.clear();
                const geometry = loader.parse(buffer);
                geometry.center(); 
                geometry.rotateY(Math.PI / 2);
                const gunInternalMesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 }));
                gunInternalMesh.scale.set(0.001, 0.001, 0.001); 
                gunInternalMesh.rotation.y = gunRotZ; 
                gunMesh.add(gunInternalMesh);
                refreshPositions();
            }

            function parseAndApplyJigMesh(buffer) {
                internalJigContent.clear();
                const geometry = loader.parse(buffer);
                geometry.center(); 
                const m = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.6 }));
                m.scale.set(jigScale, jigScale, jigScale);
                m.rotation.x = jigRotX;
                m.rotation.y = jigRotY;
                internalJigContent.add(m);
                refreshPositions();
            }

            if(data.gunData && data.gunData.length > 0) parseAndApplyGunMesh(base64ToArrayBuffer(data.gunData));
            if(data.jigData && data.jigData.length > 0) parseAndApplyJigMesh(base64ToArrayBuffer(data.jigData));

            document.getElementById('sld-g-off-x').addEventListener('input', (e) => {
                gunOffset = parseFloat(e.target.value);
                document.getElementById('lbl-g-off-x').innerText = gunOffset.toFixed(2) + "m";
                refreshPositions();
            });
            document.getElementById('sld-g-rot-z').addEventListener('input', (e) => {
                gunRotZ = parseFloat(e.target.value) * (Math.PI / 180.0);
                document.getElementById('lbl-g-rot-z').innerText = e.target.value + "°";
                if(gunMesh.children[0]) gunMesh.children[0].rotation.y = gunRotZ;
                refreshPositions();
            });
            document.getElementById('sld-jx').addEventListener('input', (e) => {
                jigX = parseFloat(e.target.value);
                document.getElementById('lbl-jx').innerText = jigX.toFixed(2) + "m";
                refreshPositions();
            });
            document.getElementById('sld-jy').addEventListener('input', (e) => {
                jigY = parseFloat(e.target.value);
                document.getElementById('lbl-jy').innerText = jigY.toFixed(2) + "m";
                refreshPositions();
            });
            document.getElementById('sld-jz').addEventListener('input', (e) => {
                jigZ = parseFloat(e.target.value);
                document.getElementById('lbl-jz').innerText = jigZ.toFixed(2) + "m";
                refreshPositions();
            });

            function computeForwardKinematics(angles) {
                const computedTransforms = [];
                let currentMatrix = new THREE.Matrix4();
                computedTransforms.push({ pos: [0,0,0], quat: [0,0,0,1] });

                for (let i = 0; i < 6; i++) {
                    let linkData = dh[i];
                    let mTrans = new THREE.Matrix4().makeTranslation(linkData.trans[0], linkData.trans[1], linkData.trans[2]);
                    let mOrient = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(linkData.orient[0], linkData.orient[1], linkData.orient[2], 'XYZ'));
                    let mBase = mTrans.multiply(mOrient);
                    
                    if(i===0) mBase.multiply(new THREE.Matrix4().makeRotationZ(angles[1]));
                    if(i===1) mBase.multiply(new THREE.Matrix4().makeRotationY(angles[2]));
                    if(i===2) mBase.multiply(new THREE.Matrix4().makeRotationY(angles[3]));
                    if(i===3) mBase.multiply(new THREE.Matrix4().makeRotationX(angles[4]));
                    if(i===4) mBase.multiply(new THREE.Matrix4().makeRotationY(angles[5]));
                    if(i===5) {
                        if (data.profileName === "Yaskawa_3500") mBase.multiply(new THREE.Matrix4().makeRotationZ(angles[6]));
                        else mBase.multiply(new THREE.Matrix4().makeRotationX(angles[6]));
                    }
                    currentMatrix.multiply(mBase);
                    computedTransforms.push({
                        pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                        quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                    });
                }
                return computedTransforms;
            }

            function updateSceneTransforms(transforms, offsetValue, jX, jY, jZ, e1RotAngle) {
                for(let i=0; i<7; i++) {
                    if(links[i] && transforms[i]) {
                        links[i].position.fromArray(transforms[i].pos);
                        links[i].quaternion.fromArray(transforms[i].quat);
                    }
                }
                if(links[6]) {
                    links[6].updateMatrixWorld();
                    gunMesh.position.copy(links[6].position);
                    gunMesh.quaternion.copy(links[6].quaternion);
                    if (data.profileName === "Yaskawa_3500") gunMesh.translateZ(-offsetValue);
                    else gunMesh.translateX(offsetValue);
                }
                jigMesh.position.set(jX, jY, jZ);
                internalJigContent.rotation.z = e1RotAngle;
            }

            function refreshPositions() {
                lastComputedTransforms = computeForwardKinematics(localJointAngles);
                updateSceneTransforms(lastComputedTransforms, gunOffset, jigX, jigY, jigZ, localJointAngles[7]);
            }

            const rowsContainer = document.getElementById('jog-rows-container');
            for(let i=1; i<=6; i++) {
                const row = document.createElement('div');
                row.className = 'jog-row';
                row.innerHTML = `<button class="jog-btn" id="btn-m-${i}">-</button><div class="jog-label">A${i}</div><div class="val-display" id="val-${i}">${(localJointAngles[i]*180/Math.PI).toFixed(1)}°</div><button class="jog-btn" id="btn-p-${i}">+</button>`;
                rowsContainer.appendChild(row);
                document.getElementById(`btn-m-${i}`).addEventListener('click', () => jogJoint(i, -1));
                document.getElementById(`btn-p-${i}`).addEventListener('click', () => jogJoint(i, 1));
            }
            
            function jogJoint(jointIdx, direction) {
                localJointAngles[jointIdx] += direction * J_STEP;
                document.getElementById(`val-${jointIdx}`).innerText = `${(localJointAngles[jointIdx]*180/Math.PI).toFixed(1)}°`;
                refreshPositions();
            }

            document.getElementById('btn-save-step').addEventListener('click', () => {
                lastComputedTransforms = computeForwardKinematics(localJointAngles);
                embeddedTrajectory.push({ angles: [...localJointAngles], transforms: JSON.parse(JSON.stringify(lastComputedTransforms)) });
                document.getElementById('lbl-steps').innerText = "Steps: " + embeddedTrajectory.length;
            });

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }

            refreshPositions();
            animate();
        </script>
    </body>
    </html>
    """.replace("__PAYLOAD_OBJECT__", json_stream)
    
    components.html(html_source, height=780, scrolling=False)

# --- 7. DYNAMIC HARDWARE FILE BINDING LAYER ---
path_gun = os.path.join(TEMP_DIR, "gun.stl")
path_jig = os.path.join(TEMP_DIR, "jig.stl")

link_b64s = []
for i in range(7):
    mesh_filename = f"link_{i}.stl" if i > 0 else "base_link.stl"
    target_mesh_path = os.path.join(BASE_DIR, "assets", "robots", selected_profile, mesh_filename)
    if not os.path.exists(target_mesh_path):
        target_mesh_path = os.path.join(BASE_DIR, "assets", "meshes", mesh_filename)
    link_b64s.append(get_file_base64_cached(target_mesh_path))

scene_payload = {
    "profileName": selected_profile,
    "availableProfiles": available_profiles,
    "trajectory": st.session_state.program,
    "initialAngles": st.session_state.j_angles,
    "linkGeometries": link_b64s,
    "fallbackHeights": active_cfg["fallback_heights"],
    "dhConfig": active_cfg["links"],
    "gunData": get_file_base64_cached(path_gun),
    "jigData": get_file_base64_cached(path_jig),
    "gunOffset": 0.0, 
    "gunRotZ": np.pi,
    "jigX": 1.6, "jigY": 0.0, "jigZ": 0.55,
    "rotX": 0.0, "rotY": 0.0, "jigScale": 0.001
}

build_embedded_viewport(scene_payload)
