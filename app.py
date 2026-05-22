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

# --- 3. MINIMIZABLE STREAMLIT SIDEBAR CONTROLS ---
available_profiles = list(ROBOT_REGISTRY.keys())
if os.path.exists(os.path.join(BASE_DIR, "assets", "robots")):
    scanned = [f for f in os.listdir(os.path.join(BASE_DIR, "assets", "robots")) if os.path.isdir(os.path.join(BASE_DIR, "assets", "robots", f))]
    if scanned:
        available_profiles = sorted(list(set(scanned + available_profiles)))

with st.sidebar:
    st.markdown("### ⚙️ SIMULATOR ENGINE CONTROLS")
    st.info("💡 Minimize this sidebar using the top-left arrow for full 100% viewport space.")
    
    selected_profile = st.selectbox(
        "Active Robot Profile",
        options=available_profiles,
        index=available_profiles.index("ABB_6700") if "ABB_6700" in available_profiles else 0,
        key="robot_profile_selection"
    )

active_cfg = ROBOT_REGISTRY.get(selected_profile, ROBOT_REGISTRY["ABB_6700"])

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

# --- 4. EMBEDDED JOG AND ADVANCED TCP INTERFACE ---
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
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TransformControls.js"></script>
        <style>
            body { margin: 0; background-color: #111111; overflow: hidden; font-family: sans-serif; user-select: none; color: white; }
            #canvas-container { width: 100vw; height: 100vh; position: absolute; top:0; left:0; z-index:1; }
            
            .control-panel { position: absolute; background: rgba(20, 20, 20, 0.85); border: 1px solid #333; border-radius: 8px; padding: 12px; z-index: 10; box-shadow: 0 4px 20px rgba(0,0,0,0.6); backdrop-filter: blur(5px); }
            #hardware-panel { top: 10px; left: 10px; width: 260px; }
            #jog-pendant { top: 10px; right: 10px; width: 250px; border: 1px solid #ff9800; }
            
            .panel-header { font-size: 11px; font-weight: bold; text-transform: uppercase; color: #ff9800; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
            .section-title { font-size: 11px; font-weight: bold; color: #00ffcc; margin-top: 10px; margin-bottom: 6px; text-transform: uppercase; border-bottom: 1px solid #222; padding-bottom: 2px;}
            
            .mode-container { display: flex; gap: 4px; margin-bottom: 10px; }
            .mode-btn { flex: 1; background: #222; border: 1px solid #444; color: #aaa; padding: 6px; font-size: 11px; font-weight: bold; cursor: pointer; border-radius: 4px; }
            .mode-btn.active { background: #ff9800; color: black; border-color: #ff9800; }
            
            .jog-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
            .jog-label { font-size: 12px; font-weight: bold; font-family: monospace; color: #bbb; }
            .jog-btn { background: #222; border: 1px solid #444; color: white; width: 45px; height: 26px; font-size: 14px; font-weight: bold; cursor: pointer; border-radius: 4px; }
            .jog-btn:active { background: #ff9800; color: black; border-color: #ff9800; }
            .val-display { font-family: monospace; font-size: 11px; color: #00ffcc; width: 65px; text-align: center; }
            
            .ui-slider-group { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; }
            .ui-slider-label { font-size: 11px; color: #aaa; display: flex; justify-content: space-between; }
            .ui-slider { width: 100%; accent-color: #ff9800; margin: 2px 0; }
            
            .file-upload-btn { position: relative; display: flex; align-items: center; justify-content: center; background: #2a2a2a; border: 1px dashed #555; padding: 6px; border-radius: 4px; font-size: 11px; cursor: pointer; text-align: center; margin-bottom: 6px; }
            .file-upload-btn input[type="file"] { position: absolute; left: 0; top: 0; opacity: 0; width: 100%; height: 100%; cursor: pointer; }

            .btn-action { width: 100%; border: none; font-weight: bold; height: 30px; border-radius: 4px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 4px; }
            .btn-danger { background: #d32f2f; color: white; }
            #btn-save-step { background: #ff9800; color: black; }
            
            .toggle-panel-btn { background: none; border: none; color: #ff9800; cursor: pointer; font-size: 14px; }
            .collapsed { height: 18px !important; overflow: hidden; padding-bottom: 0 !important; }
        </style>
    </head>
    <body>
        <div id="hardware-panel" class="control-panel">
            <div class="panel-header">
                <span>🛠️ SETTINGS & GUN TOOLING</span>
                <button class="toggle-panel-btn" onclick="togglePanel('hardware-panel')">⌃</button>
            </div>
            <div class="panel-content">
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
                    <div class="ui-slider-label"><span>Elevation Level Z</span><span id="lbl-jz">0.55m</span></div>
                    <input type="range" id="sld-jz" min="0.0" max="1.5" step="0.01" class="ui-slider">
                </div>
            </div>
        </div>

        <div id="jog-pendant" class="control-panel">
            <div class="panel-header">
                <span>⚡ JOG SELECTION PENDANT</span>
                <button class="toggle-panel-btn" onclick="togglePanel('jog-pendant')">⌃</button>
            </div>
            <div class="panel-content">
                <div class="mode-container">
                    <button id="mode-joint" class="mode-btn active">JOINT JOG</button>
                    <button id="mode-tcp" class="mode-btn">🔴 TCP JOG (ROBODK)</button>
                </div>

                <div id="joint-jog-section">
                    <div id="jog-rows-container"></div>
                </div>

                <div id="tcp-display-section" style="display:none; background: rgba(0,0,0,0.3); padding: 8px; border-radius:4px; margin-bottom:10px;">
                    <div class="section-title" style="margin-top:0;">📡 Live TCP Coordinates</div>
                    <div style="font-family:monospace; font-size:11px; display:grid; grid-template-columns: 1fr 1fr; gap:4px;">
                        <div>X: <span id="tcp-x" style="color:#ff4444">0.000</span></div>
                        <div>Y: <span id="tcp-y" style="color:#44ff44">0.000</span></div>
                        <div>Z: <span id="tcp-z" style="color:#4444ff">0.000</span></div>
                    </div>
                    <p style="font-size:10px; color:#aaa; margin:6px 0 0 0;">Drag the colored 3D arrows on Axis 6 to move the robot dynamically.</p>
                </div>

                <button class="btn-action" id="btn-save-step">💾 RECORD STEP POSITION</button>
                <div style="font-size: 11px; font-family: monospace; text-align: center; color: #aaa; margin-top: 6px;" id="lbl-steps">Steps Recorded: 0</div>
            </div>
        </div>

        <div id="canvas-container"></div>

        <script>
            const data = __PAYLOAD_OBJECT__;
            const dh = data.dhConfig;

            let localJointAngles = [...data.initialAngles]; // [0, A1, A2, A3, A4, A5, A6, E1]
            let lastComputedTransforms = [];
            let embeddedTrajectory = [...data.trajectory];
            let activeJogMode = "joint"; // "joint" or "tcp"
            
            const J_STEP = 3 * (Math.PI / 180);
            let gunOffset = data.gunOffset;
            let gunRotZ = data.gunRotZ;
            let jigX = data.jigX;
            let jigY = data.jigY;
            let jigZ = data.jigZ;

            function togglePanel(id) {
                const el = document.getElementById(id);
                el.classList.toggle('collapsed');
            }

            // --- THREE.JS INITIALIZATION ---
            THREE.Object3D.DefaultUp.set(0, 0, 1);
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
            camera.position.set(3.5, -3.5, 2.5);

            const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(0.8, 0, 0.8);

            // --- LIGHTS & GRIDS ---
            scene.add(new THREE.AmbientLight(0x787878));
            const light1 = new THREE.DirectionalLight(0xffffff, 0.9); light1.position.set(5, 5, 10); scene.add(light1);
            const grid = new THREE.GridHelper(15, 30, 0x555555, 0x222222);
            grid.rotation.x = Math.PI / 2;
            scene.add(grid);

            // --- ROBOT MESH BUILDING ---
            const loader = new THREE.STLLoader();
            const links = [];
            let targetColor = 0xcccccc; 
            if (data.profileName === "Yaskawa_3500") targetColor = 0x0055ff; 
            else if (data.profileName === "KUKA_KR150") targetColor = 0xff6600;

            // Target pivot helper to attach the RoboDK interactive arrows
            const tcpTargetPivot = new THREE.Object3D();
            scene.add(tcpTargetPivot);

            for(let i=0; i<7; i++) {
                let mesh;
                let clr = (i === 0) ? 0x252525 : targetColor;
                if(data.linkGeometries && data.linkGeometries[i] && data.linkGeometries[i].length > 0) {
                    const geometry = loader.parse(base64ToArrayBuffer(data.linkGeometries[i]));
                    mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: clr, roughness: 0.4 }));
                } else {
                    const h = data.fallbackHeights[i];
                    const geometry = new THREE.CylinderGeometry(0.18, 0.22, h, 24);
                    geometry.rotateX(Math.PI / 2); 
                    if(i > 0 && i < 4) geometry.translate(0, 0, h / 2);
                    mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: clr, roughness: 0.4 }));
                }
                scene.add(mesh);
                links.push(mesh);
            }

            // --- ROBODK TRANSFORM CONTROL ARROWS SETUP ---
            const transformControls = new THREE.TransformControls(camera, renderer.domElement);
            transformControls.size = 0.6; // Keep arrows short and concise
            transformControls.setMode("translate");
            transformControls.attach(tcpTargetPivot);
            // Hide transform control arrows on start up
            transformControls.visible = false;
            transformControls.enabled = false;
            scene.add(transformControls);

            // Turn off OrbitControls when dragging arrows so viewport doesn't move around
            transformControls.addEventListener('dragging-changed', function (event) {
                controls.enabled = !event.value;
            });

            // Trigger analytical/numerical inverse adjustments when cursor drags the arrows
            transformControls.addEventListener('objectChange', function () {
                if (activeJogMode !== "tcp") return;
                
                // Get absolute mouse translation coords from target pivot position
                const targetPos = tcpTargetPivot.position;
                
                // Invoke simple geometric inverse calculation loop
                runInverseKinematicsForTCP(targetPos);
            });

            // --- KINEMATICS ENGINE PIPELINE ---
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

            // Simple cyclic coordinate loop calculation for dynamic tracking
            function runInverseKinematicsForTCP(targetGlobalPos) {
                // Iterative numeric correction loop to minimize delta position error
                for (let iter = 0; iter < 5; iter++) {
                    let currentTransforms = computeForwardKinematics(localJointAngles);
                    let endEffectorPos = new THREE.Vector3().fromArray(currentTransforms[6].pos);
                    
                    let errorVec = new THREE.Vector3().copy(targetGlobalPos).sub(endEffectorPos);
                    if (errorVec.length() < 0.001) break;

                    // Compute tracking adjustments across links A1-A3
                    for (let j = 1; j <= 3; j++) {
                        let jointPos = new THREE.Vector3().fromArray(currentTransforms[j-1].pos);
                        let axisDir = new THREE.Vector3(0, 0, 1); // Default rotation profile
                        if (j === 2 || j === 3) axisDir.set(0, 1, 0);

                        let curToEE = new THREE.Vector3().subVectors(endEffectorPos, jointPos).normalize();
                        let curToTarget = new THREE.Vector3().subVectors(targetGlobalPos, jointPos).normalize();

                        let dotProd = curToEE.dot(curToTarget);
                        let angleDiff = Math.acos(Math.max(-1, Math.min(1, dotProd))) * 0.1; // damping multiplier
                        
                        if (angleDiff > 0.001) {
                            let crossProd = new THREE.Vector3().crossVectors(curToEE, curToTarget);
                            if (crossProd.dot(axisDir) < 0) {
                                localJointAngles[j] -= angleDiff;
                            } else {
                                localJointAngles[j] += angleDiff;
                            }
                        }
                    }
                }
                refreshPositions(false); // Update meshes without updating the transform arrows loop
                updateLiveTCPTextDisplays(targetGlobalPos);
            }

            function refreshPositions(updateGizmoLocation = true) {
                lastComputedTransforms = computeForwardKinematics(localJointAngles);
                
                // Map coordinates out to layout link geometries
                for(let i=0; i<7; i++) {
                    if(links[i] && lastComputedTransforms[i]) {
                        links[i].position.fromArray(lastComputedTransforms[i].pos);
                        links[i].quaternion.fromArray(lastComputedTransforms[i].quat);
                    }
                }

                // Keep arrow coordinates synchronized with real-time frame locations
                if (updateGizmoLocation && lastComputedTransforms[6]) {
                    tcpTargetPivot.position.fromArray(lastComputedTransforms[6].pos);
                    tcpTargetPivot.quaternion.fromArray(lastComputedTransforms[6].quat);
                    updateLiveTCPTextDisplays(tcpTargetPivot.position);
                }

                // Synchronize angles into text readouts
                for (let i = 1; i <= 6; i++) {
                    let valDisplay = document.getElementById(`val-${i}`);
                    if(valDisplay) valDisplay.innerText = `${(localJointAngles[i]*180/Math.PI).toFixed(1)}°`;
                }
            }

            function updateLiveTCPTextDisplays(posVector) {
                document.getElementById("tcp-x").innerText = posVector.x.toFixed(3);
                document.getElementById("tcp-y").innerText = posVector.y.toFixed(3);
                document.getElementById("tcp-z").innerText = posVector.z.toFixed(3);
            }

            // --- PENDANT MODE TOGGLE BUTTON BINDINGS ---
            const btnJointMode = document.getElementById("mode-joint");
            const btnTCPMode = document.getElementById("mode-tcp");
            const jointSection = document.getElementById("joint-jog-section");
            const tcpSection = document.getElementById("tcp-display-section");

            btnJointMode.addEventListener("click", () => {
                activeJogMode = "joint";
                btnJointMode.classList.add("active");
                btnTCPMode.classList.remove("active");
                jointSection.style.display = "block";
                tcpSection.style.display = "none";
                
                transformControls.visible = false;
                transformControls.enabled = false;
            });

            btnTCPMode.addEventListener("click", () => {
                activeJogMode = "tcp";
                btnTCPMode.classList.add("active");
                btnJointMode.classList.remove("active");
                jointSection.style.display = "none";
                tcpSection.style.display = "block";
                
                // Reposition target base anchor point frame and display RoboDK interactive gizmo
                transformControls.visible = true;
                transformControls.enabled = true;
                refreshPositions(true);
            });

            // --- RENDER REFRESH LOOP ---
            const rowsContainer = document.getElementById('jog-rows-container');
            for(let i=1; i<=6; i++) {
                const row = document.createElement('div');
                row.className = 'jog-row';
                row.innerHTML = `<button class="jog-btn" id="btn-m-${i}">-</button><div class="jog-label">A${i}</div><div class="val-display" id="val-${i}">0.0°</div><button class="jog-btn" id="btn-p-${i}">+</button>`;
                rowsContainer.appendChild(row);
                document.getElementById(`btn-m-${i}`).addEventListener('click', () => jogJoint(i, -1));
                document.getElementById(`btn-p-${i}`).addEventListener('click', () => jogJoint(i, 1));
            }
            
            function jogJoint(jointIdx, direction) {
                localJointAngles[jointIdx] += direction * J_STEP;
                refreshPositions(true);
            }

            function base64ToArrayBuffer(base64Str) {
                const binaryString = window.atob(base64Str);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) { bytes[i] = binaryString.charCodeAt(i); }
                return bytes.buffer;
            }

            document.getElementById('btn-save-step').addEventListener('click', () => {
                embeddedTrajectory.push({ angles: [...localJointAngles] });
                document.getElementById('lbl-steps').innerText = "Steps Recorded: " + embeddedTrajectory.length;
            });

            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }

            refreshPositions(true);
            animate();
        </script>
    </body>
    </html>
    """.replace("__PAYLOAD_OBJECT__", json_stream)
    
    components.html(html_source, height=780, scrolling=False)

# --- 5. RUNTIME STACK ASSET LOADER ---
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
    "trajectory": st.session_state.program,
    "initialAngles": st.session_state.j_angles,
    "linkGeometries": link_b64s,
    "fallbackHeights": active_cfg["fallback_heights"],
    "dhConfig": active_cfg["links"],
    "gunData": get_file_base64_cached(path_gun),
    "jigData": get_file_base64_cached(path_jig),
    "gunOffset": 0.0, "gunRotZ": np.pi,
    "jigX": 1.6, "jigY": 0.0, "jigZ": 0.55
}

build_embedded_viewport(scene_payload)
