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
            {"name": "A1", "trans": [0.0, 0.0, 0.55],   "orient": [0.0, -1.5708, 0.0], "rot": [0, 0, 1]},
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
            {"name": "A4", "trans": [0.0, 0.0, 0.21],   "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [1.0, 0.0, 0.0],    "orient": [0.0, -1.5708, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.0, 0.0, -0.17],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.70, 0.45, 1.15, 0.35, 0.18, 0.18, 0.10]
    }
}

# --- 3. SELECTION INITIALIZATION LAYER ---
with st.sidebar:
    st.title("📟 Teach Pendant Pro")
    with st.expander("🛠️ Layout Setup", expanded=True):
        robot_folder_path = os.path.join(BASE_DIR, "assets", "robots")
        available_profiles = list(ROBOT_REGISTRY.keys())
        
        if os.path.exists(robot_folder_path):
            scanned_folders = [f for f in os.listdir(robot_folder_path) if os.path.isdir(os.path.join(robot_folder_path, f))]
            if scanned_folders:
                available_profiles = sorted(list(set(scanned_folders + available_profiles)))

        selected_profile = st.selectbox(
            "Select Active Hardware Profile", 
            options=available_profiles,
            key="robot_profile_selection"
        )

active_cfg = ROBOT_REGISTRY.get(selected_profile, ROBOT_REGISTRY["ABB_6700"])

# --- 4. CONFIGURATION ADJUSTED KINEMATICS ENGINE ---
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

def get_file_hash(filepath):
    if os.path.exists(filepath):
        return str(os.path.getmtime(filepath))
    return ""

# --- 5. EVENT PARAMETER INTERCEPT LAYER ---
query_params = st.query_params
if "event" in query_params:
    event_type = query_params.get("event")
    if event_type == "sync_sequence":
        try:
            raw_program = json.loads(query_params.get("program_data", "[]"))
            st.session_state.program = raw_program
            if len(raw_program) > 0:
                st.session_state.j_angles = raw_program[-1]["angles"]
        except Exception as e:
            st.error(f"Pendant sync failure: {e}")
    elif event_type == "clear_sequence":
        st.session_state.program = []
        st.session_state.j_angles = [0.0] * 8
    st.query_params.clear()

# --- 6. OPERATOR INTERFACE CONTROLS ---
with st.sidebar:
    with st.expander("🛠️ Layout Setup", expanded=False):
        if st.button("🔴 RESET GUN & JIG", use_container_width=True):
            for f in [os.path.join(TEMP_DIR, "gun.stl"), os.path.join(TEMP_DIR, "jig.stl")]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            st.session_state.program = []
            st.session_state.j_angles = [0.0] * 8
            st.cache_data.clear()      
            st.cache_resource.clear()  
            st.query_params.clear()
            st.rerun()
            
        st.divider()
        st.write("**🔫 Welding Gun Tooling**")
        up_gun = st.file_uploader("Upload Gun STL", type=["stl"], key="gun_up")
        if up_gun:
            with open(os.path.join(TEMP_DIR, "gun.stl"), "wb") as f: 
                f.write(up_gun.getbuffer())
            st.cache_data.clear()
        
        g_off_x = st.slider("Gun Offset X (TCP)", -0.5, 0.5, 0.30, step=0.01)
        g_rot_z = st.slider("Gun Twist Orientation (Y-Axis Rot)", -180, 180, 180, step=90)
        
        st.divider()
        st.write("**🏗️ Rotary Positioning Jig**")
        up_jig = st.file_uploader("Upload Jig STL", type=["stl"], key="jig_up")
        if up_jig:
            with open(os.path.join(TEMP_DIR, "jig.stl"), "wb") as f: 
                f.write(up_jig.getbuffer())
            st.cache_data.clear()
            
        jx_pos = st.number_input("Jig Base X Location", value=1.6, step=0.1)
        jy_pos = st.number_input("Jig Base Y Location", value=0.0, step=0.1)
        jz_pos = st.number_input("Jig Base Z Elevation Level", value=0.55, step=0.01, format="%.3f")
        
        st.write("**📐 CAD Vector Calibration**")
        j_rot_x = st.slider("CAD Rotate X Axis", -180, 180, 0, step=90)
        j_rot_y = st.slider("CAD Rotate Y Axis", -180, 180, 0, step=90)
        js_scale = st.number_input("Jig Geometry Scale", value=0.001, format="%.5f")

if 'g_off_x' not in locals(): g_off_x = 0.30
if 'g_rot_z' not in locals(): g_rot_z = 180
if 'jx_pos' not in locals(): jx_pos = 1.6
if 'jy_pos' not in locals(): jy_pos = 0.0
if 'jz_pos' not in locals(): jz_pos = 0.55
if 'j_rot_x' not in locals(): j_rot_x = 0
if 'j_rot_y' not in locals(): j_rot_y = 0
if 'js_scale' not in locals(): js_scale = 0.001

# --- 7. VIRTUAL WEBGL SIMULATOR VIEWPORT WITH INTERACTIVE GIZMO ARROWS ---
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
            body { margin: 0; background-color: #111111; overflow: hidden; font-family: sans-serif; user-select: none; }
            #canvas-container { width: 100vw; height: 100vh; position: absolute; top:0; left:0; z-index:1; }
            #status { position: absolute; top: 10px; left: 10px; color: #ffffff; font-size: 13px; background: rgba(20,20,20,0.8); padding: 6px 12px; border-radius:4px; border: 1px solid #333; z-index: 10; }
            #jog-pendant { position: absolute; top: 10px; right: 10px; background: rgba(20, 20, 20, 0.85); border: 1px solid #ff9800; border-radius: 6px; width: 250px; padding: 10px; color: white; z-index: 10; box-shadow: 0 4px 15px rgba(0,0,0,0.5); display: flex; flex-direction: column; gap: 4px; }
            .pendant-title { font-size: 11px; font-weight: bold; text-transform: uppercase; color: #ff9800; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 4px; margin-bottom: 4px; text-align: center; }
            .mode-select { display: flex; gap: 4px; margin-bottom: 6px; }
            .mode-btn { flex: 1; background: #222; border: 1px solid #444; color: #aaa; font-size: 11px; padding: 4px; cursor: pointer; font-weight: bold; border-radius: 3px; }
            .mode-btn.active { background: #ff9800; color: #000; border-color: #ff9800; }
            .jog-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
            .jog-label { font-size: 12px; font-weight: bold; font-family: monospace; color: #bbb; width: 25px; }
            .jog-btn { background: #222; border: 1px solid #444; color: white; width: 40px; height: 24px; font-size: 13px; font-weight: bold; cursor: pointer; border-radius: 4px; transition: all 0.1s; }
            .jog-btn:active { background: #ff9800; color: black; border-color: #ff9800; }
            .val-display { font-family: monospace; font-size: 11px; color: #00ffcc; width: 65px; text-align: center; }
            .action-block { margin-top: 4px; border-top: 1px solid #333; padding-top: 6px; display: flex; flex-direction: column; gap: 4px; }
            .btn-action { width: 100%; border: none; font-weight: bold; height: 28px; border-radius: 4px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center; gap: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.3); }
            #btn-save-step { background: #ff9800; color: black; }
            #btn-run-sim { background: #4caf50; color: white; }
            #btn-clear-seq { background: #f44336; color: white; }
            .step-counter { font-size: 11px; font-family: monospace; text-align: center; color: #aaa; }
        </style>
    </head>
    <body>
        <div id="status">WebGL Processing...</div>
        <div id="jog-pendant">
            <div class="pendant-title">⚡ INSTANT JOG PENDANT</div>
            <div class="mode-select">
                <button class="mode-btn active" id="mode-joint">JOINT</button>
                <button class="mode-btn" id="mode-tcp">TCP (XYZ)</button>
            </div>
            
            <div id="jog-panel-container"></div>
            
            <div class="action-block">
                <div class="jog-row" style="margin-bottom: 2px;">
                    <div class="jog-label" style="font-size: 10px; color: #aaa; width: auto;">SPEED</div>
                    <input type="range" id="sld-speed" min="5" max="100" value="50" style="flex-grow: 1; margin: 0 6px; accent-color: #ff9800;">
                    <div class="val-display" id="val-speed" style="width: 30px; color: #ff9800; font-weight: bold;">50%</div>
                </div>
                <button class="btn-action" id="btn-save-step">💾 RECORD STEP POSITION</button>
                <button class="btn-action" id="btn-run-sim">▶️ RUN SIMULATION</button>
                <button class="btn-action" id="btn-clear-seq">🗑️ CLEAR SEQUENCE</button>
                <div class="step-counter" id="lbl-steps">Steps: 0</div>
            </div>
        </div>

        <div id="canvas-container"></div>

        <script>
            const data = __PAYLOAD_OBJECT__;
            const dh = data.dhConfig;

            let localJointAngles = [...data.initialAngles];
            let lastComputedTransforms = [];
            let embeddedTrajectory = [...data.trajectory];
            let activeMode = 'joint'; 
            
            const J_STEP = 3 * (Math.PI / 180); 
            const L_STEP = 0.02; 

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

            // --- ROBODK STYLE GIZMO ACCELERATION STACK ---
            const gizmoGroup = new THREE.Group();
            scene.add(gizmoGroup);

            // Base color variables
            const colorX = 0xff3333; // Red
            const colorY = 0x33cc33; // Green
            const colorZ = 0x3333ff; // Blue
            const colorHover = 0xffff00; // Yellow Highlight

            function createGizmoArrow(dir, color, axisName) {
                const arrowGroup = new THREE.Group();
                arrowGroup.userData = { axis: axisName, baseColor: color };

                // Shaft
                const shaftGeom = new THREE.CylinderGeometry(0.012, 0.012, 0.35, 8);
                shaftGeom.translate(0, 0.175, 0); 
                shaftGeom.rotateX(Math.PI / 2); 
                const shaftMat = new THREE.MeshBasicMaterial({ color: color, depthTest: false, depthWrite: false });
                const shaft = new THREE.Mesh(shaftGeom, shaftMat);
                shaft.renderOrder = 999;
                arrowGroup.add(shaft);

                // Cone tip
                const coneGeom = new THREE.ConeGeometry(0.035, 0.09, 12);
                coneGeom.translate(0, 0.35 + 0.045, 0);
                coneGeom.rotateX(Math.PI / 2);
                const coneMat = new THREE.MeshBasicMaterial({ color: color, depthTest: false, depthWrite: false });
                const cone = new THREE.Mesh(coneGeom, coneMat);
                cone.renderOrder = 999;
                arrowGroup.add(cone);

                // Reorient standard up-pointing vectors directly toward target directions
                if (dir.x > 0.9) arrowGroup.rotation.y = Math.PI / 2;
                else if (dir.x < -0.9) arrowGroup.rotation.y = -Math.PI / 2;
                else if (dir.y > 0.9) arrowGroup.rotation.x = -Math.PI / 2;
                else if (dir.y < -0.9) arrowGroup.rotation.x = Math.PI / 2;
                else if (dir.z < -0.9) arrowGroup.rotation.x = Math.PI;

                gizmoGroup.add(arrowGroup);
                return arrowGroup;
            }

            const arrowX = createGizmoArrow(new THREE.Vector3(1, 0, 0), colorX, 'X');
            const arrowY = createGizmoArrow(new THREE.Vector3(0, 1, 0), colorY, 'Y');
            const arrowZ = createGizmoArrow(new THREE.Vector3(0, 0, 1), colorZ, 'Z');

            // Raycasting cursor interaction
            const raycaster = new THREE.Raycaster();
            const mouseVec = new THREE.Vector2();
            let hoveredAxisMesh = null;

            window.addEventListener('mousemove', (e) => {
                const rect = renderer.domElement.getBoundingClientRect();
                mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

                raycaster.setFromCamera(mouseVec, camera);
                const intersects = raycaster.intersectObjects(gizmoGroup.children, true);

                if (intersects.length > 0 && activeMode === 'tcp' && !runSimulation) {
                    let parentGroup = intersects[0].object.parent;
                    if (hoveredAxisMesh !== parentGroup) {
                        resetGizmoColors();
                        hoveredAxisMesh = parentGroup;
                        parentGroup.children.forEach(child => child.material.color.setHex(colorHover));
                        document.body.style.cursor = 'pointer';
                    }
                } else {
                    if (hoveredAxisMesh) {
                        resetGizmoColors();
                        hoveredAxisMesh = null;
                        document.body.style.cursor = 'default';
                    }
                }
            });

            // Trigger click step on matching hover arrow axes
            window.addEventListener('click', () => {
                if (hoveredAxisMesh && activeMode === 'tcp' && !runSimulation) {
                    const axis = hoveredAxisMesh.userData.axis;
                    // Determine look target vector to decide positive or negative step orientation safely
                    const cameraDir = new THREE.Vector3();
                    camera.getWorldDirection(cameraDir);
                    const dot = (axis === 'X') ? cameraDir.x : (axis === 'Y' ? cameraDir.y : cameraDir.z);
                    const targetDirectionSign = (dot > 0) ? -1 : 1; 
                    
                    window.jogCartesian(axis, targetDirectionSign);
                }
            });

            function resetGizmoColors() {
                gizmoGroup.children.forEach(arrow => {
                    arrow.children.forEach(child => child.material.color.setHex(arrow.userData.baseColor));
                });
            }

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
                    const material = new THREE.MeshStandardMaterial({ color: currentLinkColor, roughness: 0.4 });
                    mesh = new THREE.Mesh(geometry, material);
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

            if(data.gunData && data.gunData.length > 0) {
                const geometry = loader.parse(base64ToArrayBuffer(data.gunData));
                geometry.center(); 
                geometry.rotateY(Math.PI / 2);
                const gunInternalMesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 }));
                gunInternalMesh.scale.set(0.001, 0.001, 0.001); 
                gunInternalMesh.rotation.y = data.gunRotZ; 
                gunMesh.add(gunInternalMesh);
            }

            if(data.jigData && data.jigData.length > 0) {
                const geometry = loader.parse(base64ToArrayBuffer(data.jigData));
                geometry.center(); 
                const m = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.6 }));
                m.scale.set(data.jigScale, data.jigScale, data.jigScale);
                m.rotation.x = data.rotX;
                m.rotation.y = data.rotY;
                internalJigContent.add(m);
            }

            function getLinkStructureBaseMatrix(linkData) {
                let mTrans = new THREE.Matrix4().makeTranslation(linkData.trans[0], linkData.trans[1], linkData.trans[2]);
                let mOrient = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(linkData.orient[0], linkData.orient[1], linkData.orient[2], 'XYZ'));
                return mTrans.multiply(mOrient);
            }

            function computeForwardKinematics(angles) {
                const computedTransforms = [];
                let currentMatrix = new THREE.Matrix4();
                
                computedTransforms.push({
                    pos: new THREE.Vector3(0,0,0).toArray(),
                    quat: new THREE.Quaternion().toArray()
                });

                let m1 = getLinkStructureBaseMatrix(dh[0]);
                m1.multiply(new THREE.Matrix4().makeRotationZ(angles[1]));
                currentMatrix.multiply(m1);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m2 = getLinkStructureBaseMatrix(dh[1]);
                m2.multiply(new THREE.Matrix4().makeRotationY(angles[2]));
                currentMatrix.multiply(m2);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m3 = getLinkStructureBaseMatrix(dh[2]);
                m3.multiply(new THREE.Matrix4().makeRotationY(angles[3]));
                currentMatrix.multiply(m3);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m4 = getLinkStructureBaseMatrix(dh[3]);
                m4.multiply(new THREE.Matrix4().makeRotationX(angles[4]));
                currentMatrix.multiply(m4);
                
                if (data.profileName === "Yaskawa_3500") {
                    computedTransforms.push({
                        pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                        quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                    });
                } else {
                    let correctionMatrix = currentMatrix.clone();
                    let dirVec = new THREE.Vector3(1, 0, 0);
                    let directionVector = dirVec.applyQuaternion(new THREE.Quaternion().setFromRotationMatrix(correctionMatrix));
                    let fixedPos = new THREE.Vector3().setFromMatrixPosition(correctionMatrix).add(directionVector.multiplyScalar(-1.0));
                    computedTransforms.push({
                        pos: fixedPos.toArray(),
                        quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                    });
                }

                let m5 = getLinkStructureBaseMatrix(dh[4]);
                m5.multiply(new THREE.Matrix4().makeRotationY(angles[5]));
                currentMatrix.multiply(m5);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m6 = getLinkStructureBaseMatrix(dh[5]);
                if (data.profileName === "Yaskawa_3500") {
                    m6.multiply(new THREE.Matrix4().makeRotationZ(angles[6]));
                } else {
                    m6.multiply(new THREE.Matrix4().makeRotationX(angles[6]));
                }
                currentMatrix.multiply(m6);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                return computedTransforms;
            }

            function getTcpWorldPosition(transforms) {
                let wristMatrix = new THREE.Matrix4().compose(
                    new THREE.Vector3().fromArray(transforms[6].pos),
                    new THREE.Quaternion().fromArray(transforms[6].quat),
                    new THREE.Vector3(1, 1, 1)
                );
                let tcpPos = new THREE.Vector3();
                if (data.profileName === "Yaskawa_3500") {
                    tcpPos.set(0, 0, -data.gunOffset).applyMatrix4(wristMatrix);
                } else {
                    tcpPos.set(data.gunOffset, 0, 0).applyMatrix4(wristMatrix);
                }
                return tcpPos;
            }

            function solveIKDelta(targetGlobalTcp) {
                let currentAngles = [...localJointAngles];
                let maxIterations = 15;
                let convergenceTolerance = 0.001;
                let dampFactor = 0.05;

                for (let iter = 0; iter < maxIterations; iter++) {
                    let currentTransforms = computeForwardKinematics(currentAngles);
                    let currentGlobalTcp = getTcpWorldPosition(currentTransforms);
                    
                    let errorVec = new THREE.Vector3().subVectors(targetGlobalTcp, currentGlobalTcp);
                    if (errorVec.length() < convergenceTolerance) break;

                    let jacobian = [];
                    for (let j = 1; j <= 6; j++) {
                        let deltaAngles = [...currentAngles];
                        let eps = 0.001;
                        deltaAngles[j] += eps;
                        
                        let fwdTransforms = computeForwardKinematics(deltaAngles);
                        let fwdGlobalTcp = getTcpWorldPosition(fwdTransforms);
                        
                        let partialDeriv = new THREE.Vector3().subVectors(fwdGlobalTcp, currentGlobalTcp).multiplyScalar(1.0 / eps);
                        jacobian.push([partialDeriv.x, partialDeriv.y, partialDeriv.z]);
                    }

                    for (let j = 0; j < 6; j++) {
                        let dotProd = jacobian[j][0]*errorVec.x + jacobian[j][1]*errorVec.y + jacobian[j][2]*errorVec.z;
                        currentAngles[j + 1] += dotProd * dampFactor;
                    }
                }
                return currentAngles;
            }

            function updateSceneTransforms(transforms, gunOffset, jigX, jigY, jigZ, e1RotAngle) {
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
                    
                    if (data.profileName === "Yaskawa_3500") {
                        gunMesh.translateZ(-gunOffset);
                    } else {
                        gunMesh.translateX(gunOffset);
                    }

                    // Dynamically position the RoboDK Gizmo group center right at the active tool center point
                    let currentTcp = getTcpWorldPosition(transforms);
                    gizmoGroup.position.copy(currentTcp);
                }
                jigMesh.position.set(jigX, jigY, jigZ);
                internalJigContent.rotation.z = e1RotAngle;
            }

            function validateAndCommitMovement(candidateAngles) {
                let projectedTransforms = computeForwardKinematics(candidateAngles);
                
                for (let i = 0; i < projectedTransforms.length; i++) {
                    if (projectedTransforms[i].pos[2] < -0.001) {
                        triggerSafetyViolationUI();
                        return false; 
                    }
                }

                let projectedTcp = getTcpWorldPosition(projectedTransforms);
                if (projectedTcp.z < -0.001) {
                    triggerSafetyViolationUI();
                    return false; 
                }

                localJointAngles = candidateAngles;
                lastComputedTransforms = projectedTransforms;
                updateSceneTransforms(lastComputedTransforms, data.gunOffset, data.jigX, data.jigY, data.jigZ, localJointAngles[7]);
                refreshPendantValues();
                return true;
            }

            function triggerSafetyViolationUI() {
                const statusBox = document.getElementById('status');
                statusBox.innerText = "⚠️ KINEMATIC BLOCKED: VIOLATION DETECTED ON Z PLANE (- SIDE)";
                statusBox.style.color = "#ff3333";
                statusBox.style.fontWeight = "bold";
                setTimeout(() => {
                    statusBox.innerText = "Status: Online (WebGL Ready)";
                    statusBox.style.color = "#ffffff";
                    statusBox.style.fontWeight = "normal";
                }, 1800);
            }

            function drawPendantInterface() {
                const panel = document.getElementById('jog-panel-container');
                panel.innerHTML = '';
                
                if (activeMode === 'joint') {
                    gizmoGroup.visible = false; // Hide tracking arrows in joint mode
                    for(let i=1; i<=6; i++) {
                        const row = document.createElement('div');
                        row.className = 'jog-row';
                        row.innerHTML = `
                            <button class="jog-btn" onclick="jogJoint(${i}, -1)">-</button>
                            <div class="jog-label">A${i}</div>
                            <div class="val-display" id="val-${i}">0.0°</div>
                            <button class="jog-btn" onclick="jogJoint(${i}, 1)">+</button>
                        `;
                        panel.appendChild(row);
                    }
                } else {
                    gizmoGroup.visible = true; // Make RoboDK arrows visible
                    const axLabels = ['X', 'Y', 'Z'];
                    axLabels.forEach((axis, idx) => {
                        const row = document.createElement('div');
                        row.className = 'jog-row';
                        row.innerHTML = `
                            <button class="jog-btn" onclick="jogCartesian('${axis}', -1)">-</button>
                            <div class="jog-label">${axis}</div>
                            <div class="val-display" id="val-tcp-${axis}">0.00m</div>
                            <button class="jog-btn" onclick="jogCartesian('${axis}', 1)">+</button>
                        `;
                        panel.appendChild(row);
                    });
                }

                const e1Row = document.createElement('div');
                e1Row.className = 'jog-row';
                e1Row.style.marginTop = '6px';
                e1Row.style.borderTop = '1px solid #333';
                e1Row.style.paddingTop = '6px';
                e1Row.innerHTML = `
                    <button class="jog-btn" onclick="jogJoint(7, -1)">-</button>
                    <div class="jog-label" style="color:#ff9800;">E1</div>
                    <div class="val-display" id="val-7">0.0°</div>
                    <button class="jog-btn" onclick="jogJoint(7, 1)">+</button>
                `;
                panel.appendChild(e1Row);
                refreshPendantValues();
            }

            function refreshPendantValues() {
                if (activeMode === 'joint') {
                    for(let i=1; i<=6; i++) {
                        const elem = document.getElementById(`val-${i}`);
                        if(elem) elem.innerText = (localJointAngles[i] * (180 / Math.PI)).toFixed(1) + "°";
                    }
                } else {
                    let currentTransforms = computeForwardKinematics(localJointAngles);
                    let tcp = getTcpWorldPosition(currentTransforms);
                    if(document.getElementById('val-tcp-X')) document.getElementById('val-tcp-X').innerText = tcp.x.toFixed(3) + "m";
                    if(document.getElementById('val-tcp-Y')) document.getElementById('val-tcp-Y').innerText = tcp.y.toFixed(3) + "m";
                    if(document.getElementById('val-tcp-Z')) document.getElementById('val-tcp-Z').innerText = tcp.z.toFixed(3) + "m";
                }
                const e1Elem = document.getElementById('val-7');
                if(e1Elem) e1Elem.innerText = (localJointAngles[7] * (180 / Math.PI)).toFixed(1) + "°";
            }

            window.jogJoint = function(jointIdx, direction) {
                if(runSimulation) return;
                let candidate = [...localJointAngles];
                candidate[jointIdx] += direction * J_STEP;
                validateAndCommitMovement(candidate);
            };

            window.jogCartesian = function(axis, direction) {
                if(runSimulation) return;
                
                let currentTransforms = computeForwardKinematics(localJointAngles);
                let targetTcp = getTcpWorldPosition(currentTransforms);
                
                if(axis === 'X') targetTcp.x += direction * L_STEP;
                if(axis === 'Y') targetTcp.y += direction * L_STEP;
                if(axis === 'Z') targetTcp.z += direction * L_STEP;
                
                let candidateAngles = solveIKDelta(targetTcp);
                candidateAngles[7] = localJointAngles[7]; 
                
                validateAndCommitMovement(candidateAngles);
            };

            document.getElementById('mode-joint').addEventListener('click', () => {
                activeMode = 'joint';
                document.getElementById('mode-joint').className = 'mode-btn active';
                document.getElementById('mode-tcp').className = 'mode-btn';
                drawPendantInterface();
            });
            document.getElementById('mode-tcp').addEventListener('click', () => {
                activeMode = 'tcp';
                document.getElementById('mode-tcp').className = 'mode-btn active';
                document.getElementById('mode-joint').className = 'mode-btn';
                drawPendantInterface();
            });

            document.getElementById('sld-speed').addEventListener('input', (e) => {
                document.getElementById('val-speed').innerText = e.target.value + "%";
            });

            document.getElementById('btn-save-step').addEventListener('click', () => {
                if(runSimulation) return;
                lastComputedTransforms = computeForwardKinematics(localJointAngles);
                embeddedTrajectory.push({
                    angles: [...localJointAngles],
                    transforms: JSON.parse(JSON.stringify(lastComputedTransforms))
                });
                document.getElementById('lbl-steps').innerText = "Steps: " + embeddedTrajectory.length;
                
                const targetUrl = new URL(window.parent.location.href);
                targetUrl.searchParams.set("event", "sync_sequence");
                targetUrl.searchParams.set("program_data", JSON.stringify(embeddedTrajectory));
                window.parent.history.replaceState({}, '', targetUrl.toString());
            });

            document.getElementById('btn-run-sim').addEventListener('click', () => {
                if(embeddedTrajectory.length < 2) {
                    alert("Record at least 2 target points to play coordinate tracking routes.");
                    return;
                }
                simStepIndex = 0;
                interpolationFraction = 0;
                runSimulation = true;
            });

            document.getElementById('btn-clear-seq').addEventListener('click', () => {
                embeddedTrajectory = [];
                document.getElementById('lbl-steps').innerText = "Steps: 0";
                const targetUrl = new URL(window.parent.location.href);
                targetUrl.searchParams.set("event", "clear_sequence");
                window.parent.location.href = targetUrl.toString();
            });

            let simStepIndex = 0;
            let interpolationFraction = 0;
            let runSimulation = false;

            function animate() {
                requestAnimationFrame(animate);
                controls.update();

                if (runSimulation && embeddedTrajectory.length >= 2) {
                    gizmoGroup.visible = false; // Hide during auto playback
                    document.getElementById('jog-pendant').style.opacity = "0.3"; 
                    document.getElementById('status').innerText = "Status: Running Sequence Simulation";
                    let currentPoint = embeddedTrajectory[simStepIndex];
                    let nextPoint = embeddedTrajectory[simStepIndex + 1];

                    let currentPercent = parseFloat(document.getElementById('sld-speed').value);
                    let computedStepIncrement = (currentPercent / 100) * 0.04;

                    interpolationFraction += computedStepIncrement;
                    if(interpolationFraction >= 1.0) {
                        interpolationFraction = 0;
                        simStepIndex++;
                        if (simStepIndex >= embeddedTrajectory.length - 1) {
                            runSimulation = false;
                            simStepIndex = 0;
                        }
                    }

                    let blendedTransforms = [];
                    for(let i=0; i<7; i++) {
                        let pV1 = new THREE.Vector3().fromArray(currentPoint.transforms[i].pos);
                        let pV2 = new THREE.Vector3().fromArray(nextPoint.transforms[i].pos);
                        let blendedPos = new THREE.Vector3().lerpVectors(pV1, pV2, interpolationFraction).toArray();

                        let qV1 = new THREE.Quaternion().fromArray(currentPoint.transforms[i].quat);
                        let qV2 = new THREE.Quaternion().fromArray(nextPoint.transforms[i].quat);
                        let blendedQuat = qV1.slerp(qV2, interpolationFraction).toArray();

                        blendedTransforms.push({ "pos": blendedPos, "quat": blendedQuat });
                    }
                    let intermediateE1 = (1 - interpolationFraction) * currentPoint.angles[7] + interpolationFraction * nextPoint.angles[7];
                    updateSceneTransforms(blendedTransforms, data.gunOffset, data.jigX, data.jigY, data.jigZ, intermediateE1);
                } else {
                    document.getElementById('jog-pendant').style.opacity = "1.0";
                    document.getElementById('status').innerText = "Status: Online (WebGL Ready)";
                    if (activeMode === 'tcp') gizmoGroup.visible = true;
                }
                renderer.render(scene, camera);
            }

            window.addEventListener('resize', () => {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });

            lastComputedTransforms = computeForwardKinematics(localJointAngles);
            updateSceneTransforms(lastComputedTransforms, data.gunOffset, data.jigX, data.jigY, data.jigZ, localJointAngles[7]);
            drawPendantInterface();
            document.getElementById('lbl-steps').innerText = "Steps: " + embeddedTrajectory.length;
            animate();
        </script>
    </body>
    </html>
    """.replace("__PAYLOAD_OBJECT__", json_stream)
    
    components.html(html_source, height=750, scrolling=False)

# --- 8. DYNAMIC HARDWARE FILE BINDING LAYER ---
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
    "gunData": get_file_base64_cached(path_gun, get_file_hash(path_gun)),
    "jigData": get_file_base64_cached(path_jig, get_file_hash(path_jig)),
    "gunOffset": g_off_x,
    "gunRotZ": float(g_rot_z) * (np.pi / 180.0),
    "jigX": jx_pos,
    "jigY": jy_pos,
    "jigZ": jz_pos,
    "rotX": float(j_rot_x) * (np.pi / 180.0),
    "rotY": float(j_rot_y) * (np.pi / 180.0),
    "jigScale": js_scale
}

build_embedded_viewport(scene_payload)
