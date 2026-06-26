import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import os
import json
import base64
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

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
        "fallback_heights": [0.78, 0.5, 1.28, 0.4, 0.2, 0.2, 0.1],
        "limits": [
            [-np.pi, np.pi],               
            [-1.047, 1.483],               
            [-1.3962, 1.3962],             
            [-6.108, 6.108],               
            [-2.181, 2.181],               
            [-6.108, 6.108],               
            [-np.pi, np.pi]                
        ]
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
        "fallback_heights": [0.68, 0.4, 0.88, 0.3, 0.15, 0.15, 0.08],
        "limits": [
            [-np.pi, np.pi], 
            [-1.221, 1.658], 
            [-1.3962, 1.3962], 
            [-5.235, 5.235], 
            [-2.094, 2.094], 
            [-6.981, 6.981], 
            [-np.pi, np.pi]
        ]
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
        "fallback_heights": [0.715, 0.45, 1.145, 0.38, 0.2, 0.2, 0.1],
        "limits": [
            [-np.pi, np.pi], 
            [-1.134, 1.483], 
            [-1.3962, 1.3962], 
            [-5.235, 5.235], 
            [-2.094, 2.094], 
            [-5.235, 5.235], 
            [-np.pi, np.pi]
        ]
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
        "fallback_heights": [0.75, 0.5, 1.25, 0.35, 0.23, 0.21, 0.09],
        "limits": [
            [-3.228, 3.228], 
            [-0.785, 1.483], 
            [-1.3962, 1.3962], 
            [-6.108, 6.108], 
            [-2.181, 2.181], 
            [-6.108, 6.108], 
            [-np.pi, np.pi]
        ]
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
        "fallback_heights": [0.70, 0.45, 1.15, 0.35, 0.18, 0.18, 0.10],
        "limits": [
            [-3.141, 3.141], 
            [-1.570, 1.308], 
            [-1.3962, 1.3962], 
            [-3.141, 3.141], 
            [-2.268, 2.268], 
            [-6.283, 6.283], 
            [-np.pi, np.pi]
        ]
    }
}

# --- 3. SELECTION INITIALIZATION LAYER ---
with st.sidebar:
    st.title("📟 Simulation Setup")
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

# --- 4. KINEMATICS BUILDER ---
@st.cache_resource
def build_robot_chain(profile_name, hardware_config):
    links_data = hardware_config["links"]
    return Chain(name=profile_name, links=[
        OriginLink(),
        URDFLink(name=links_data[0]["name"], origin_translation=links_data[0]["trans"], origin_orientation=links_data[0]["orient"], rotation=links_data[0]["rot"]),
        URDFLink(name=links_data[1]["name"], origin_translation=links_data[1]["trans"], origin_orientation=links_data[1]["orient"], rotation=links_data[1]["rot"]),
        URDFLink(name=links_data[2]["name"], origin_translation=links_data[2]["trans"], origin_orientation=links_data[2]["orient"], rotation=links_data[2]["rot"]),
        URDFLink(name=links_data[3]["name"], origin_translation=links_data[3]["trans"], origin_orientation=links_data[3]["orient"], rotation=links_data[3]["rot"]),
        URDFLink(name=links_data[4]["name"], origin_translation=links_data[4]["trans"], origin_orientation=links_data[4]["orient"], rotation=hardware_config["links"][4]["rot"]),
        URDFLink(name=links_data[5]["name"], origin_translation=links_data[5]["trans"], origin_orientation=links_data[5]["orient"], rotation=hardware_config["links"][5]["rot"]),
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
    elif event_type == "reset_joints":
        st.session_state.j_angles = [0.0] * 7 + [st.session_state.j_angles[7]]
    st.query_params.clear()

# --- 6. OPERATOR INTERFACE CONTROLS ---
with st.sidebar:
    with st.expander("🛠️ Layout Setup", expanded=False):
        if st.button("🔴 RESET TOOL & JIG", use_container_width=True):
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

    with st.expander("⚙️ End-Effector Tooling Library", expanded=True):
        tool_source = st.radio("Tooling Model Source", options=["Internal Library", "External STL Upload"])
        selected_tool_path = None

        if tool_source == "Internal Library":
            CATEGORY_MAPPING = {
                "Welding Guns": "welding_guns",
                "Grippers": "grippers",
                "Welding Torches": "welding_torches"
            }
            selected_category = st.selectbox("Select Tool Category Type", options=list(CATEGORY_MAPPING.keys()))
            folder_target_name = CATEGORY_MAPPING[selected_category]
            library_scan_path = os.path.join(BASE_DIR, "assets", "robot_tools", folder_target_name)
            
            available_tools = []
            if os.path.exists(library_scan_path):
                available_tools = [f for f in os.listdir(library_scan_path) if f.lower().endswith('.stl')]
                
            if available_tools:
                selected_tool_file = st.selectbox("Select Tooling Model", options=available_tools)
                selected_tool_path = os.path.join(library_scan_path, selected_tool_file)
            else:
                st.caption("⚠️ No internal library templates found. Please upload an external file.")
        else:
            up_gun = st.file_uploader("Upload Custom External Tool STL", type=["stl"], key="gun_up")
            if up_gun:
                selected_tool_path = os.path.join(TEMP_DIR, "gun.stl")
                with open(selected_tool_path, "wb") as f: 
                    f.write(up_gun.getbuffer())
                st.cache_data.clear()

        st.divider()
        st.write("**📐 6-Axis Tool Center Point (TCP) Offsets**")
        col_ox, col_oy, col_oz = st.columns(3)
        with col_ox: t_off_x = st.number_input("Offset X", value=0.00, step=0.05, format="%.2f")
        with col_oy: t_off_y = st.number_input("Offset Y", value=0.00, step=0.05, format="%.2f")
        with col_oz: t_off_z = st.number_input("Offset Z", value=0.00, step=0.05, format="%.2f")
        
        st.write("**🔄 Tool Mounting Matrix Rotations (Degrees)**")
        t_rot_x = st.slider("Rotate X Axis (Roll)", -180, 180, 0, step=5)
        t_rot_y = st.slider("Rotate Y Axis (Pitch)", -180, 180, 0, step=5)
        t_rot_z = st.slider("Rotate Z Axis (Yaw)", -180, 180, 0, step=5)

if 'jx_pos' not in locals(): jx_pos = 1.6
if 'jy_pos' not in locals(): jy_pos = 0.0
if 'jz_pos' not in locals(): jz_pos = 0.55
if 'j_rot_x' not in locals(): j_rot_x = 0
if 'j_rot_y' not in locals(): j_rot_y = 0
if 'js_scale' not in locals(): js_scale = 0.001

# --- 7. VIRTUAL WEBGL SIMULATOR VIEWPORT ---
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
        
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

        <style>
            body { margin: 0; background-color: #111111; overflow: hidden; font-family: sans-serif; user-select: none; }
            #canvas-container { width: 100vw; height: 100vh; position: absolute; top:0; left:0; z-index:1; }
            #status { position: absolute; top: 10px; left: 10px; color: #ffffff; font-size: 13px; background: rgba(20,20,20,0.8); padding: 6px 12px; border-radius:4px; border: 1px solid #333; z-index: 10; }
            #jog-pendant { position: absolute; top: 10px; right: 10px; background: rgba(20, 20, 20, 0.85); border: 1px solid #ff9800; border-radius: 6px; width: 220px; padding: 10px; color: white; z-index: 10; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            .pendant-title { font-size: 11px; font-weight: bold; text-transform: uppercase; color: #ff9800; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 4px; margin-bottom: 8px; text-align: center; }
            
            .mode-container { display: flex; gap: 4px; margin-bottom: 8px; }
            .mode-btn { flex: 1; background: #222; border: 1px solid #444; color: #aaa; padding: 5px; font-size: 10px; font-weight: bold; cursor: pointer; border-radius: 4px; text-transform: uppercase; }
            .mode-btn.active { background: #ff9800; color: black; border-color: #ff9800; }
            
            .jog-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
            .jog-label { font-size: 12px; font-weight: bold; font-family: monospace; color: #bbb; }
            .jog-btn { background: #222; border: 1px solid #444; color: white; width: 45px; height: 26px; font-size: 14px; font-weight: bold; cursor: pointer; border-radius: 4px; transition: all 0.1s; }
            .jog-btn:active { background: #ff9800; color: black; border-color: #ff9800; }
            .val-display { font-family: monospace; font-size: 11px; color: #00ffcc; width: 60px; text-align: center; }
            .action-block { margin-top: 10px; border-top: 1px solid #333; padding-top: 10px; display: flex; flex-direction: column; gap: 6px; }
            .btn-action { width: 100%; border: none; font-weight: bold; height: 32px; border-radius: 4px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center; gap: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.3); transition: background 0.1s; }
            #btn-save-step { background: #ff9800; color: black; }
            #btn-run-sim { background: #4caf50; color: white; }
            #btn-clear-seq { background: #f44336; color: white; }
            #btn-reset-pos { background: #3f51b5; color: white; margin-bottom: 4px; border: 1px solid #5c6bc0; }
            #btn-reset-pos:active { background: #283593; }
            .step-counter { font-size: 12px; font-family: monospace; text-align: center; color: #aaa; margin-top: 2px; }
            
            #tcp-monitor { background: rgba(0,0,0,0.4); padding: 6px; border-radius: 4px; font-size: 11px; font-family: monospace; margin-bottom: 8px; display: none; }
            .tcp-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; text-align: center; margin-top: 4px; font-weight: bold; }
            
            .ar-viewport-container {
                position: absolute;
                bottom: 15px;
                left: 15px;
                width: 320px;
                height: 240px;
                border: 2px solid #ff9800;
                border-radius: 6px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.6);
                z-index: 10;
                display: none;
                overflow: hidden;
                transform: scaleX(-1);
            }
            #webcam-feedback {
                width: 100%;
                height: 100%;
                object-fit: cover;
                position: absolute;
                top: 0;
                left: 0;
                z-index: 11;
            }
            #ar-overlay-canvas {
                width: 100%;
                height: 100%;
                position: absolute;
                top: 0;
                left: 0;
                z-index: 12;
                pointer-events: none;
            }
            
            .ar-gesture-remarks-hud {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                background: rgba(0, 0, 0, 0.8);
                border-bottom: 1px solid #ff9800;
                color: #ffffff;
                font-size: 11px;
                text-align: center;
                padding: 5px 0;
                font-family: 'Segoe UI', sans-serif;
                font-weight: 500;
                letter-spacing: 0.5px;
                z-index: 20;
                transform: scaleX(-1);
            }
        </style>
    </head>
    <body>
        <div id="status">WebGL Processing...</div>
        
        <div class="ar-viewport-container" id="ar-viewport">
            <div class="ar-gesture-remarks-hud" id="ar-hud-text">
                PLACE HAND INSIDE BOX TO ENGAGE
            </div>
            <video id="webcam-feedback" autoplay playsinline></video>
            <canvas id="ar-overlay-canvas" width="320" height="240"></canvas>
        </div>

        <div id="jog-pendant">
            <div class="pendant-title">⚡ HYBRID ULTIMATE PENDANT</div>
            <div class="mode-container">
                <button id="mode-joint" class="mode-btn active">Joint</button>
                <button id="mode-tcp" class="mode-btn">⌖ TCP</button>
                <button id="mode-gesture" class="mode-btn">✋ Gesture</button>
            </div>
            <div id="tcp-monitor">
                <div style="color: #00ffcc; font-size: 10px; text-align:center;">TCP LIVE MONITOR (METERS)</div>
                <div class="tcp-grid">
                    <span style="color:#ff4444">X:<span id="lbl-tx">0.00</span></span>
                    <span style="color:#44ff44">Y:<span id="lbl-ty">0.00</span></span>
                    <span style="color:#4444ff">Z:<span id="lbl-tz">0.00</span></span>
                </div>
            </div>
            <div id="joint-jog-container"></div>
            <div class="action-block">
                <button class="btn-action" id="btn-reset-pos">🔄 RESET JOINT POSITIONS (0°)</button>
                
                <div class="jog-row" style="margin-bottom: 4px;">
                    <div class="jog-label" style="font-size: 11px; color: #aaa;">SPEED</div>
                    <input type="range" id="sld-speed" min="5" max="100" value="50" step="5" style="flex-grow: 1; margin: 0 10px; accent-color: #ff9800;">
                    <div class="val-display" id="val-speed" style="width: 35px; color: #ff9800; font-weight: bold;">50%</div>
                </div>
                <button class="btn-action" id="btn-save-step">💾 RECORD STEP POSITION</button>
                <button class="btn-action" id="btn-run-sim">▶️ RUN SIMULATION</button>
                <button class="btn-action" id="btn-clear-seq">🗑️ CLEAR SEQUENCE</button>
                <div class="step-counter" id="lbl-steps">Steps: 0</div>
            </div>
        </div>

        <div id="canvas-container"></div>

        <script>
            const data = JSON.parse(JSON.stringify(__PAYLOAD_STREAM__));
            const dh = data.dhConfig;
            const limitsConfig = data.jointLimits;

            let localJointAngles = [...data.initialAngles];
            let lastComputedTransforms = [];
            let embeddedTrajectory = [...data.trajectory];
            let activeJogMode = "joint";
            
            const J_STEP = 5 * (Math.PI / 180);

            THREE.Object3D.DefaultUp.set(0,
