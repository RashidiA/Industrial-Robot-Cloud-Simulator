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
# All original parameters are exactly preserved. Yaskawa reflects your modified architecture.
ROBOT_REGISTRY = {
    "ABB_6700": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.78],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.32, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 1.28],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [1.142, 0.0, 0.2],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.2, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.2, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.78, 0.5, 1.28, 0.4, 0.2, 0.2, 0.1]
    },
    "ABB_4400": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.68],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.20, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 0.88],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [0.85, 0.0, 0.15],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.15, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.14, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.68, 0.4, 0.88, 0.3, 0.15, 0.15, 0.08]
    },
    "ABB_6600": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.715],  "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.30, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 1.145],  "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [1.145, 0.0, 0.20], "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.20, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.20, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.715, 0.45, 1.145, 0.38, 0.2, 0.2, 0.1]
    },
    "KUKA_KR150": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.75],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.35, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 1.25],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [1.10, 0.0, 0.05],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.23, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.21, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.75, 0.5, 1.25, 0.35, 0.23, 0.21, 0.09]
    },
    "Yaskawa": {
        "links": [
            {"name": "A1", "trans": [0.0, 0.0, 0.70],   "orient": [0.0, 0.0, 0.0], "rot": [0, 0, 1]},
            {"name": "A2", "trans": [0.30, 0.0, 0.0],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A3", "trans": [0.0, 0.0, 1.15],   "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A4", "trans": [1.10, 0.0, 0.15],  "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
            {"name": "A5", "trans": [0.18, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [0, 1, 0]},
            {"name": "A6", "trans": [0.18, 0.0, 0.0],    "orient": [0.0, 0.0, 0.0], "rot": [1, 0, 0]},
        ],
        "fallback_heights": [0.70, 0.45, 1.15, 0.35, 0.18, 0.18, 0.10]
    }
}

# --- 3. SELECTION INITIALIZATION LAYER ---
with st.sidebar:
    st.title("📟 Teach Pendant Pro")
    with st.expander("🛠️ Layout Setup", expanded=True):
        selected_profile = st.selectbox(
            "Select Active Hardware Profile", 
            options=list(ROBOT_REGISTRY.keys()),
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

def get_link_transforms(angles):
    matrices = robot_chain.forward_kinematics(angles[:7], full_kinematics=True)
    transforms = []
    for i, m in enumerate(matrices):
        pos_val = m[:3, 3].tolist()
        if i == 4:
            m_corr = m.copy()
            m_corr[:3, 3] += m_corr[:3, 0] * -1.0
            pos_val = m_corr[:3, 3].tolist()
        rot_matrix = m[:3, :3]
        quat = R.from_matrix(rot_matrix).as_quat().tolist()
        transforms.append({"pos": pos_val, "quat": quat})
    return transforms

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
        
        g_off_x = st.slider("Gun Offset X (TCP)", -0.5, 0.5, 0.0, step=0.01)
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

if 'g_off_x' not in locals(): g_off_x = 0.0
if 'g_rot_z' not in locals(): g_rot_z = 180
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
        <style>
            body { margin: 0; background-color: #111111; overflow: hidden; font-family: sans-serif; user-select: none; }
            #canvas-container { width: 100vw; height: 100vh; position: absolute; top:0; left:0; z-index:1; }
            #status { position: absolute; top: 10px; left: 10px; color: #ffffff; font-size: 13px; background: rgba(20,20,20,0.8); padding: 6px 12px; border-radius:4px; border: 1px solid #333; z-index: 10; }
            #jog-pendant { position: absolute; top: 10px; right: 10px; background: rgba(20, 20, 20, 0.85); border: 1px solid #ff9800; border-radius: 6px; width: 220px; padding: 10px; color: white; z-index: 10; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            .pendant-title { font-size: 11px; font-weight: bold; text-transform: uppercase; color: #ff9800; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 4px; margin-bottom: 8px; text-align: center; }
            .jog-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
            .jog-label { font-size: 12px; font-weight: bold; font-family: monospace; color: #bbb; }
            .jog-btn { background: #222; border: 1px solid #444; color
