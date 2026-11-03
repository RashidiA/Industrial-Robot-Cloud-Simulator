import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import os
import json
import base64
try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink, URDFLink
except ImportError:      # ikpy is optional: the browser does the real IK now
    Chain = None

# --- 1. SYSTEM INITIALIZATION ---
st.set_page_config(page_title="Industrial Robot Cloud Simulator", page_icon="🤖", layout="wide")
st.markdown("""<style>
.block-container{padding-top:.6rem;padding-bottom:0;max-width:100%}
header[data-testid=stHeader]{background:transparent}
iframe{border-radius:10px;border:1px solid #262b35}
</style>""", unsafe_allow_html=True)

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

robot_chain = build_robot_chain(selected_profile, active_cfg) if Chain is not None else None

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
                "Nil": None,
                "Welding Guns": "welding_guns",
                "Grippers": "grippers",
                "Welding Torches": "welding_torches"
            }
            selected_category = st.selectbox("Select Tool Category Type", options=list(CATEGORY_MAPPING.keys()))
            
            if selected_category == "Nil":
                selected_tool_path = None
            else:
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
    
    html_source = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TransformControls.js"></script>

    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

    <style>
        :root { --bg:#0d0f13; --panel:#14171d; --panel2:#1a1e26; --line:#262b35; --txt:#e6e9ef; --mut:#8a93a3; --acc:#ff9800; --ok:#4caf50; --bad:#f44336; --cy:#00ffcc; }
        * { box-sizing: border-box; }
        html, body { height: 100%; }
        body { margin: 0; background: var(--bg); color: var(--txt); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; user-select: none; overflow: hidden; display: grid; grid-template-rows: 40px 1fr 26px; }
        button { font-family: inherit; }
        /* ---------- top bar ---------- */
        #topbar { display: flex; align-items: center; gap: 14px; padding: 0 12px; background: var(--panel); border-bottom: 1px solid var(--line); }
        .brand { font-weight: 800; letter-spacing: .5px; color: var(--acc); white-space: nowrap; }
        .brand small { color: var(--mut); font-weight: 500; margin-left: 6px; }
        .tb-sp { flex: 1; }
        .chip { background: var(--panel2); border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; color: var(--mut); font-size: 11px; white-space: nowrap; }
        .chip b { color: var(--txt); font-weight: 600; }
        .tbtn { background: var(--panel2); border: 1px solid var(--line); color: var(--txt); border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 12px; }
        .tbtn:hover { border-color: var(--acc); }
        /* ---------- main grid ---------- */
        #main { display: grid; grid-template-columns: 210px minmax(0, 1fr) 340px; min-height: 0; }
        #tree { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; padding: 10px 8px; }
        #side { background: var(--panel); border-left: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0; }
        .h { font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase; color: var(--mut); margin: 12px 6px 6px; }
        .h:first-child { margin-top: 2px; }
        .node { display: flex; align-items: center; gap: 8px; padding: 7px 8px; border-radius: 6px; color: var(--txt); cursor: default; }
        .node:hover { background: var(--panel2); }
        .node .nm { flex: 1; }
        .eye { cursor: pointer; opacity: .9; font-size: 14px; }
        .eye.off { opacity: .25; }
        .infobox { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; line-height: 1.7; margin: 0 2px; }
        .infobox span { color: var(--mut); display: inline-block; width: 52px; }
        .note { color: var(--mut); font-size: 11px; line-height: 1.5; margin: 10px 6px; }
        /* ---------- viewport ---------- */
        #view { position: relative; min-width: 0; min-height: 0; background: radial-gradient(ellipse at 50% 30%, #1b2029 0%, #0d0f13 70%); }
        #canvas-container { position: absolute; inset: 0; z-index: 1; }
        #viewbtns { position: absolute; top: 10px; left: 10px; z-index: 10; display: flex; gap: 6px; flex-wrap: wrap; }
        #status { position: absolute; top: 10px; right: 10px; z-index: 10; color: #fff; font-size: 12px; background: rgba(20,22,28,.85); padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line); transition: background .2s; }
        #toast { position: absolute; left: 50%; bottom: 16px; transform: translateX(-50%); z-index: 30; background: rgba(20,22,28,.95); border: 1px solid var(--acc); color: #fff; padding: 8px 14px; border-radius: 8px; display: none; max-width: 80%; text-align: center; }
        #warn { position: absolute; left: 10px; bottom: 12px; z-index: 10; display: flex; gap: 6px; flex-wrap: wrap; }
        .pill { font-size: 11px; padding: 3px 9px; border-radius: 999px; border: 1px solid #2f6b34; color: #7fd884; background: rgba(76,175,80,.08); }
        .pill.bad { border-color: #8a2a24; color: #ff8a80; background: rgba(244,67,54,.1); }
        /* ---------- side panel tabs ---------- */
        #tabs { display: flex; border-bottom: 1px solid var(--line); }
        .tab { flex: 1; text-align: center; padding: 10px 4px; color: var(--mut); cursor: pointer; border-bottom: 2px solid transparent; font-weight: 600; }
        .tab.active { color: var(--acc); border-bottom-color: var(--acc); }
        .pane { display: none; padding: 12px; overflow-y: auto; flex: 1; min-height: 0; }
        .pane.active { display: block; }
        .mode-container { display: flex; gap: 4px; margin-bottom: 10px; }
        .mode-btn { flex: 1; background: var(--panel2); border: 1px solid var(--line); color: var(--mut); padding: 8px 4px; font-size: 11px; font-weight: 700; cursor: pointer; border-radius: 6px; text-transform: uppercase; }
        .mode-btn.active { background: var(--acc); color: #000; border-color: var(--acc); }
        .jog-row { display: grid; grid-template-columns: 30px 28px minmax(0,1fr) 28px 58px; align-items: center; gap: 6px; margin-bottom: 8px; }
        .jog-label { font-size: 12px; font-weight: 800; font-family: Consolas, monospace; color: #6fb3ff; }
        .jog-btn { background: var(--panel2); border: 1px solid var(--line); color: #fff; height: 26px; font-size: 15px; font-weight: 700; cursor: pointer; border-radius: 5px; padding: 0; }
        .jog-btn:active { background: var(--acc); color: #000; border-color: var(--acc); }
        .val-display { font-family: Consolas, monospace; font-size: 12px; color: var(--cy); text-align: right; }
        .val-display.near { color: var(--bad); }
        input[type=range] { width: 100%; accent-color: var(--acc); margin: 0; }
        .e1sep { border-top: 1px dashed var(--line); padding-top: 8px; margin-top: 4px; }
        #tcp-monitor { background: rgba(0,0,0,.35); border: 1px solid var(--line); padding: 8px; border-radius: 8px; font-family: Consolas, monospace; margin-bottom: 8px; display: none; }
        .tcp-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; text-align: center; margin-top: 4px; font-weight: 700; }
        .row { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
        .row label { color: var(--mut); font-size: 11px; }
        select { background: var(--panel2); color: var(--txt); border: 1px solid var(--line); border-radius: 5px; padding: 3px 6px; }
        .action-block { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 12px; display: flex; flex-direction: column; gap: 7px; }
        .btn-action { width: 100%; border: none; font-weight: 700; height: 34px; border-radius: 6px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center; gap: 6px; }
        #btn-save-step { background: var(--acc); color: #000; }
        #btn-run-sim { background: var(--ok); color: #fff; }
        #btn-run-sim.stop { background: #c62828; }
        #btn-clear-seq { background: #3a2220; color: #ff8a80; border: 1px solid #5a2e2a; }
        #btn-reset-pos { background: #2b3358; color: #cfd6ff; border: 1px solid #3f4a86; }
        .step-counter { font-size: 12px; font-family: Consolas, monospace; text-align: center; color: var(--mut); }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; color: var(--mut); padding: 4px 4px; border-bottom: 1px solid var(--line); }
        td { padding: 6px 4px; border-bottom: 1px solid #1d222b; font-family: Consolas, monospace; font-size: 11px; }
        tr.cur td { background: rgba(255,152,0,.15); }
        .mini { background: var(--panel2); border: 1px solid var(--line); color: #fff; border-radius: 4px; cursor: pointer; padding: 1px 6px; font-size: 11px; }
        .learn h4 { margin: 14px 0 4px; color: var(--acc); font-size: 12px; }
        .learn p { margin: 0 0 6px; color: #c3c9d4; line-height: 1.55; }
        /* ---------- status bar ---------- */
        #statusbar { display: flex; gap: 14px; align-items: center; padding: 0 12px; background: #0a0c10; border-top: 1px solid var(--line); font-family: Consolas, monospace; font-size: 11px; color: var(--mut); white-space: nowrap; overflow: hidden; }
        #statusbar b { color: var(--cy); font-weight: 600; }
        body.running #side .pane, body.running #tree { opacity: .45; pointer-events: none; }
        body.running #btn-run-sim { pointer-events: auto; opacity: 1; }
        /* ---------- gesture camera ---------- */
        .ar-viewport-container { position: absolute; bottom: 44px; left: 12px; width: 320px; height: 240px; border: 2px solid var(--acc); border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,.6); z-index: 12; display: none; overflow: hidden; transform: scaleX(-1); }
        #webcam-feedback { width: 100%; height: 100%; object-fit: cover; position: absolute; top: 0; left: 0; z-index: 11; }
        #ar-overlay-canvas { width: 100%; height: 100%; position: absolute; top: 0; left: 0; z-index: 12; pointer-events: none; }
        .ar-gesture-remarks-hud { position: absolute; top: 0; left: 0; width: 100%; background: rgba(0,0,0,.8); border-bottom: 1px solid var(--acc); color: #fff; font-size: 11px; text-align: center; padding: 5px 0; font-weight: 500; letter-spacing: .5px; z-index: 20; transform: scaleX(-1); }
        /* ---------- phone / small screens ---------- */
        @media (max-width: 900px) {
            body { grid-template-rows: 40px 1fr 24px; }
            #main { grid-template-columns: 1fr; grid-template-rows: minmax(0, 46%) minmax(0, 54%); }
            #tree { display: none; }
            #side { border-left: 0; border-top: 1px solid var(--line); }
            .chip, .brand small { display: none; }
            .tbtn { white-space: nowrap; padding: 4px 8px; }
            #viewbtns { max-width: 60%; }
            .ar-viewport-container { width: 200px; height: 150px; bottom: 10px; }
            #statusbar span:nth-child(n+4) { display: none; }
        }
    </style>
</head>
<body>
    <div id="topbar">
        <div class="brand">ROBOT OLP<small>Offline Programming Studio</small></div>
        <span class="chip">Robot <b id="chip-robot">-</b></span>
        <span class="chip">Mode <b id="chip-mode">Joint</b></span>
        <div class="tb-sp"></div>
        <button class="tbtn" id="btn-export" title="Download the recorded program as a text file">⬇ Export program</button>
        <button class="tbtn" id="btn-help" title="Keyboard shortcuts">? Help</button>
    </div>

    <div id="main">
        <div id="tree">
            <div class="h">Station</div>
            <div class="node"><span>🦾</span><span class="nm">Robot</span><span class="eye" data-vis="robot">👁</span></div>
            <div class="node"><span>🔧</span><span class="nm">Tool</span><span class="eye" data-vis="tool">👁</span></div>
            <div class="node"><span>⚙</span><span class="nm">Jig / positioner</span><span class="eye" data-vis="jig">👁</span></div>
            <div class="node"><span>▦</span><span class="nm">Floor grid</span><span class="eye" data-vis="grid">👁</span></div>
            <div class="h">Selected robot</div>
            <div class="infobox">
                <div><span>Model</span><b id="inf-model">-</b></div>
                <div><span>Axes</span>6 + E1 positioner</div>
                <div><span>Reach</span>~<b id="inf-reach">-</b> m</div>
            </div>
            <div class="h">Program</div>
            <div class="node" id="node-prog"><span>▶</span><span class="nm">Main program</span><span id="inf-steps" class="chip">0</span></div>
            <div class="note">Educational models. Link shapes come from your CAD files; use for learning, not for cell sign-off.</div>
        </div>

        <div id="view">
            <div id="canvas-container"></div>
            <div id="viewbtns">
                <button class="tbtn" data-view="iso">Iso</button><button class="tbtn" data-view="front">Front</button><button class="tbtn" data-view="side">Side</button><button class="tbtn" data-view="top">Top</button><button class="tbtn" data-view="fit">Fit</button>
            </div>
            <div id="status">WebGL Processing...</div>
            <div id="warn"><span class="pill" id="pill-lim">Joint limits OK</span><span class="pill" id="pill-reach">Within reach</span></div>
            <div id="toast"></div>
            <div class="ar-viewport-container" id="ar-viewport">
                <div class="ar-gesture-remarks-hud" id="ar-hud-text">PLACE HAND INSIDE BOX TO ENGAGE</div>
                <video id="webcam-feedback" autoplay playsinline></video>
                <canvas id="ar-overlay-canvas" width="320" height="240"></canvas>
            </div>
        </div>

        <div id="side">
            <div id="tabs"><div class="tab active" data-tab="jog">Jog</div><div class="tab" data-tab="prog">Program</div><div class="tab" data-tab="learn">Learn</div></div>

            <div class="pane active" id="pane-jog">
              <div id="jog-pendant">
                <div class="mode-container">
                    <button id="mode-joint" class="mode-btn active">Joint</button>
                    <button id="mode-tcp" class="mode-btn">⌖ TCP</button>
                    <button id="mode-gesture" class="mode-btn">✋ Gesture</button>
                </div>
                <div id="tcp-monitor">
                    <div style="color: var(--cy); font-size: 10px; text-align:center;">TCP LIVE MONITOR (METERS)</div>
                    <div class="tcp-grid">
                        <span style="color:#ff6b6b">X:<span id="lbl-tx">0.00</span></span>
                        <span style="color:#6bff6b">Y:<span id="lbl-ty">0.00</span></span>
                        <span style="color:#6b8bff">Z:<span id="lbl-tz">0.00</span></span>
                    </div>
                    <div style="color:var(--mut);font-size:10px;text-align:center;margin-top:6px;">Drag the arrows in the 3D view to move the tool.</div>
                </div>
                <div class="row" id="step-row"><label>Button step</label><select id="sel-step"><option>1</option><option selected>5</option><option>10</option><option>15</option></select><label>°</label></div>
                <div id="joint-jog-container"></div>
                <div class="action-block">
                    <button class="btn-action" id="btn-reset-pos">🔄 RESET JOINT POSITIONS (0°)</button>
                    <div class="jog-row" style="grid-template-columns: 50px minmax(0,1fr) 44px; margin:0;">
                        <div class="jog-label" style="color:var(--mut)">SPEED</div>
                        <input type="range" id="sld-speed" min="5" max="100" value="50" step="5">
                        <div class="val-display" id="val-speed" style="color:var(--acc);font-weight:700">50%</div>
                    </div>
                    <button class="btn-action" id="btn-save-step">💾 RECORD STEP POSITION</button>
                </div>
              </div>
            </div>

            <div class="pane" id="pane-prog">
                <div class="action-block" style="margin:0 0 10px;border:0;padding:0;">
                    <button class="btn-action" id="btn-run-sim">▶️ RUN SIMULATION</button>
                    <div class="row" style="margin:0"><label><input type="checkbox" id="chk-loop"> Loop</label><div style="flex:1"></div><span class="step-counter" id="lbl-steps">Steps: 0</span></div>
                    <button class="btn-action" id="btn-clear-seq">🗑️ CLEAR SEQUENCE</button>
                </div>
                <table><thead><tr><th>#</th><th>Joint targets (°)</th><th></th></tr></thead><tbody id="prog-body"></tbody></table>
                <div class="note" id="prog-empty">No steps yet. Jog the robot (or use gesture), then press Record.</div>
            </div>

            <div class="pane learn" id="pane-learn">
                <h4>Joint jog vs TCP jog</h4><p>Joint mode moves one axis (A1 to A6) at a time. TCP mode lets you drag the tool tip in space; the computer finds the joint angles for you (inverse kinematics).</p>
                <h4>What is the TCP?</h4><p>The Tool Centre Point is the working point of the tool, for example the tip of the welding electrode. Robot programs store where the TCP must go.</p>
                <h4>Joint limits</h4><p>Every axis has a travel range. The red value and the warning chip show when an axis is close to its end stop.</p>
                <h4>Singularity</h4><p>When axis A5 is near 0°, axes A4 and A6 line up and the wrist loses one direction of motion. Keep A5 away from 0° in welding moves.</p>
                <h4>Recording a program</h4><p>Move the robot, press Record, repeat. Run Simulation then blends smoothly from one recorded pose to the next. Use the speed slider to slow it down.</p>
                <h4>Gesture mode</h4><p>Hold your palm in the box for 2 seconds, move your hand to move the tool, roll the hand to turn A6, and pinch to record.</p>
                <h4>Shortcuts</h4><p>R record · Space run or stop · 1 to 6 select axis, then ← → to jog · I T F S fit views.</p>
            </div>
        </div>
    </div>

    <div id="statusbar"><span id="sb-tcp">TCP -</span><span id="sb-joints">Joints -</span><span id="sb-steps">Steps 0</span><span id="sb-speed">Speed 50%</span></div>


    <script>
        const data = JSON.parse(JSON.stringify(__PAYLOAD_STREAM__));
        const dh = data.dhConfig;
        const limitsConfig = data.jointLimits;

        let localJointAngles = [...data.initialAngles];
        let lastComputedTransforms = [];
        let embeddedTrajectory = [...data.trajectory];
        let activeJogMode = "joint";
        
        const J_STEP = 5 * (Math.PI / 180);

        (THREE.Object3D.DefaultUp || THREE.Object3D.DEFAULT_UP).set(0, 0, 1);
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

        const tcpAnchorPivot = new THREE.Object3D();
        scene.add(tcpAnchorPivot);

        const transformGizmo = new THREE.TransformControls(camera, renderer.domElement);
        transformGizmo.size = 0.65;
        transformGizmo.setMode("translate");
        transformGizmo.attach(tcpAnchorPivot);
        transformGizmo.visible = false;
        transformGizmo.enabled = false;
        scene.add(transformGizmo.isObject3D ? transformGizmo : transformGizmo.getHelper());

        transformGizmo.addEventListener('dragging-changed', function (e) {
            controls.enabled = !e.value;
        });

        transformGizmo.addEventListener('objectChange', function () {
            if ((activeJogMode !== "tcp" && activeJogMode !== "gesture") || runSimulation) return;
            executeCyclicInverseKinematics(tcpAnchorPivot.position);
        });

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

        let toolAdjustmentGroup = new THREE.Group();
        gunMesh.add(toolAdjustmentGroup);

        if(data.gunData && data.gunData.length > 0) {
            const geometry = loader.parse(base64ToArrayBuffer(data.gunData));
            geometry.center(); 
            const gunInternalMesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 }));
            gunInternalMesh.scale.set(0.001, 0.001, 0.001); 
            toolAdjustmentGroup.add(gunInternalMesh);
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
            computedTransforms.push({ pos: new THREE.Vector3(0,0,0).toArray(), quat: new THREE.Quaternion().toArray() });

            let m1 = getLinkStructureBaseMatrix(dh[0]).multiply(new THREE.Matrix4().makeRotationZ(angles[1]));
            currentMatrix.multiply(m1);
            computedTransforms.push({ pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(), quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray() });

            let m2 = getLinkStructureBaseMatrix(dh[1]).multiply(new THREE.Matrix4().makeRotationY(angles[2]));
            currentMatrix.multiply(m2);
            computedTransforms.push({ pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(), quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray() });

            let m2_adj = getLinkStructureBaseMatrix(dh[2]).multiply(new THREE.Matrix4().makeRotationY(angles[3]));
            currentMatrix.multiply(m2_adj);
            computedTransforms.push({ pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(), quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray() });

            let m4 = getLinkStructureBaseMatrix(dh[3]).multiply(new THREE.Matrix4().makeRotationX(angles[4]));
            currentMatrix.multiply(m4);
            
            if (data.profileName === "Yaskawa_3500") {
                computedTransforms.push({ pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(), quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray() });
            } else {
                let correctionMatrix = currentMatrix.clone();
                let directionVector = new THREE.Vector3(1, 0, 0).applyQuaternion(new THREE.Quaternion().setFromRotationMatrix(correctionMatrix));
                let fixedPos = new THREE.Vector3().setFromMatrixPosition(correctionMatrix).add(directionVector.multiplyScalar(-1.0));
                computedTransforms.push({ pos: fixedPos.toArray(), quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray() });
            }

            let m5 = getLinkStructureBaseMatrix(dh[4]).multiply(new THREE.Matrix4().makeRotationY(angles[5]));
            currentMatrix.multiply(m5);
            computedTransforms.push({ pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(), quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray() });

            let m6 = getLinkStructureBaseMatrix(dh[5]);
            if (data.profileName === "Yaskawa_3500") { m6.multiply(new THREE.Matrix4().makeRotationZ(angles[6])); }
            else { m6.multiply(new THREE.Matrix4().makeRotationX(angles[6])); }
            currentMatrix.multiply(m6);
            computedTransforms.push({ pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(), quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray() });

            return computedTransforms;
        }

        function executeCyclicInverseKinematics(targetGlobalPos) {
            if (links[6]) {
                let physicalWristPos = new THREE.Vector3().setFromMatrixPosition(links[6].matrixWorld);
                let deviationDistance = targetGlobalPos.distanceTo(physicalWristPos);
                
                if (deviationDistance > 0.10) {
                    let allowedDirection = new THREE.Vector3().subVectors(targetGlobalPos, physicalWristPos).normalize();
                    targetGlobalPos.copy(physicalWristPos).add(allowedDirection.multiplyScalar(0.10));
                    tcpAnchorPivot.position.copy(targetGlobalPos);
                }
            }

            let maxReachRadius = 2.95; 
            if (data.profileName === "ABB_4400") maxReachRadius = 1.95;
            if (data.profileName === "Yaskawa_3500") maxReachRadius = 2.45;

            let distanceFromBase = targetGlobalPos.length();
            if (distanceFromBase > maxReachRadius) {
                targetGlobalPos.normalize().multiplyScalar(maxReachRadius);
                tcpAnchorPivot.position.copy(targetGlobalPos);
            }

            for (let iteration = 0; iteration < 12; iteration++) {
                let currentTransforms = computeForwardKinematics(localJointAngles);
                let endEffectorPos = new THREE.Vector3().fromArray(currentTransforms[6].pos);
                let errorDistance = new THREE.Vector3().copy(targetGlobalPos).sub(endEffectorPos);
                if (errorDistance.length() < 0.0002) break;

                for (let j = 1; j <= 6; j++) {
                    if (activeJogMode === "gesture" && j === 6) continue;

                    let jointPosition = new THREE.Vector3().fromArray(currentTransforms[j-1].pos);
                    let axisVectorDirection = new THREE.Vector3(0, 0, 1);
                    if (j === 2 || j === 3 || j === 5) axisVectorDirection.set(0, 1, 0);
                    if (j === 4 || (j === 6 && data.profileName !== "Yaskawa_3500")) axisVectorDirection.set(1, 0, 0);

                    let componentToEE = new THREE.Vector3().subVectors(endEffectorPos, jointPosition).normalize();
                    let componentToTarget = new THREE.Vector3().subVectors(targetGlobalPos, jointPosition).normalize();

                    let angleScalarDot = componentToEE.dot(componentToTarget);
                    let localizedDeltaTheta = Math.acos(Math.max(-1, Math.min(1, angleScalarDot))) * 0.12;
                    
                    if (j === 4) {
                        localizedDeltaTheta *= 0.15;
                        if (Math.abs(localJointAngles[5]) < 0.10) {
                            localizedDeltaTheta *= 0.05;
                        }
                    }
                    if (j === 5) localizedDeltaTheta *= 0.25; 
                    if (j === 6) localizedDeltaTheta *= 0.35;

                    if (localizedDeltaTheta > 0.0001) {
                        let directionalCross = new THREE.Vector3().crossVectors(componentToEE, componentToTarget);
                        if (directionalCross.dot(axisVectorDirection) < 0) { localJointAngles[j] -= localizedDeltaTheta; }
                        else { localJointAngles[j] += localizedDeltaTheta; }
                        localJointAngles[j] = Math.max(limitsConfig[j-1][0], Math.min(limitsConfig[j-1][1], localJointAngles[j]));
                    }
                }
            }
            refreshSceneDisplay(false); 
            updateMonitorHUDText(targetGlobalPos);
        }

        function refreshSceneDisplay(updateGizmoPosition = true) {
            lastComputedTransforms = computeForwardKinematics(localJointAngles);
            for(let i=0; i<7; i++) {
                if(links[i] && lastComputedTransforms[i]) {
                    links[i].position.fromArray(lastComputedTransforms[i].pos);
                    links[i].quaternion.fromArray(lastComputedTransforms[i].quat);
                }
            }
            if(links[6]) {
                links[6].updateMatrixWorld(); gunMesh.position.copy(links[6].position); gunMesh.quaternion.copy(links[6].quaternion);
                gunMesh.translateX(data.toolOffsetX); gunMesh.translateY(data.toolOffsetY); gunMesh.translateZ(data.toolOffsetZ);
                toolAdjustmentGroup.rotation.set(data.toolRotX, data.toolRotY, data.toolRotZ, 'XYZ');

                if (updateGizmoPosition || runSimulation) {
                    tcpAnchorPivot.position.copy(links[6].position);
                    tcpAnchorPivot.quaternion.copy(links[6].quaternion);
                    transformGizmo.updateMatrixWorld();
                    updateMonitorHUDText(tcpAnchorPivot.position);
                }
            }
            jigMesh.position.set(data.jigX, data.jigY, data.jigZ);
            internalJigContent.rotation.z = localJointAngles[7];

            for(let i=1; i<=7; i++) {
                let displayElement = document.getElementById('val-' + i);
                if (displayElement) { displayElement.innerText = (localJointAngles[i] * (180 / Math.PI)).toFixed(1) + '°'; }
            }
        }

        function triggerManualStepRecord() {
            if(runSimulation) return;
            lastComputedTransforms = computeForwardKinematics(localJointAngles);
            embeddedTrajectory.push({ angles: [...localJointAngles], transforms: JSON.parse(JSON.stringify(lastComputedTransforms)) });
            updateUIElements();
            syncProgramToParent('sync_sequence');
        }

        function updateMonitorHUDText(vectorPos) {
            document.getElementById("lbl-tx").innerText = vectorPos.x.toFixed(2);
            document.getElementById("lbl-ty").innerText = vectorPos.y.toFixed(2);
            document.getElementById("lbl-tz").innerText = vectorPos.z.toFixed(2);
        }

        const btnJoint = document.getElementById("mode-joint");
        const btnTCP = document.getElementById("mode-tcp");
        const btnGesture = document.getElementById("mode-gesture");
        const rowsWrap = document.getElementById("joint-jog-container");
        const hudMonitor = document.getElementById("tcp-monitor");
        
        const arContainer = document.getElementById("ar-viewport");
        const videoElement = document.getElementById("webcam-feedback");

        btnJoint.addEventListener("click", () => {
            activeJogMode = "joint"; 
            btnJoint.classList.add("active"); btnTCP.classList.remove("active"); btnGesture.classList.remove("active");
            rowsWrap.style.display = "block"; hudMonitor.style.display = "none"; arContainer.style.display = "none";
            transformGizmo.visible = false; transformGizmo.enabled = false;
            stopWebcam();
            refreshSceneDisplay(true); 
        });

        btnTCP.addEventListener("click", () => {
            activeJogMode = "tcp"; 
            btnTCP.classList.add("active"); btnJoint.classList.remove("active"); btnGesture.classList.remove("active");
            rowsWrap.style.display = "none"; hudMonitor.style.display = "block"; arContainer.style.display = "none";
            transformGizmo.visible = true; transformGizmo.enabled = true; 
            refreshSceneDisplay(true); 
            stopWebcam();
        });

        btnGesture.addEventListener("click", () => {
            activeJogMode = "gesture";
            btnGesture.classList.add("active"); btnJoint.classList.remove("active"); btnTCP.classList.remove("active");
            rowsWrap.style.display = "none"; hudMonitor.style.display = "block"; arContainer.style.display = "block";
            transformGizmo.visible = true; transformGizmo.enabled = true; 
            refreshSceneDisplay(true);
            startWebcam();
        });

        function jogJoint(jointIdx, direction) {
            if(runSimulation) return;
            let nextAngle = localJointAngles[jointIdx] + (direction * J_STEP);
            if (nextAngle >= limitsConfig[jointIdx-1][0] && nextAngle <= limitsConfig[jointIdx-1][1]) {
                localJointAngles[jointIdx] = nextAngle;
                refreshSceneDisplay(true); 
            }
        }

        document.getElementById('btn-reset-pos').addEventListener('click', () => {
            if(runSimulation) return;
            for(let i = 1; i <= 6; i++) { localJointAngles[i] = 0.0; }
            refreshSceneDisplay(true);
            syncProgramToParent('reset_joints');
        });

        var updateUIElements = function() { document.getElementById('lbl-steps').innerText = "Steps: " + embeddedTrajectory.length; }
        document.getElementById('sld-speed').addEventListener('input', (e) => { document.getElementById('val-speed').innerText = e.target.value + "%"; });

        document.getElementById('btn-save-step').addEventListener('click', () => { triggerManualStepRecord(); toast('Step ' + embeddedTrajectory.length + ' recorded'); });

        document.getElementById('btn-clear-seq').addEventListener('click', () => {
            if (runSimulation) return;
            embeddedTrajectory = []; updateUIElements();
            syncProgramToParent('clear_sequence');
        });

        let simStepIndex = 0; let interpolationFraction = 0; let runSimulation = false;

        function animate() {
            requestAnimationFrame(animate);
            controls.update();

            if (runSimulation && embeddedTrajectory.length >= 2) {
                document.getElementById('status').innerText = "Status: Running Sequence Simulation";
                let currentPoint = embeddedTrajectory[simStepIndex];
                let nextPoint = embeddedTrajectory[simStepIndex + 1];
                let computedStepIncrement = (parseFloat(document.getElementById('sld-speed').value) / 100) * 0.04;

                interpolationFraction += computedStepIncrement;
                if(interpolationFraction >= 1.0) {
                    interpolationFraction = 0; simStepIndex++;
                    if (simStepIndex >= embeddedTrajectory.length - 1) { if (!document.getElementById('chk-loop').checked) { runSimulation = false; } simStepIndex = 0; }
                }

                let blendedTransforms = [];
                for(let i=0; i<7; i++) {
                    let blendedPos = new THREE.Vector3().lerpVectors(new THREE.Vector3().fromArray(currentPoint.transforms[i].pos), new THREE.Vector3().fromArray(nextPoint.transforms[i].pos), interpolationFraction).toArray();
                    let blendedQuat = new THREE.Quaternion().fromArray(currentPoint.transforms[i].quat).slerp(new THREE.Quaternion().fromArray(nextPoint.transforms[i].quat), interpolationFraction).toArray();
                    blendedTransforms.push({ "pos": blendedPos, "quat": blendedQuat });
                }
                localJointAngles = [...currentPoint.angles]; 
                updateSceneTransforms(blendedTransforms, data.toolOffsetX, data.toolOffsetY, data.toolOffsetZ, data.toolRotX, data.toolRotY, data.toolRotZ, data.jigX, data.jigY, data.jigZ, ((1 - interpolationFraction) * currentPoint.angles[7] + interpolationFraction * nextPoint.angles[7]));
            } else {
                if(activeJogMode === "gesture") {
                    document.getElementById('status').innerText = "Status: Hybrid Gesture Control Active";
                } else {
                    document.getElementById('status').innerText = "Status: Online (WebGL Ready)";
                }
            }
            renderer.render(scene, camera);
        }

        function updateSceneTransforms(transforms, ox, oy, oz, rx, ry, rz, jigX, jigY, jigZ, e1RotAngle) {
            for(let i=0; i<7; i++) {
                if(links[i] && transforms[i]) { links[i].position.fromArray(transforms[i].pos); links[i].quaternion.fromArray(transforms[i].quat); }
            }
            if(links[6]) {
                links[6].updateMatrixWorld(); gunMesh.position.copy(links[6].position); gunMesh.quaternion.copy(links[6].quaternion);
                gunMesh.translateX(ox); gunMesh.translateY(oy); gunMesh.translateZ(oz);
                toolAdjustmentGroup.rotation.set(rx, ry, rz, 'XYZ');
                
                tcpAnchorPivot.position.copy(links[6].position);
                tcpAnchorPivot.quaternion.copy(links[6].quaternion);
                transformGizmo.updateMatrixWorld();
            }
            jigMesh.position.set(jigX, jigY, jigZ); internalJigContent.rotation.z = e1RotAngle;
        }

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });

        let mpHands = null;
        let mpCamera = null;
        const arCanvas = document.getElementById("ar-overlay-canvas");
        const arCtx = arCanvas.getContext("2d");
        
        let anchorTCPPos = new THREE.Vector3();
        let isPinchActive = false; 
        let lastPinchTime = 0;
        
        let isHandInsideBox = false;
        let handEngagementStartTime = 0;
        let isHandEngaged = false; 
        
        let handCalibrationBaseline = null; 
        let robotCalibrationBaseline = null;

        const boxMinX = 0.33;
        const boxMaxX = 0.67;
        const boxMinY = 0.30;
        const boxMaxY = 0.70;

        function onHandResults(results) {
            arCtx.clearRect(0, 0, arCanvas.width, arCanvas.height);
            if (activeJogMode !== "gesture" || runSimulation) return;

            arCtx.lineWidth = 2;
            if (isHandEngaged) {
                arCtx.strokeStyle = "#4caf50"; 
                arCtx.fillStyle = "rgba(76, 175, 80, 0.05)";
            } else if (isHandInsideBox) {
                arCtx.strokeStyle = "#ffeb3b"; 
                arCtx.fillStyle = "rgba(255, 235, 59, 0.05)";
            } else {
                arCtx.strokeStyle = "#ff9800"; 
                arCtx.fillStyle = "rgba(255, 152, 0, 0.03)";
            }
            
            const bx = boxMinX * arCanvas.width;
            const by = boxMinY * arCanvas.height;
            const bw = (boxMaxX - boxMinX) * arCanvas.width;
            const bh = (boxMaxY - boxMinY) * arCanvas.height;
            arCtx.fillRect(bx, by, bw, bh);
            arCtx.strokeRect(bx, by, bw, bh);
            
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                const handLandmarks = results.multiHandLandmarks[0];
                const palmCenter = handLandmarks[9];
                const wrist = handLandmarks[0];
                
                const insideBox = (palmCenter.x >= boxMinX && palmCenter.x <= boxMaxX && palmCenter.y >= boxMinY && palmCenter.y <= boxMaxY);
                const currentTimeMillis = Date.now();

                let currentRawX = 0.8 + (1.0 - palmCenter.z) * 0.9;
                let currentRawY = -(palmCenter.x - 0.5) * 1.6;
                let currentRawZ = 0.3 + (1.0 - palmCenter.y) * 1.0;
                let rawHandVec = new THREE.Vector3(currentRawX, currentRawY, currentRawZ);

                if (!isHandEngaged) {
                    if (insideBox) {
                        if (!isHandInsideBox) {
                            isHandInsideBox = true;
                            handEngagementStartTime = currentTimeMillis;
                        }
                        
                        let secondsElapsed = (currentTimeMillis - handEngagementStartTime) / 1000;
                        let remainingTime = Math.max(0, (2.0 - secondsElapsed)).toFixed(1);
                        
                        if (secondsElapsed >= 2.0) {
                            isHandEngaged = true; 
                            document.getElementById("ar-hud-text").innerHTML = "🟢 TRACKING ACTIVE: MOVE PALM";
                            document.getElementById("ar-hud-text").style.color = "#4caf50";
                            
                            handCalibrationBaseline = rawHandVec.clone();
                            if (links[6]) {
                                robotCalibrationBaseline = new THREE.Vector3().setFromMatrixPosition(links[6].matrixWorld);
                                tcpAnchorPivot.position.copy(robotCalibrationBaseline);
                            }
                        } else {
                            document.getElementById("ar-hud-text").innerHTML = "⏳ INITIALIZING: HOLD STILL (" + remainingTime + "s)";
                            document.getElementById("ar-hud-text").style.color = "#ffeb3b";
                            
                            if (links[6]) {
                                let currentRobotPos = new THREE.Vector3().setFromMatrixPosition(links[6].matrixWorld);
                                tcpAnchorPivot.position.copy(currentRobotPos);
                            }
                        }
                    } else {
                        isHandInsideBox = false;
                        handCalibrationBaseline = null;
                        robotCalibrationBaseline = null;
                        document.getElementById("ar-hud-text").innerHTML = "⚠️ PLACE PALM INSIDE THE CENTER BOX";
                        document.getElementById("ar-hud-text").style.color = "#ff9800";
                        
                        if (links[6]) {
                            tcpAnchorPivot.position.copy(new THREE.Vector3().setFromMatrixPosition(links[6].matrixWorld));
                        }
                    }
                } else {
                    if (palmCenter.x < 0.05 || palmCenter.x > 0.95 || palmCenter.y < 0.05 || palmCenter.y > 0.95) {
                        isHandEngaged = false; 
                        isHandInsideBox = false;
                        handCalibrationBaseline = null;
                        robotCalibrationBaseline = null;
                    }
                }

                const screenX = palmCenter.x * arCanvas.width;
                const screenY = palmCenter.y * arCanvas.height;

                if (isHandEngaged && handCalibrationBaseline && robotCalibrationBaseline) {
                    let deltaMove = new THREE.Vector3().subVectors(rawHandVec, handCalibrationBaseline);
                    let calculatedTarget = new THREE.Vector3().addVectors(robotCalibrationBaseline, deltaMove);
                    
                    tcpAnchorPivot.position.lerp(calculatedTarget, 0.10);
                    executeCyclicInverseKinematics(tcpAnchorPivot.position);

                    const p5 = handLandmarks[5];   
                    const p17 = handLandmarks[17]; 
                    let currentRoll = Math.atan2(p17.y - p5.y, p17.x - p5.x);
                    
                    if (window.lastHandRoll === undefined) { window.lastHandRoll = currentRoll; }
                    let deltaRoll = currentRoll - window.lastHandRoll;
                    if (deltaRoll > Math.PI) deltaRoll -= 2 * Math.PI;
                    if (deltaRoll < -Math.PI) deltaRoll += 2 * Math.PI;
                    window.lastHandRoll = currentRoll;

                    if (Math.abs(deltaRoll) > 0.005) {
                        localJointAngles[6] += deltaRoll * 0.45; 
                        localJointAngles[6] = Math.max(limitsConfig[5][0], Math.min(limitsConfig[5][1], localJointAngles[6]));
                    }

                    const thumbTip = handLandmarks[4];
                    const indexTip = handLandmarks[8];
                    let dx = thumbTip.x - indexTip.x;
                    let dy = thumbTip.y - indexTip.y;
                    let dz = thumbTip.z - indexTip.z;
                    let pinchDistance = Math.sqrt(dx*dx + dy*dy + dz*dz);

                    if (pinchDistance < 0.045) {
                        if (!isPinchActive) {
                            let currentTime = Date.now();
                            if (currentTime - lastPinchTime > 1200) { 
                                isPinchActive = true;
                                lastPinchTime = currentTime;
                                triggerManualStepRecord();
                                document.getElementById('status').style.background = "rgba(76, 175, 80, 0.95)";
                                setTimeout(() => { document.getElementById('status').style.background = "rgba(20,20,20,0.8)"; }, 400);
                            }
                        }
                    } else {
                        isPinchActive = false;
                    }

                    const baseScreenX = wrist.x * arCanvas.width;
                    const baseScreenY = wrist.y * arCanvas.height;
                    const dirX = screenX - baseScreenX;
                    const dirY = screenY - baseScreenY;
                    const distance = Math.sqrt(dirX * dirX + dirY * dirY);
                    const uX = dirX / (distance || 1);
                    const uY = dirY / (distance || 1);
                    const nX = -uY;
                    const nY = uX;
                    const arrowLength = Math.max(30, distance * 0.7);

                    arCtx.lineWidth = 4;
                    arCtx.lineCap = "round";
                    arCtx.strokeStyle = "#2196f3";
                    arCtx.beginPath(); arCtx.moveTo(screenX, screenY); arCtx.lineTo(screenX + uX * arrowLength, screenY + uY * arrowLength); arCtx.stroke();

                    arCtx.strokeStyle = "#f44336";
                    arCtx.beginPath(); arCtx.moveTo(screenX, screenY); arCtx.lineTo(screenX + nX * (arrowLength * 0.7), screenY + nY * (arrowLength * 0.7)); arCtx.stroke();
                }

                arCtx.fillStyle = isHandEngaged ? (isPinchActive ? "#4caf50" : "#00ffcc") : (isHandInsideBox ? "#ffeb3b" : "#ff9800");
                arCtx.shadowColor = arCtx.fillStyle;
                arCtx.shadowBlur = 10;
                arCtx.beginPath(); arCtx.arc(screenX, screenY, 6, 0, 2 * Math.PI); arCtx.fill();
                arCtx.shadowBlur = 0;
            } else {
                if (isHandInsideBox || isHandEngaged) {
                    isHandEngaged = false;
                    isHandInsideBox = false;
                    handCalibrationBaseline = null;
                    robotCalibrationBaseline = null;
                    document.getElementById("ar-hud-text").innerHTML = "PLACE HAND INSIDE BOX TO ENGAGE";
                    document.getElementById("ar-hud-text").style.color = "#ff9800";
                }
            }
        }

        function startWebcam() {
            isHandEngaged = false;
            isHandInsideBox = false;
            handCalibrationBaseline = null;
            robotCalibrationBaseline = null;
            if (!mpHands) {
                mpHands = new Hands({ locateFile: (file) => 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/' + file });
                mpHands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });
                mpHands.onResults(onHandResults);
            }
            if (!mpCamera) {
                mpCamera = new Camera(videoElement, {
                    onFrame: async () => { if (activeJogMode === "gesture") { await mpHands.send({ image: videoElement }); } },
                    width: 320, height: 240
                });
            }
            mpCamera.start().catch(err => { alert("Camera access denied or unavailable: " + err); });
        }

        function stopWebcam() { 
            isHandEngaged = false;
            isHandInsideBox = false;
            handCalibrationBaseline = null;
            robotCalibrationBaseline = null;
            if (mpCamera) { mpCamera.stop(); mpCamera = null; } 
        }

        /* ======================= UI LAYER (added; kinematics above is unchanged) ======================= */
        const $ = (id) => document.getElementById(id);
        const D2R = Math.PI / 180, R2D = 180 / Math.PI;
        const REACH = { ABB_6700: 2.65, ABB_4400: 1.95, ABB_6600: 2.55, KUKA_KR150: 3.1, Yaskawa_3500: 2.45 };
        $('chip-robot').innerText = data.profileName.replace('_', ' ');
        $('inf-model').innerText = data.profileName.replace('_', ' ');
        $('inf-reach').innerText = (REACH[data.profileName] || 2.5).toFixed(2);

        function toast(msg) { const t = $('toast'); t.innerText = msg; t.style.display = 'block'; clearTimeout(toast.h); toast.h = setTimeout(() => t.style.display = 'none', 3500); }

        // tabs
        document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
            document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + t.dataset.tab));
            if (t.dataset.tab === 'prog') renderProgram();
        }));

        // visibility toggles
        const visTargets = { robot: () => links, tool: () => [gunMesh], jig: () => [jigMesh], grid: () => [grid] };
        document.querySelectorAll('.eye').forEach(e => e.addEventListener('click', () => {
            const on = e.classList.toggle('off') === false;
            visTargets[e.dataset.vis]().forEach(o => o.visible = on);
        }));

        // camera views
        function setView(name) {
            const t = controls.target;
            if (name === 'iso' || name === 'fit') { t.set(0.8, 0, 0.8); camera.position.set(4.0, -4.0, 3.0); }
            if (name === 'front') { t.set(0.8, 0, 0.8); camera.position.set(0.8, -6, 1.0); }
            if (name === 'side') { t.set(0.8, 0, 0.8); camera.position.set(6.5, 0, 1.0); }
            if (name === 'top') { t.set(0.8, 0, 0); camera.position.set(0.8, 0.001, 7); }
            camera.up.set(0, 0, 1); controls.update();
        }
        document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));

        // joint rows: slider + minus/plus + value (ids val-i, btn-m-i, btn-p-i are the originals)
        function setJointAbs(i, ang) {
            if (runSimulation) return;
            ang = Math.max(limitsConfig[i - 1][0], Math.min(limitsConfig[i - 1][1], ang));
            localJointAngles[i] = ang;
            refreshSceneDisplay(true);
        }
        function stepRad() { return parseFloat($('sel-step').value) * D2R; }
        function jogJointUI(i, dir) { setJointAbs(i, localJointAngles[i] + dir * stepRad()); }
        function buildJogRows() {
            const wrap = $('joint-jog-container'); wrap.innerHTML = '';
            for (let i = 1; i <= 7; i++) {
                const row = document.createElement('div'); row.className = 'jog-row' + (i === 7 ? ' e1sep' : '');
                const lo = Math.round(limitsConfig[i - 1][0] * R2D), hi = Math.round(limitsConfig[i - 1][1] * R2D);
                row.innerHTML = '<div class="jog-label"' + (i === 7 ? ' style="color:var(--acc)"' : '') + '>' + (i === 7 ? 'E1' : 'A' + i) + '</div>' +
                    '<button class="jog-btn" id="btn-m-' + i + '">-</button>' +
                    '<input type="range" id="sld-j-' + i + '" min="' + lo + '" max="' + hi + '" step="0.5" value="0">' +
                    '<button class="jog-btn" id="btn-p-' + i + '">+</button>' +
                    '<div class="val-display" id="val-' + i + '">0.0°</div>';
                wrap.appendChild(row);
                $('btn-m-' + i).addEventListener('click', () => jogJointUI(i, -1));
                $('btn-p-' + i).addEventListener('click', () => jogJointUI(i, 1));
                $('sld-j-' + i).addEventListener('input', (e) => setJointAbs(i, parseFloat(e.target.value) * D2R));
            }
        }
        buildJogRows();

        // live read-outs (called every frame; cheap)
        let lastSig = '';
        function updateReadouts() {
            const a = localJointAngles;
            let near = false;
            for (let i = 1; i <= 7; i++) {
                const s = $('sld-j-' + i); if (s && document.activeElement !== s) s.value = (a[i] * R2D).toFixed(1);
                const lo = limitsConfig[i - 1][0], hi = limitsConfig[i - 1][1], span = hi - lo;
                const isNear = (a[i] - lo) < 0.03 * span || (hi - a[i]) < 0.03 * span;
                const v = $('val-' + i); if (v) v.classList.toggle('near', isNear);
                if (isNear && i <= 6) near = true;
            }
            const pl = $('pill-lim'); pl.className = 'pill' + (near ? ' bad' : ''); pl.innerText = near ? 'Joint near limit' : 'Joint limits OK';
            if (links[6]) {
                const p = links[6].position; const sh = links[2] ? links[2].position : new THREE.Vector3(); const r = Math.hypot(p.x - sh.x, p.y - sh.y, p.z - sh.z);   // reach is measured from the shoulder (A2)
                const out = r > (REACH[data.profileName] || 2.5);
                const pr = $('pill-reach'); pr.className = 'pill' + (out ? ' bad' : ''); pr.innerText = out ? 'Beyond reach' : 'Within reach';
                const sig = [p.x, p.y, p.z, a.join()].join('|');
                if (sig !== lastSig) {
                    lastSig = sig;
                    $('sb-tcp').innerHTML = 'TCP <b>X ' + (p.x * 1000).toFixed(0) + ' Y ' + (p.y * 1000).toFixed(0) + ' Z ' + (p.z * 1000).toFixed(0) + '</b> mm';
                    $('sb-joints').innerHTML = 'Joints <b>' + a.slice(1, 7).map(v => (v * R2D).toFixed(0)).join(' ') + '</b> · E1 <b>' + (a[7] * R2D).toFixed(0) + '</b>';
                }
            }
            $('sb-steps').innerText = 'Steps ' + embeddedTrajectory.length;
            $('sb-speed').innerText = 'Speed ' + $('sld-speed').value + '%';
            $('chip-mode').innerText = activeJogMode === 'joint' ? 'Joint' : activeJogMode === 'tcp' ? 'TCP' : 'Gesture';
        }

        // program table
        function syncProgramToParent(evt) {
            try {
                const u = new URL(window.parent.location.href);
                u.searchParams.set('event', evt);
                if (evt === 'sync_sequence') u.searchParams.set('program_data', JSON.stringify(embeddedTrajectory));
                else u.searchParams.delete('program_data');
                window.parent.history.replaceState({}, '', u.toString());   // no page reload: Streamlit reads it on its next rerun
            } catch (e) { /* parent not reachable: the program still works locally */ }
        }
        function renderProgram() {
            const tb = $('prog-body'); tb.innerHTML = '';
            embeddedTrajectory.forEach((st, i) => {
                const tr = document.createElement('tr'); tr.dataset.i = i;
                tr.innerHTML = '<td>' + (i + 1) + '</td><td>' + st.angles.slice(1, 7).map(v => (v * R2D).toFixed(0)).join(' ') + (Math.abs(st.angles[7]) > 0.01 ? ' · E1 ' + (st.angles[7] * R2D).toFixed(0) : '') + '</td>' +
                    '<td style="white-space:nowrap"><button class="mini" data-go="' + i + '" title="Move robot to this step">⇥</button> <button class="mini" data-del="' + i + '" title="Delete step">✕</button></td>';
                tb.appendChild(tr);
            });
            $('prog-empty').style.display = embeddedTrajectory.length ? 'none' : 'block';
            $('inf-steps').innerText = embeddedTrajectory.length;
        }
        $('prog-body').addEventListener('click', (e) => {
            const g = e.target.getAttribute('data-go'), d = e.target.getAttribute('data-del');
            if (g !== null && !runSimulation) { localJointAngles = [...embeddedTrajectory[+g].angles]; refreshSceneDisplay(true); toast('Moved to step ' + (+g + 1)); }
            if (d !== null && !runSimulation) { embeddedTrajectory.splice(+d, 1); updateUIElements(); syncProgramToParent(embeddedTrajectory.length ? 'sync_sequence' : 'clear_sequence'); }
        });
        const _origUpdateUI = updateUIElements;
        updateUIElements = function () { _origUpdateUI(); renderProgram(); };

        // export
        $('btn-export').addEventListener('click', () => {
            if (!embeddedTrajectory.length) { toast('Nothing to export yet. Record some steps first.'); return; }
            let t = '! Robot program exported from Industrial Robot Cloud Simulator\n! Robot: ' + data.profileName + '\n! Joint targets in degrees (A1..A6), E1 positioner\n\n';
            embeddedTrajectory.forEach((s, i) => {
                const q = s.angles.slice(1, 7).map(v => (v * R2D).toFixed(2)).join(', ');
                t += 'MoveAbsJ  [' + q + '], E1=' + (s.angles[7] * R2D).toFixed(2) + ', speed=' + $('sld-speed').value + '%;   ! step ' + (i + 1) + '\n';
            });
            const url = URL.createObjectURL(new Blob([t], { type: 'text/plain' }));
            const a = document.createElement('a'); a.href = url; a.download = data.profileName + '_program.txt'; a.click(); URL.revokeObjectURL(url);
        });
        $('btn-help').addEventListener('click', () => { document.querySelector('.tab[data-tab="learn"]').click(); });

        // run / stop button
        function toggleRun() {
            if (runSimulation) { runSimulation = false; simStepIndex = 0; interpolationFraction = 0; return; }
            if (embeddedTrajectory.length < 2) { toast('Record at least 2 steps first.'); return; }
            simStepIndex = 0; interpolationFraction = 0; runSimulation = true;
        }
        $('btn-run-sim').addEventListener('click', toggleRun);
        function updateRunUI() {
            document.body.classList.toggle('running', runSimulation);
            const b = $('btn-run-sim'); b.classList.toggle('stop', runSimulation);
            b.innerText = runSimulation ? '⏹ STOP SIMULATION' : '▶️ RUN SIMULATION';
        }

        // keyboard
        let selAxis = 1;
        window.addEventListener('keydown', (e) => {
            if (e.target && (e.target.tagName === 'INPUT' && e.target.type !== 'range' || e.target.tagName === 'SELECT')) return;
            const k = e.key.toLowerCase();
            if (k === 'r') { triggerManualStepRecord(); toast('Step recorded'); }
            else if (k === ' ') { e.preventDefault(); toggleRun(); }
            else if (k >= '1' && k <= '6') { selAxis = +k; toast('Axis A' + k + ' selected: use ← → to jog'); }
            else if (k === 'arrowleft' && activeJogMode === 'joint') { jogJointUI(selAxis, -1); e.preventDefault(); }
            else if (k === 'arrowright' && activeJogMode === 'joint') { jogJointUI(selAxis, 1); e.preventDefault(); }
            else if (k === 'i') setView('iso'); else if (k === 'f') setView('front'); else if (k === 's') setView('side'); else if (k === 't') setView('top');
        });

        // keep the 3D canvas sized to its panel
        if (window.ResizeObserver) new ResizeObserver(() => {
            if (!container.clientWidth || !container.clientHeight) return;
            camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }).observe(container);

        setInterval(updateReadouts, 120);
        setInterval(updateRunUI, 150);
        updateReadouts();

        refreshSceneDisplay(true); updateUIElements(); animate();
    </script>
</body>
</html>"""
    
    html_source = html_source.replace("__PAYLOAD_STREAM__", json_stream)
    components.html(html_source, height=820, scrolling=False)

# --- 9. DYNAMIC HARDWARE FILE BINDING LAYER ---
path_jig = os.path.join(TEMP_DIR, "jig.stl")

link_b64s = []
for i in range(7):
    mesh_filename = f"link_{i}.stl" if i > 0 else "base_link.stl"
    target_mesh_path = os.path.join(BASE_DIR, "assets", "robots", selected_profile, mesh_filename)
    if not os.path.exists(target_mesh_path):
        target_mesh_path = os.path.join(BASE_DIR, "assets", "meshes", mesh_filename)
    link_b64s.append(get_file_base64_cached(target_mesh_path))

tooling_b64 = ""
if selected_tool_path and os.path.exists(selected_tool_path):
    tooling_b64 = get_file_base64_cached(selected_tool_path, get_file_hash(selected_tool_path))

scene_payload = {
    "profileName": selected_profile,
    "trajectory": st.session_state.program,
    "initialAngles": st.session_state.j_angles,
    "linkGeometries": link_b64s,
    "fallbackHeights": active_cfg["fallback_heights"],
    "dhConfig": active_cfg["links"],
    "jointLimits": active_cfg["limits"],
    "gunData": tooling_b64,
    "jigData": get_file_base64_cached(path_jig, get_file_hash(path_jig)),
    "toolOffsetX": t_off_x,
    "toolOffsetY": t_off_y,
    "toolOffsetZ": t_off_z,
    "toolRotX": float(t_rot_x) * (np.pi / 180.0),
    "toolRotY": float(t_rot_y) * (np.pi / 180.0),
    "toolRotZ": float(t_rot_z) * (np.pi / 180.0),
    "jigX": jx_pos,
    "jigY": jy_pos,
    "jigZ": jz_pos,
    "rotX": float(j_rot_x) * (np.pi / 180.0),
    "rotY": float(j_rot_y) * (np.pi / 180.0),
    "jigScale": js_scale
}

build_embedded_viewport(scene_payload)
