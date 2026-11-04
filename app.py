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
st.set_page_config(page_title="Industrial Robot Cloud Simulator", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")
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

# --- 3. SELECTION INITIALIZATION LAYER (main page; click the header to minimise) ---
robot_folder_path = os.path.join(BASE_DIR, "assets", "robots")
available_profiles = list(ROBOT_REGISTRY.keys())
if os.path.exists(robot_folder_path):
    scanned_folders = [f for f in os.listdir(robot_folder_path) if os.path.isdir(os.path.join(robot_folder_path, f))]
    if scanned_folders:
        available_profiles = sorted(list(set(scanned_folders + available_profiles)))
first_profile_idx = available_profiles.index("ABB_6700") if "ABB_6700" in available_profiles else 0

CATEGORY_MAPPING = {"Nil": None, "Welding Guns": "welding_guns", "Grippers": "grippers", "Welding Torches": "welding_torches"}
DEFAULT_BASES = [(0.0, 0.0, 0.0, 0.0), (3.2, 0.0, 0.0, np.pi), (1.6, -2.4, 0.0, np.pi / 2), (1.6, 2.4, 0.0, -np.pi / 2)]

setup_box = st.expander("⚙️ Simulation Setup  (click here to minimise after setting)", expanded=True)
with setup_box:
    top_a, top_b = st.columns([1, 2])
    with top_a:
        st.write("**🏭 Station**")
        n_robots = int(st.number_input("Number of robots (max 4)", min_value=1, max_value=4, value=1, step=1, key="n_robots"))
        if st.button("🔴 RESET TOOLS & JIG", use_container_width=True):
            for f in [os.path.join(TEMP_DIR, "jig.stl")] + [os.path.join(TEMP_DIR, f"gun_{k}.stl") for k in range(4)]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            st.session_state.program = []
            st.session_state.j_angles = [0.0] * 8
            st.cache_data.clear()
            st.cache_resource.clear()
            st.query_params.clear()
            st.rerun()
    with top_b:
        st.write("**🏗️ Rotary Positioning Jig (shared by all robots)**")
        jc1, jc2 = st.columns([2, 1])
        with jc1:
            up_jig = st.file_uploader("Upload Jig STL", type=["stl"], key="jig_up")
        with jc2:
            js_scale = st.number_input("Jig Geometry Scale", value=0.001, format="%.5f")
        if up_jig:
            with open(os.path.join(TEMP_DIR, "jig.stl"), "wb") as f:
                f.write(up_jig.getbuffer())
            st.cache_data.clear()
    st.write("**🧱 Fixtures, fences & other objects (optional)**")
    up_objs = st.file_uploader("Upload fixtures, fences, tables or any other STL files (up to 8)", type=["stl"], accept_multiple_files=True, key="obj_up")
    st.caption("Tip: click a robot, its tool or the jig in the 3D view to move it. Use the Layout tab for exact numbers and the Station tab to build the sequence.")

    robot_tabs = st.tabs([f"🤖 Robot {k + 1}" for k in range(n_robots)])
    robot_cfgs = []
    for k, rtab in enumerate(robot_tabs):
        with rtab:
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                profile_k = st.selectbox(f"Robot {k + 1} hardware profile", options=available_profiles,
                                         index=(first_profile_idx + k) % len(available_profiles), key=f"robot_profile_{k}")
            tool_path_k = None
            with rc2:
                tool_source = st.radio("Tooling Model Source", options=["Internal Library", "External STL Upload"], horizontal=True, key=f"tool_src_{k}")
                if tool_source == "Internal Library":
                    cat = st.selectbox("Select Tool Category Type", options=list(CATEGORY_MAPPING.keys()), key=f"tool_cat_{k}")
                    if cat != "Nil":
                        scan_path = os.path.join(BASE_DIR, "assets", "robot_tools", CATEGORY_MAPPING[cat])
                        found = [f for f in os.listdir(scan_path) if f.lower().endswith('.stl')] if os.path.exists(scan_path) else []
                        if found:
                            chosen = st.selectbox("Select Tooling Model", options=found, key=f"tool_model_{k}")
                            tool_path_k = os.path.join(scan_path, chosen)
                        else:
                            st.caption("⚠️ No internal library templates found. Please upload an external file.")
                else:
                    up_gun = st.file_uploader("Upload Custom External Tool STL", type=["stl"], key=f"gun_up_{k}")
                    if up_gun:
                        tool_path_k = os.path.join(TEMP_DIR, f"gun_{k}.stl")
                        with open(tool_path_k, "wb") as f:
                            f.write(up_gun.getbuffer())
                        st.cache_data.clear()
            robot_cfgs.append({"profile": profile_k, "tool_path": tool_path_k})
selected_profile = robot_cfgs[0]["profile"]

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

# --- 6. (controls now live in the Simulation Setup panel above) ---
jx_pos, jy_pos, jz_pos = 1.6, 0.0, 0.55
j_rot_x = j_rot_y = 0

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
        .tab { flex: 1; text-align: center; padding: 10px 2px; font-size: 12px; color: var(--mut); cursor: pointer; border-bottom: 2px solid transparent; font-weight: 600; }
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
        #btn-seq-run { background: var(--ok); color: #fff; }
        #btn-seq-run.stop { background: #c62828; }
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
        #parttools { position: absolute; top: 52px; left: 10px; z-index: 10; display: none; background: rgba(20,22,28,.9); border: 1px solid var(--acc); color: #fff; padding: 5px 10px; border-radius: 8px; font-size: 12px; }
        #parttools .ptrow { display: flex; gap: 4px; margin-top: 6px; }
        #parttools .mode-btn { padding: 6px 8px; white-space: nowrap; }
        #tool-only { display: none; }
        .node.sel { background: rgba(255,152,0,.18); outline: 1px solid var(--acc); }
        .node.pick { cursor: pointer; }
        .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
        .grid3 label { display: block; color: var(--mut); font-size: 10px; }
        .grid3 input { width: 100%; background: var(--panel2); color: var(--cy); border: 1px solid var(--line); border-radius: 5px; padding: 4px 5px; font-family: Consolas, monospace; font-size: 12px; }
        .rchip { display: inline-flex; align-items: center; gap: 4px; margin-right: 8px; font-size: 11px; color: var(--txt); }
        .rbadge { display: flex; align-items: center; gap: 6px; padding: 5px 6px; border-radius: 6px; font-size: 12px; }
        .rbadge.act { background: rgba(255,152,0,.15); }
        .node.act .nm { color: var(--acc); font-weight: 700; }
        tr.seqcur td { background: rgba(76,175,80,.22); }
        #ctxmenu { position: absolute; z-index: 40; display: none; background: rgba(20,22,28,.98); border: 1px solid var(--acc); border-radius: 8px; padding: 4px; min-width: 210px; box-shadow: 0 8px 24px rgba(0,0,0,.6); }
        #ctxmenu .chead { padding: 6px 10px; font-size: 11px; color: var(--acc); font-weight: 700; border-bottom: 1px solid var(--line); margin-bottom: 3px; }
        #ctxmenu .csep { height: 1px; background: var(--line); margin: 4px 2px; }
        #ctxmenu .citem { display: flex; align-items: center; gap: 8px; width: 100%; text-align: left; background: transparent; border: 0; color: var(--txt); padding: 7px 10px; border-radius: 5px; cursor: pointer; font-size: 12px; }
        #ctxmenu .citem:hover { background: var(--panel2); }
        #ctxmenu .citem.on { background: rgba(255,152,0,.2); color: var(--acc); }
        #ctxmenu .citem i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
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
        <button class="tbtn" id="btn-setup" title="Hide or show the Simulation Setup panel above">⚙ Hide setup</button>
        <button class="tbtn" id="btn-export" title="Download the recorded program as a text file">⬇ Export program</button>
        <button class="tbtn" id="btn-help" title="Keyboard shortcuts">? Help</button>
    </div>

    <div id="main">
        <div id="tree">
            <div class="h">Station</div>
            <div id="robot-nodes"></div>
            <div class="node pick" data-part="tool"><span>🔧</span><span class="nm">Tool of active robot</span><span class="eye" data-vis="tool">👁</span></div>
            <div class="node pick" data-part="jig"><span>⚙</span><span class="nm">Jig / positioner</span><span class="eye" data-vis="jig">👁</span></div>
            <div id="obj-nodes"></div>
            <div class="node"><span>▦</span><span class="nm">Floor grid</span><span class="eye" data-vis="grid">👁</span></div>
            <div class="h">Active robot</div>
            <div class="infobox">
                <div><span>Model</span><b id="inf-model">-</b></div>
                <div><span>Axes</span>6 + E1 positioner</div>
                <div><span>Reach</span>~<b id="inf-reach">-</b> m</div>
            </div>
            <div class="h">Program</div>
            <div class="node" id="node-prog"><span>▶</span><span class="nm">Active robot program</span><span id="inf-steps" class="chip">0</span></div>
            <div class="note">Educational models. Link shapes come from your CAD files; use for learning, not for cell sign-off.</div>
        </div>

        <div id="view">
            <div id="canvas-container"></div>
            <div id="viewbtns">
                <button class="tbtn" data-view="iso">Iso</button><button class="tbtn" data-view="front">Front</button><button class="tbtn" data-view="side">Side</button><button class="tbtn" data-view="top">Top</button><button class="tbtn" data-view="fit">Fit</button>
            </div>
            <div id="parttools"><div id="pt-name"></div><div class="ptrow"><button class="mode-btn pt-move">⤢ Move (W)</button><button class="mode-btn pt-rot">⟳ Rotate (E)</button><button class="mode-btn pt-space">Space: World (Q)</button></div><div class="ptrow" id="tool-only"><button class="mode-btn" id="pt-snap">Snap 90°: ON</button><button class="mode-btn rot90" data-ax="x">+90° X</button><button class="mode-btn rot90" data-ax="y">+90° Y</button><button class="mode-btn rot90" data-ax="z">+90° Z</button><button class="mode-btn" id="pt-round">Round to 90°</button></div></div>
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
            <div id="tabs"><div class="tab active" data-tab="jog">Jog</div><div class="tab" data-tab="prog">Program</div><div class="tab" data-tab="layout">Layout</div><div class="tab" data-tab="station">Station</div><div class="tab" data-tab="learn">Learn</div></div>

            <div class="pane active" id="pane-jog">
              <div id="jog-pendant">
                <div class="row" id="robot-sel-row"><label>Controlling</label><select id="sel-robot" style="flex:1"></select></div>
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


            <div class="pane" id="pane-layout">
                <div class="note" style="margin:0 0 8px">Click the <b>Tool</b> or the <b>Jig</b> in the 3D view (or in the Station list) to select it. Coloured arrows show its X (red), Y (green) and Z (blue) axes. Drag an arrow to move it, or switch to Rotate and drag a ring.</div>
                <div class="mode-container">
                    <button class="mode-btn pt-move">Move (W)</button>
                    <button class="mode-btn pt-rot">Rotate (E)</button>
                    <button class="mode-btn pt-space">Space: World (Q)</button>
                </div>
                <div class="row"><button class="tbtn" data-sel="robot">🦾 Select robot</button><button class="tbtn" data-sel="tool">🔧 Select tool</button><button class="tbtn" data-sel="jig">⚙ Select jig</button><button class="tbtn" data-sel="">Deselect</button></div>
                <div class="h" style="margin-left:0">Robot base position (active robot)</div>
                <div class="grid3" id="g-base"></div>
                <div class="h" style="margin-left:0">Tool relative to flange A6</div>
                <div class="grid3" id="g-tool"></div>
                <div class="row"><button class="tbtn" id="btn-snap">🧲 Snap tool to flange</button></div>
                <div id="obj-section" style="display:none"><div class="h" style="margin-left:0">Uploaded fixture / fence / object</div>
                <div class="note" style="margin:0 0 6px" id="obj-pick"></div>
                <div class="grid3" id="g-obj"></div>
                <div class="row"><button class="tbtn" id="btn-okeep">Use CAD origin / centre</button></div></div>
                <div class="h" style="margin-left:0">Jig / positioner in the cell</div>
                <div class="grid3" id="g-jig"></div>
                <div class="row"><button class="tbtn" id="btn-cadx">CAD turn 90° X</button><button class="tbtn" id="btn-cady">CAD turn 90° Y</button></div>
                <div class="row"><button class="tbtn" id="btn-lay-reset">↺ Reset layout</button></div>
                <div class="note" style="margin-left:0">Your layout is remembered in this browser for each robot.</div>
            </div>


            <div class="pane" id="pane-station">
                <div class="h" style="margin-top:0">Robots in this station</div>
                <div id="st-robots"></div>
                <div class="note" style="margin-left:0">Number of robots and their tools are chosen in Simulation Setup above (max 4). Record each robot's program in the Jog tab, then build the station sequence below.</div>
                <div class="h">Station sequence</div>
                <div class="row"><select id="seq-kind" style="flex:1"><option value="run">Run robot program(s)</option><option value="home">Send robot(s) home (0°)</option><option value="e1">Turn positioner E1</option><option value="wait">Wait</option></select><button class="tbtn" id="seq-add">+ Add</button></div>
                <div class="row"><button class="tbtn" id="seq-auto" title="Run R1, then R2, then R3, then R4 one after another">⚡ Auto: one after another</button><button class="tbtn" id="seq-clear">Clear</button></div>
                <table><tbody id="seq-body"></tbody></table>
                <div class="note" id="seq-empty" style="margin-left:0">No sequence yet. Tick two or more robots in one "Run" step to make them work at the same time.</div>
                <div class="action-block">
                    <button class="btn-action" id="btn-seq-run">▶️ RUN STATION SEQUENCE</button>
                    <div class="row" style="margin:0"><label><input type="checkbox" id="chk-sloop"> Loop</label><div style="flex:1"></div><span class="step-counter" id="seq-status">Idle</span></div>
                </div>
            </div>

            <div class="pane learn" id="pane-learn">
                <h4>Joint jog vs TCP jog</h4><p>Joint mode moves one axis (A1 to A6) at a time. TCP mode lets you drag the tool tip in space; the computer finds the joint angles for you (inverse kinematics).</p>
                <h4>What is the TCP?</h4><p>The Tool Centre Point is the working point of the tool, for example the tip of the welding electrode. Robot programs store where the TCP must go.</p>
                <h4>Joint limits</h4><p>Every axis has a travel range. The red value and the warning chip show when an axis is close to its end stop.</p>
                <h4>Singularity</h4><p>When axis A5 is near 0°, axes A4 and A6 line up and the wrist loses one direction of motion. Keep A5 away from 0° in welding moves.</p>
                <h4>Recording a program</h4><p>Move the robot, press Record, repeat. Run Simulation then blends smoothly from one recorded pose to the next. Use the speed slider to slow it down.</p>
                <h4>Gesture mode</h4><p>Hold your palm in the box for 2 seconds, move your hand to move the tool, roll the hand to turn A6, and pinch to record.</p>
                <h4>Station sequences</h4><p><b>Sequential</b>: one robot at a time, safest when work areas overlap. <b>Parallel</b>: tick several robots in one step so they work together, for example two guns welding opposite sides. <b>Positioner</b>: turn E1 between robot steps so every robot reaches its spots. Check cycle time by watching the longest robot in each step.</p><h4>Shortcuts</h4><p>R record · Space run or stop · 1 to 6 select axis, then ← → to jog · I T F S fit views.</p>
            </div>
        </div>
    </div>

    <div id="statusbar"><span id="sb-tcp">TCP -</span><span id="sb-joints">Joints -</span><span id="sb-steps">Steps 0</span><span id="sb-speed">Speed 50%</span></div>


    <script>
        const data = JSON.parse(JSON.stringify(__PAYLOAD_STREAM__));
        /* Multi-robot station: every robot has its own state; the globals below always point at the ACTIVE robot,
           so the original kinematics / IK / gesture code works unchanged for whichever robot is selected. */
        let dh = null, limitsConfig = null;
        let localJointAngles = new Array(8).fill(0);
        let lastComputedTransforms = [];
        let embeddedTrajectory = [];
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
        let links = [];
        let gunMesh = null, toolAdjustmentGroup = null, tcpAnchorPivot = null;
        let jigMesh = new THREE.Group();
        let internalJigContent = new THREE.Group();
        jigMesh.add(internalJigContent);
        scene.add(jigMesh);

        const transformGizmo = new THREE.TransformControls(camera, renderer.domElement);
        transformGizmo.size = 0.65;
        transformGizmo.setMode("translate");
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

        /* ---------- one robot = one root group at its base pose; links, tool and TCP pivot are children ---------- */
        const stationRobots = [];
        let activeR = null;
        function labelSprite(text, color) {
            const c = document.createElement('canvas'); c.width = 256; c.height = 64; const x = c.getContext('2d');
            x.fillStyle = 'rgba(10,12,16,.8)'; x.fillRect(0, 0, 256, 64); x.strokeStyle = color; x.lineWidth = 3; x.strokeRect(2, 2, 252, 60);
            x.font = 'bold 30px Segoe UI, Arial'; x.fillStyle = color; x.textAlign = 'center'; x.textBaseline = 'middle'; x.fillText(text, 128, 34);
            const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }));
            s.scale.set(0.9, 0.225, 1); s.renderOrder = 997; return s;
        }
        function createRobot(idx, rc) {
            const prof = data.profiles[rc.profile];
            const root = new THREE.Group();
            root.position.set(rc.base.x, rc.base.y, rc.base.z || 0); root.rotation.z = rc.base.yaw || 0;
            scene.add(root);

            let targetColor = 0xcccccc;
            if (rc.profile === "Yaskawa_3500") { targetColor = 0x0055ff; }
            else if (rc.profile === "KUKA_KR150") { targetColor = 0xff6600; }

            const rlinks = [];
            for (let i = 0; i < 7; i++) {
                let mesh;
                let currentLinkColor = (i === 0) ? 0x222222 : targetColor;
                if (prof.linkGeometries && prof.linkGeometries[i] && prof.linkGeometries[i].length > 0) {
                    const geometry = loader.parse(base64ToArrayBuffer(prof.linkGeometries[i]));
                    mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: currentLinkColor, roughness: 0.4 }));
                } else {
                    const h = prof.fallbackHeights[i];
                    const geometry = new THREE.CylinderGeometry(0.18, 0.22, h, 24);
                    geometry.rotateX(Math.PI / 2);
                    if (i === 1 || i === 2 || i === 3) { geometry.translate(0, 0, h / 2); }
                    mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: currentLinkColor, roughness: 0.4 }));
                }
                root.add(mesh); rlinks.push(mesh);
            }
            const gm = new THREE.Group(); root.add(gm);
            const tg = new THREE.Group(); gm.add(tg);
            let hasTool = false;
            const tb = rc.toolKey ? data.tools[rc.toolKey] : "";
            if (tb && tb.length > 0) {
                const geometry = loader.parse(base64ToArrayBuffer(tb));
                geometry.center();
                const m = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 }));
                m.scale.set(0.001, 0.001, 0.001);
                tg.add(m); hasTool = true;
            }
            const pivot = new THREE.Object3D(); root.add(pivot);
            const sprite = labelSprite('R' + (idx + 1) + '  ' + rc.profile.replace('_', ' '), '#ffb74d'); sprite.position.set(0, 0, 2.9); root.add(sprite);
            return { idx: idx, label: 'R' + (idx + 1), profile: rc.profile, root: root, links: rlinks, gunMesh: gm, tg: tg, pivot: pivot, dh: prof.dhConfig, limits: prof.jointLimits,
                     angles: new Array(8).fill(0), traj: [], tool: { ox: 0, oy: 0, oz: 0, rx: 0, ry: 0, rz: 0 }, hasTool: hasTool, toolName: rc.toolName || '', sprite: sprite };
        }
        function bindRobot(R) {
            activeR = R; dh = R.dh; limitsConfig = R.limits; localJointAngles = R.angles; embeddedTrajectory = R.traj;
            links = R.links; gunMesh = R.gunMesh; toolAdjustmentGroup = R.tg; tcpAnchorPivot = R.pivot; data.profileName = R.profile;
            data.toolOffsetX = R.tool.ox; data.toolOffsetY = R.tool.oy; data.toolOffsetZ = R.tool.oz; data.toolRotX = R.tool.rx; data.toolRotY = R.tool.ry; data.toolRotZ = R.tool.rz;
        }
        function commitRobot(R) {
            if (!R) return;
            R.angles = localJointAngles; R.traj = embeddedTrajectory;
            R.tool = { ox: data.toolOffsetX, oy: data.toolOffsetY, oz: data.toolOffsetZ, rx: data.toolRotX, ry: data.toolRotY, rz: data.toolRotZ };
        }
        function withRobot(R, fn) {
            if (R === activeR) return fn();
            const prev = activeR; commitRobot(prev); bindRobot(R);
            try { return fn(); } finally { commitRobot(R); bindRobot(prev); }
        }
        data.robots.forEach((rc, i) => stationRobots.push(createRobot(i, rc)));
        bindRobot(stationRobots[0]);
        transformGizmo.attach(tcpAnchorPivot);

        if(data.jigData && data.jigData.length > 0) {
            const geometry = loader.parse(base64ToArrayBuffer(data.jigData));
            geometry.center(); 
            const m = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.6 }));
            m.scale.set(data.jigScale, data.jigScale, data.jigScale);
            m.rotation.x = data.rotX;
            m.rotation.y = data.rotY;
            internalJigContent.add(m);
        }

        /* ---------- uploaded fixtures, fences and other objects (each one can be selected, moved and rotated) ---------- */
        const stationObjs = [];
        (data.objects || []).forEach((oc, k) => {
            try {
                const geometry = loader.parse(base64ToArrayBuffer(oc.data));
                geometry.computeBoundingBox();
                const bb = geometry.boundingBox, size = bb.getSize(new THREE.Vector3()), cen = bb.getCenter(new THREE.Vector3());
                const sc = Math.max(size.x, size.y, size.z) > 20 ? 0.001 : 1;        // millimetres or metres, detected from the size
                geometry.translate(-cen.x, -cen.y, -cen.z);
                const n = oc.name.toLowerCase(), fence = /fence|guard|cage|barrier|safety|screen/.test(n);
                const mat = new THREE.MeshStandardMaterial({ color: fence ? 0xffc107 : 0x90a4ae, roughness: 0.5, metalness: 0.3, transparent: fence, opacity: fence ? 0.45 : 1, side: fence ? THREE.DoubleSide : THREE.FrontSide });
                const mesh = new THREE.Mesh(geometry, mat); mesh.scale.setScalar(sc);
                const root = new THREE.Group(); root.add(mesh);
                const def = [-1.5 + (k % 4) * 1.4, -3.2 - Math.floor(k / 4) * 1.4, size.z * sc / 2];
                root.position.set(def[0], def[1], def[2]); scene.add(root);
                stationObjs.push({ idx: k, name: oc.name, root: root, mesh: mesh, sc: sc, cen: cen.clone(), keep: false, def: def });
            } catch (e) { console.log('object failed: ' + oc.name); }
        });
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
                let physicalWristPos = links[6].position.clone();
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
                                robotCalibrationBaseline = links[6].position.clone();
                                tcpAnchorPivot.position.copy(robotCalibrationBaseline);
                            }
                        } else {
                            document.getElementById("ar-hud-text").innerHTML = "⏳ INITIALIZING: HOLD STILL (" + remainingTime + "s)";
                            document.getElementById("ar-hud-text").style.color = "#ffeb3b";
                            
                            if (links[6]) {
                                let currentRobotPos = links[6].position.clone();
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
                            tcpAnchorPivot.position.copy(links[6].position.clone());
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

        /* ======================= UI LAYER (added; the kinematics above is the original) ======================= */
        const $ = (id) => document.getElementById(id);
        const D2R = Math.PI / 180, R2D = 180 / Math.PI;
        const REACH = { ABB_6700: 2.65, ABB_4400: 1.95, ABB_6600: 2.55, KUKA_KR150: 3.1, Yaskawa_3500: 2.45 };
        let e1Global = 0;                    // the positioner (E1) is shared by the whole station
        let dirty = false;
        function markDirty() { dirty = true; }
        function syncProgramToParent() { markDirty(); }     // programs are now kept in the browser (no page reload, no URL hack)
        const busy = () => runSimulation || SR.on;
        const rName = (R) => R.label + ' ' + R.profile.replace('_', ' ');

        function toast(msg) { const t = $('toast'); t.innerText = msg; t.style.display = 'block'; clearTimeout(toast.h); toast.h = setTimeout(() => t.style.display = 'none', 3500); }

        // ---------- tabs ----------
        document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
            document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + t.dataset.tab));
            if (t.dataset.tab === 'prog') renderProgram();
            if (t.dataset.tab === 'station') { renderStationRobots(); renderSeq(); }
        }));

        // ---------- camera views (centred on the whole station) ----------
        function stationCentre() {
            let x = 0, y = 0; stationRobots.forEach(R => { x += R.root.position.x; y += R.root.position.y; }); x /= stationRobots.length; y /= stationRobots.length;
            x = (x * stationRobots.length + data.jigX) / (stationRobots.length + 1); y = (y * stationRobots.length + data.jigY) / (stationRobots.length + 1);
            let r = 1.5; stationRobots.forEach(R => { r = Math.max(r, Math.hypot(R.root.position.x - x, R.root.position.y - y)); });
            return { x: x, y: y, r: r };
        }
        function setView(name) {
            const c = stationCentre(), t = controls.target, d = 4.2 + c.r * 1.6;
            t.set(c.x, c.y, 0.8);
            if (name === 'iso' || name === 'fit') camera.position.set(c.x + d * 0.75, c.y - d * 0.75, 0.8 + d * 0.55);
            if (name === 'front') camera.position.set(c.x, c.y - d * 1.2, 1.0);
            if (name === 'side') camera.position.set(c.x + d * 1.2, c.y, 1.0);
            if (name === 'top') { t.set(c.x, c.y, 0); camera.position.set(c.x, c.y + 0.001, d * 1.4); }
            camera.up.set(0, 0, 1); controls.update();
        }
        document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));

        // ---------- joint rows ----------
        function setJointAbs(i, ang) {
            if (busy()) return;
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

        // ---------- robots: tree, selector, activation ----------
        function updateChips() {
            $('chip-robot').innerText = rName(activeR); $('inf-model').innerText = rName(activeR);
            $('inf-reach').innerText = (REACH[activeR.profile] || 2.5).toFixed(2);
        }
        function updateTreeClasses() {
            document.querySelectorAll('#robot-nodes .node').forEach(n => { const on = +n.dataset.robot === activeR.idx; n.classList.toggle('act', on); n.classList.toggle('sel', on && selPart === 'robot'); });
            document.querySelectorAll('.node.pick[data-part]').forEach(n => n.classList.toggle('sel', n.dataset.part === selPart));
            document.querySelectorAll('#obj-nodes .node').forEach(n => n.classList.toggle('sel', selPart === 'obj' && +n.dataset.obj === selObj));
            stationRobots.forEach(R => R.sprite.material.opacity = (R === activeR) ? 1 : 0.55);
        }
        function buildTree() {
            const box = $('robot-nodes'); box.innerHTML = '';
            const sel = $('sel-robot'); sel.innerHTML = '';
            stationRobots.forEach(R => {
                const d = document.createElement('div'); d.className = 'node pick'; d.dataset.robot = R.idx;
                d.innerHTML = '<span>🦾</span><span class="nm">' + rName(R) + '</span><span class="eye">👁</span>';
                box.appendChild(d);
                d.addEventListener('click', (e) => {
                    if (e.target.classList.contains('eye')) { R.root.visible = !R.root.visible; e.target.classList.toggle('off', !R.root.visible); return; }
                    activate(R); selectPart('robot');
                });
                const o = document.createElement('option'); o.value = R.idx; o.textContent = rName(R); sel.appendChild(o);
            });
            const ob = $('obj-nodes'); ob.innerHTML = '';
            stationObjs.forEach(o => {
                const d = document.createElement('div'); d.className = 'node pick'; d.dataset.obj = o.idx;
                d.innerHTML = '<span>🧱</span><span class="nm">' + o.name + '</span><span class="eye">👁</span>'; ob.appendChild(d);
                d.addEventListener('click', (e) => { if (e.target.classList.contains('eye')) { o.root.visible = !o.root.visible; e.target.classList.toggle('off', !o.root.visible); return; } selectPart('obj', o.idx); });
            });
            $('robot-sel-row').style.display = stationRobots.length > 1 ? 'flex' : 'none';
            sel.addEventListener('change', () => activate(stationRobots[+sel.value]));
        }
        function activate(R) {
            if (R === activeR) return;
            if (busy()) { toast('Stop the simulation first'); $('sel-robot').value = activeR.idx; return; }
            commitRobot(activeR); R.angles[7] = e1Global; bindRobot(R);
            transformGizmo.attach(R.pivot);
            buildJogRows(); refreshSceneDisplay(true); updateUIElements(); updateChips();
            $('sel-robot').value = R.idx;
            if (selPart === 'tool' && !R.hasTool) selectPart(null);
            syncProxy(); updateTreeClasses(); applyGizmoVisibility(); markDirty();
        }
        document.querySelectorAll('.eye[data-vis]').forEach(e => e.addEventListener('click', () => {
            const on = e.classList.toggle('off') === false, k = e.dataset.vis;
            if (k === 'tool') stationRobots.forEach(R => R.gunMesh.visible = on); else if (k === 'jig') jigMesh.visible = on; else if (k === 'grid') grid.visible = on;
        }));

        // ---------- live read-outs ----------
        let lastSig = '';
        function updateReadouts() {
            const a = localJointAngles;
            if (!SR.on && !runSimulation) e1Global = a[7];
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
                const out = r > (REACH[activeR.profile] || 2.5);
                const pr = $('pill-reach'); pr.className = 'pill' + (out ? ' bad' : ''); pr.innerText = out ? 'Beyond reach' : 'Within reach';
                const sig = [p.x, p.y, p.z, a.join()].join('|');
                if (sig !== lastSig) {
                    lastSig = sig; if (!busy()) markDirty();
                    $('sb-tcp').innerHTML = activeR.label + ' TCP <b>X ' + (p.x * 1000).toFixed(0) + ' Y ' + (p.y * 1000).toFixed(0) + ' Z ' + (p.z * 1000).toFixed(0) + '</b> mm';
                    $('sb-joints').innerHTML = 'Joints <b>' + a.slice(1, 7).map(v => (v * R2D).toFixed(0)).join(' ') + '</b> · E1 <b>' + (a[7] * R2D).toFixed(0) + '</b>';
                }
            }
            $('sb-steps').innerText = 'Steps ' + embeddedTrajectory.length;
            $('sb-speed').innerText = 'Speed ' + $('sld-speed').value + '%';
            $('chip-mode').innerText = activeJogMode === 'joint' ? 'Joint' : activeJogMode === 'tcp' ? 'TCP' : 'Gesture';
        }

        // ---------- program table (active robot) ----------
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
            if ($('pane-station').classList.contains('active')) renderStationRobots();
        }
        $('prog-body').addEventListener('click', (e) => {
            const g = e.target.getAttribute('data-go'), d = e.target.getAttribute('data-del');
            if (g !== null && !busy()) { localJointAngles = [...embeddedTrajectory[+g].angles]; activeR.angles = localJointAngles; refreshSceneDisplay(true); toast('Moved to step ' + (+g + 1)); }
            if (d !== null && !busy()) { embeddedTrajectory.splice(+d, 1); updateUIElements(); markDirty(); }
        });
        const _origUpdateUI = updateUIElements;
        updateUIElements = function () { _origUpdateUI(); renderProgram(); };

        $('btn-export').addEventListener('click', () => {
            let t = '! Robot programs exported from Industrial Robot Cloud Simulator\n! Joint targets in degrees (A1..A6), E1 positioner\n';
            let any = false;
            commitRobot(activeR);
            stationRobots.forEach(R => {
                if (!R.traj.length) return; any = true;
                t += '\n! ===== ' + rName(R) + ' =====\n';
                R.traj.forEach((s, i) => { t += 'MoveAbsJ  [' + s.angles.slice(1, 7).map(v => (v * R2D).toFixed(2)).join(', ') + '], E1=' + (s.angles[7] * R2D).toFixed(2) + ', speed=' + $('sld-speed').value + '%;   ! ' + R.label + ' step ' + (i + 1) + '\n'; });
            });
            if (!any) { toast('Nothing to export yet. Record some steps first.'); return; }
            const url = URL.createObjectURL(new Blob([t], { type: 'text/plain' }));
            const a = document.createElement('a'); a.href = url; a.download = 'station_program.txt'; a.click(); URL.revokeObjectURL(url);
        });
        $('btn-help').addEventListener('click', () => { document.querySelector('.tab[data-tab="learn"]').click(); });

        // ---------- run / stop (active robot) ----------
        function toggleRun() {
            if (SR.on) return;
            if (runSimulation) { runSimulation = false; simStepIndex = 0; interpolationFraction = 0; return; }
            if (embeddedTrajectory.length < 2) { toast('Record at least 2 steps for this robot first.'); return; }
            simStepIndex = 0; interpolationFraction = 0; runSimulation = true;
        }
        $('btn-run-sim').addEventListener('click', toggleRun);
        function updateRunUI() {
            document.body.classList.toggle('running', busy());
            const b = $('btn-run-sim'); b.classList.toggle('stop', runSimulation);
            b.innerText = runSimulation ? '⏹ STOP SIMULATION' : '▶️ RUN SIMULATION';
            const s = $('btn-seq-run'); s.classList.toggle('stop', SR.on); s.innerText = SR.on ? '⏹ STOP STATION' : '▶️ RUN STATION SEQUENCE';
            s.style.pointerEvents = 'auto'; s.style.opacity = 1;
        }

        /* ======================= PART EDITING: robot base, tool, jig ======================= */
        const partProxy = new THREE.Object3D(); scene.add(partProxy);
        const partAxes = new THREE.AxesHelper(0.45); partAxes.material.depthTest = false; partAxes.renderOrder = 998; partProxy.add(partAxes); partAxes.visible = false;
        const partGizmo = new THREE.TransformControls(camera, renderer.domElement);
        if (partGizmo.setSize) partGizmo.setSize(0.8); else partGizmo.size = 0.8;
        partGizmo.attach(partProxy);
        scene.add(partGizmo.isObject3D ? partGizmo : partGizmo.getHelper());
        partGizmo.visible = false; partGizmo.enabled = false;
        let selObj = 0, selPart = null, partDragging = false, gMode = 'translate', axisSel = 'all';

        function tintPart(obj, on) { obj.traverse(o => { if (o.isMesh && o.material && o.material.emissive) o.material.emissive.setHex(on ? 0x6a3a00 : 0x000000); }); }
        function selectPart(name, idx) {
            if (busy()) return;
            if (name === '') name = null;
            if (name === 'tool' && !activeR.hasTool) { toast('This robot has no tool. Choose one in Simulation Setup above.'); name = null; }
            if (name === 'jig' && !(data.jigData && data.jigData.length)) { toast('No jig loaded. Upload a jig STL in Simulation Setup above.'); name = null; }
            if (name === 'obj' && !stationObjs[idx]) name = null;
            stationRobots.forEach(R => tintPart(R.root, false)); tintPart(jigMesh, false); stationObjs.forEach(o => tintPart(o.root, false));
            selPart = name; axisSel = 'all'; if (name === 'obj') selObj = idx;
            if (name === 'tool') tintPart(activeR.gunMesh, true); else if (name === 'robot') tintPart(activeR.root, true); else if (name === 'jig') tintPart(jigMesh, true); else if (name === 'obj') tintPart(stationObjs[idx].root, true);
            syncProxy();
            partGizmo.visible = !!name; partGizmo.enabled = !!name; partAxes.visible = !!name;
            $('parttools').style.display = name ? 'block' : 'none';
            $('pt-name').innerText = name === 'tool' ? '🔧 ' + activeR.label + ' tool (moves with the flange)' : name === 'robot' ? '🦾 ' + rName(activeR) + ' base: move it anywhere, the tool follows' : name === 'jig' ? '⚙ Jig selected' : name === 'obj' ? '🧱 ' + stationObjs[idx].name : '';
            applySnap(); updateTreeClasses(); applyGizmoVisibility();
        }
        function applyGizmoVisibility() {
            const wantTcp = (activeJogMode !== 'joint') && !selPart;
            transformGizmo.visible = wantTcp; transformGizmo.enabled = wantTcp;
        }
        setInterval(applyGizmoVisibility, 150);

        function syncProxy() {
            if (!selPart || partDragging) return;
            if (selPart === 'tool') { toolAdjustmentGroup.updateWorldMatrix(true, false); toolAdjustmentGroup.getWorldPosition(partProxy.position); toolAdjustmentGroup.getWorldQuaternion(partProxy.quaternion); }
            else if (selPart === 'robot') { partProxy.position.copy(activeR.root.position); partProxy.quaternion.copy(activeR.root.quaternion); }
            else if (selPart === 'obj') { const o = stationObjs[selObj]; partProxy.position.copy(o.root.position); partProxy.quaternion.copy(o.root.quaternion); }
            else { partProxy.position.copy(jigMesh.position); partProxy.quaternion.copy(jigMesh.quaternion); }
        }
        (function loop() { requestAnimationFrame(loop); syncProxy(); })();

        function applyProxyToPart() {
            if (selPart === 'tool' && links[6]) {
                links[6].updateWorldMatrix(true, false);
                const p6 = new THREE.Vector3(), q6 = new THREE.Quaternion(); links[6].getWorldPosition(p6); links[6].getWorldQuaternion(q6);
                const inv = q6.clone().invert();
                const off = partProxy.position.clone().sub(p6).applyQuaternion(inv);
                data.toolOffsetX = off.x; data.toolOffsetY = off.y; data.toolOffsetZ = off.z;
                const e = new THREE.Euler().setFromQuaternion(inv.multiply(partProxy.quaternion.clone()), 'XYZ');
                data.toolRotX = e.x; data.toolRotY = e.y; data.toolRotZ = e.z;
            } else if (selPart === 'robot') {
                activeR.root.position.copy(partProxy.position);
                activeR.root.rotation.set(0, 0, new THREE.Euler().setFromQuaternion(partProxy.quaternion, 'ZYX').z);   // robots only turn about the vertical axis
            } else if (selPart === 'obj') {
                const o = stationObjs[selObj]; o.root.position.copy(partProxy.position); o.root.quaternion.copy(partProxy.quaternion);
            } else if (selPart === 'jig') {
                data.jigX = partProxy.position.x; data.jigY = partProxy.position.y; data.jigZ = partProxy.position.z;
                jigMesh.quaternion.copy(partProxy.quaternion);
            }
            refreshSceneDisplay(false);
        }
        partGizmo.addEventListener('dragging-changed', (e) => { partDragging = e.value; controls.enabled = !e.value; if (!e.value) markDirty(); });
        partGizmo.addEventListener('objectChange', applyProxyToPart);

        function setGizmoMode(m, ax) {
            gMode = m; axisSel = ax || 'all'; partGizmo.setMode(m);
            document.querySelectorAll('.pt-move').forEach(b => b.classList.toggle('active', m === 'translate'));
            document.querySelectorAll('.pt-rot').forEach(b => b.classList.toggle('active', m === 'rotate'));
            applySnap();
        }
        let gSpace = 'world';
        function toggleSpace() { gSpace = gSpace === 'world' ? 'local' : 'world'; partGizmo.setSpace(gSpace); document.querySelectorAll('.pt-space').forEach(b => b.innerText = 'Space: ' + (gSpace === 'world' ? 'World' : 'Local') + ' (Q)'); }
        document.querySelectorAll('.pt-move').forEach(b => b.addEventListener('click', () => setGizmoMode('translate')));
        document.querySelectorAll('.pt-rot').forEach(b => b.addEventListener('click', () => setGizmoMode('rotate')));
        document.querySelectorAll('.pt-space').forEach(b => b.addEventListener('click', toggleSpace));
        document.querySelectorAll('[data-sel]').forEach(b => b.addEventListener('click', () => selectPart(b.dataset.sel)));
        document.querySelectorAll('.node.pick[data-part]').forEach(n => n.addEventListener('click', () => selectPart(n.dataset.part)));

        // ---------- snapping: tool = 90 degrees (match with flange A6), robot = 15 degrees ----------
        let snapOn = true;
        function applySnap() {
            const tool = selPart === 'tool' && snapOn, robot = selPart === 'robot';
            partGizmo.setRotationSnap(tool ? Math.PI / 2 : robot ? 15 * D2R : null);
            partGizmo.setTranslationSnap(tool ? 0.01 : robot ? 0.05 : null);
            const rotOnlyZ = selPart === 'robot' && gMode === 'rotate';   // robots only turn about the vertical axis
            partGizmo.showX = !rotOnlyZ && (axisSel === 'all' || axisSel === 'x'); partGizmo.showY = !rotOnlyZ && (axisSel === 'all' || axisSel === 'y'); partGizmo.showZ = rotOnlyZ || axisSel === 'all' || axisSel === 'z';
            $('pt-snap').innerText = 'Snap 90°: ' + (snapOn ? 'ON' : 'OFF'); $('pt-snap').classList.toggle('active', snapOn);
            $('tool-only').style.display = selPart === 'tool' ? 'flex' : 'none';
        }
        $('pt-snap').addEventListener('click', () => { snapOn = !snapOn; applySnap(); });
        function toolLocalQuat() { return new THREE.Quaternion().setFromEuler(new THREE.Euler(data.toolRotX, data.toolRotY, data.toolRotZ, 'XYZ')); }
        function setToolLocalQuat(q) { const e = new THREE.Euler().setFromQuaternion(q, 'XYZ'); data.toolRotX = e.x; data.toolRotY = e.y; data.toolRotZ = e.z; refreshSceneDisplay(false); markDirty(); }
        document.querySelectorAll('.rot90').forEach(b => b.addEventListener('click', () => {
            const ax = b.dataset.ax === 'x' ? new THREE.Vector3(1, 0, 0) : b.dataset.ax === 'y' ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
            setToolLocalQuat(new THREE.Quaternion().setFromAxisAngle(ax, Math.PI / 2).multiply(toolLocalQuat()));   // turned about the flange (A6) axes
        }));
        function snapQuat90(q) {   // nearest of the 24 axis-aligned orientations
            const e = new THREE.Matrix4().makeRotationFromQuaternion(q).elements;
            const col = (i) => new THREE.Vector3(e[i * 4], e[i * 4 + 1], e[i * 4 + 2]);
            const dom = (v, skip) => { let best = -1, bi = 0; ['x', 'y', 'z'].forEach((k, i) => { if (skip.includes(i)) return; if (Math.abs(v[k]) > best) { best = Math.abs(v[k]); bi = i; } }); const o = new THREE.Vector3(); o.setComponent(bi, Math.sign(v.getComponent(bi)) || 1); return [o, bi]; };
            const [a0, i0] = dom(col(0), []); const [a1] = dom(col(1), [i0]); const a2 = new THREE.Vector3().crossVectors(a0, a1);
            return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(a0, a1, a2));
        }
        function roundTool(msg) { setToolLocalQuat(snapQuat90(toolLocalQuat())); if (msg) toast('Tool rotation rounded to the nearest 90°'); }
        $('pt-round').addEventListener('click', () => roundTool(true));
        partGizmo.addEventListener('dragging-changed', (e) => { if (!e.value && selPart === 'tool' && snapOn) roundTool(false); });

        window.addEventListener('keydown', (e) => {
            if (e.target && (e.target.tagName === 'INPUT' && e.target.type !== 'range' || e.target.tagName === 'SELECT')) return;
            const k = e.key.toLowerCase();
            if (k === 'w') setGizmoMode('translate'); else if (k === 'e') setGizmoMode('rotate'); else if (k === 'q') toggleSpace(); else if (k === 'escape') selectPart(null);
            else if (k === 'r') { if (!busy()) { triggerManualStepRecord(); toast('Step recorded'); } }
            else if (k === ' ') { e.preventDefault(); if (SR.on) seqStop(); else toggleRun(); }
            else if (k >= '1' && k <= '6') { selAxis = +k; toast('Axis A' + k + ' selected: use ← → to jog'); }
            else if (k === 'arrowleft' && activeJogMode === 'joint') { jogJointUI(selAxis, -1); e.preventDefault(); }
            else if (k === 'arrowright' && activeJogMode === 'joint') { jogJointUI(selAxis, 1); e.preventDefault(); }
            else if (k === 'i') setView('iso'); else if (k === 'f') setView('front'); else if (k === 's') setView('side'); else if (k === 't') setView('top');
        });
        let selAxis = 1;

        // ---------- click in 3D to select (tool, robots and uploaded objects by ray; the jig by its bounding box) ----------
        function pickAt(e) {
            const r = renderer.domElement.getBoundingClientRect();
            const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
            const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, camera);
            const hit = (obj) => { if (!obj.visible) return false; const b = new THREE.Box3().setFromObject(obj); return !b.isEmpty() && ray.ray.intersectBox(b, new THREE.Vector3()) !== null; };
            if (activeR.hasTool && hit(activeR.gunMesh)) return ['tool'];
            for (const R of stationRobots) { if (hit(R.links[0]) || hit(R.links[1]) || hit(R.links[2]) || hit(R.links[3])) return ['robot', R]; }
            let best = null, bd = 1e9;
            stationObjs.forEach(o => { if (!o.root.visible) return; const h = ray.intersectObject(o.mesh, false); if (h.length && h[0].distance < bd) { bd = h[0].distance; best = o.idx; } });
            if (best !== null) return ['obj', best];
            if (data.jigData && data.jigData.length && hit(jigMesh)) return ['jig'];
            return null;
        }
        function selectPicked(p) { if (!p) { selectPart(null); return; } if (p[0] === 'robot') { activate(p[1]); selectPart('robot'); } else selectPart(p[0], p[1]); }
        let downPt = null;
        renderer.domElement.addEventListener('pointerdown', (e) => { downPt = { x: e.clientX, y: e.clientY }; });
        renderer.domElement.addEventListener('pointerup', (e) => {
            if (!downPt || busy()) return;
            const moved = Math.hypot(e.clientX - downPt.x, e.clientY - downPt.y); downPt = null;
            if (moved > 4 || partGizmo.axis || transformGizmo.axis || e.button !== 0) return;
            selectPicked(pickAt(e));
        });

        // ---------- numeric fields (Layout tab) ----------
        function mkGrid(id, items) {
            const g = $(id); g.innerHTML = '';
            items.forEach(([key, label]) => { const d = document.createElement('div'); d.innerHTML = '<label>' + label + '</label><input type="number" step="any" id="f-' + key + '">'; g.appendChild(d); });
        }
        mkGrid('g-base', [['bx', 'X m'], ['by', 'Y m'], ['bz', 'Z m'], ['byaw', 'Turn °']]);
        mkGrid('g-tool', [['tx', 'X mm'], ['ty', 'Y mm'], ['tz', 'Z mm'], ['rx', 'Rx °'], ['ry', 'Ry °'], ['rz', 'Rz °']]);
        mkGrid('g-obj', [['ox', 'X m'], ['oy', 'Y m'], ['oz', 'Z m'], ['orx', 'Rx °'], ['ory', 'Ry °'], ['orz', 'Rz °'], ['osc', 'Scale']]);
        mkGrid('g-jig', [['jx', 'X m'], ['jy', 'Y m'], ['jz', 'Z m'], ['jrx', 'Rx °'], ['jry', 'Ry °'], ['jrz', 'Rz °']]);
        const FIELDS = ['bx', 'by', 'bz', 'byaw', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'jx', 'jy', 'jz', 'jrx', 'jry', 'jrz', 'ox', 'oy', 'oz', 'orx', 'ory', 'orz', 'osc'];
        function syncFields() {
            if (!$('pane-layout').classList.contains('active')) return;
            const jr = new THREE.Euler().setFromQuaternion(jigMesh.quaternion, 'XYZ'), rp = activeR.root.position;
            const v = { bx: rp.x, by: rp.y, bz: rp.z, byaw: activeR.root.rotation.z * R2D, tx: data.toolOffsetX * 1000, ty: data.toolOffsetY * 1000, tz: data.toolOffsetZ * 1000, rx: data.toolRotX * R2D, ry: data.toolRotY * R2D, rz: data.toolRotZ * R2D, jx: data.jigX, jy: data.jigY, jz: data.jigZ, jrx: jr.x * R2D, jry: jr.y * R2D, jrz: jr.z * R2D };
            const so = stationObjs[selObj];
            if (so) { const oe = new THREE.Euler().setFromQuaternion(so.root.quaternion, 'XYZ'); Object.assign(v, { ox: so.root.position.x, oy: so.root.position.y, oz: so.root.position.z, orx: oe.x * R2D, ory: oe.y * R2D, orz: oe.z * R2D, osc: so.sc }); }
            $('obj-section').style.display = stationObjs.length ? 'block' : 'none';
            $('obj-pick').innerText = so ? 'Object: ' + so.name : '';
            FIELDS.forEach(k => { const el = $('f-' + k); if (!el || document.activeElement === el) return; if (k[0] === 'o' && !so) { el.value = ''; return; } el.value = k === 'osc' ? (+v[k]).toPrecision(4) : (+v[k]).toFixed(k.length === 2 && (k[0] === 'j' || k[0] === 'b' || k[0] === 'o') ? 3 : 1); });
        }
        setInterval(syncFields, 200);
        FIELDS.forEach(k => $('f-' + k).addEventListener('change', () => {
            const n = parseFloat($('f-' + k).value); if (isNaN(n)) return;
            const rp = activeR.root.position;
            if (k === 'bx') rp.x = n; if (k === 'by') rp.y = n; if (k === 'bz') rp.z = n; if (k === 'byaw') activeR.root.rotation.z = n * D2R;
            if (k === 'tx') data.toolOffsetX = n / 1000; if (k === 'ty') data.toolOffsetY = n / 1000; if (k === 'tz') data.toolOffsetZ = n / 1000;
            if (k === 'rx') data.toolRotX = n * D2R; if (k === 'ry') data.toolRotY = n * D2R; if (k === 'rz') data.toolRotZ = n * D2R;
            if (k === 'jx') data.jigX = n; if (k === 'jy') data.jigY = n; if (k === 'jz') data.jigZ = n;
            if (k[0] === 'j' && k[1] === 'r') {
                const e = new THREE.Euler().setFromQuaternion(jigMesh.quaternion, 'XYZ');
                if (k === 'jrx') e.x = n * D2R; if (k === 'jry') e.y = n * D2R; if (k === 'jrz') e.z = n * D2R;
                jigMesh.quaternion.setFromEuler(e);
            }
            const so = stationObjs[selObj];
            if (so && k[0] === 'o') {
                if (k === 'ox') so.root.position.x = n; if (k === 'oy') so.root.position.y = n; if (k === 'oz') so.root.position.z = n;
                if (k === 'osc' && n > 0) { so.sc = n; so.mesh.scale.setScalar(n); applyKeep(so); }
                if (k[1] === 'r') { const e = new THREE.Euler().setFromQuaternion(so.root.quaternion, 'XYZ'); if (k === 'orx') e.x = n * D2R; if (k === 'ory') e.y = n * D2R; if (k === 'orz') e.z = n * D2R; so.root.quaternion.setFromEuler(e); }
            }
            refreshSceneDisplay(false); markDirty();
        }));
        function applyKeep(o) { o.mesh.position.copy(o.keep ? o.cen.clone().multiplyScalar(o.sc) : new THREE.Vector3()); }
        $('btn-okeep').addEventListener('click', () => { const o = stationObjs[selObj]; if (!o) { toast('Select an uploaded object first'); return; } o.keep = !o.keep; applyKeep(o); markDirty(); toast(o.keep ? 'Using the CAD origin of ' + o.name : 'Centred on its middle'); });
        $('btn-snap').addEventListener('click', () => { data.toolOffsetX = data.toolOffsetY = data.toolOffsetZ = 0; data.toolRotX = data.toolRotY = data.toolRotZ = 0; refreshSceneDisplay(false); markDirty(); toast('Tool snapped to the flange'); });
        function cadMesh() { return internalJigContent.children[0]; }
        $('btn-cadx').addEventListener('click', () => { const m = cadMesh(); if (m) { m.rotation.x += Math.PI / 2; markDirty(); } else toast('No jig loaded'); });
        $('btn-cady').addEventListener('click', () => { const m = cadMesh(); if (m) { m.rotation.y += Math.PI / 2; markDirty(); } else toast('No jig loaded'); });

        /* ======================= STATION SEQUENCE ======================= */
        let SEQ = [];
        const SR = { on: false, i: -1, tasks: [], t: 0, last: 0, skips: 0 };
        function renderStationRobots() {
            commitRobot(activeR);
            const box = $('st-robots'); box.innerHTML = '';
            stationRobots.forEach(R => {
                const d = document.createElement('div'); d.className = 'rbadge' + (R === activeR ? ' act' : '');
                d.innerHTML = '<b>' + R.label + '</b><span style="flex:1">' + R.profile.replace('_', ' ') + (R.hasTool ? ' · tool ✓' : ' · no tool') + '</span><span class="chip">' + R.traj.length + ' steps</span><button class="mini" data-act="' + R.idx + '">Select</button>';
                box.appendChild(d);
            });
        }
        $('st-robots').addEventListener('click', (e) => { const a = e.target.getAttribute('data-act'); if (a !== null) { activate(stationRobots[+a]); renderStationRobots(); } });
        function robotChecks(i, st) {
            return stationRobots.map(R => '<label class="rchip"><input type="checkbox" data-seq="' + i + '" data-r="' + R.idx + '"' + (st.r[R.idx] ? ' checked' : '') + '> ' + R.label + '</label>').join('');
        }
        function renderSeq() {
            const tb = $('seq-body'); tb.innerHTML = '';
            SEQ.forEach((st, i) => {
                const tr = document.createElement('tr'); tr.dataset.i = i; if (SR.on && SR.i === i) tr.className = 'seqcur';
                let body = '';
                if (st.k === 'run') body = 'Run ' + robotChecks(i, st);
                else if (st.k === 'home') body = 'Home ' + robotChecks(i, st);
                else if (st.k === 'e1') body = 'Turn E1 to <input type="number" step="5" style="width:64px" data-seq="' + i + '" data-p="deg" value="' + st.deg + '"> °';
                else body = 'Wait <input type="number" step="0.5" min="0" style="width:64px" data-seq="' + i + '" data-p="s" value="' + st.s + '"> s';
                tr.innerHTML = '<td>' + (i + 1) + '</td><td>' + body + '</td><td style="white-space:nowrap"><button class="mini" data-up="' + i + '">↑</button> <button class="mini" data-dn="' + i + '">↓</button> <button class="mini" data-rm="' + i + '">✕</button></td>';
                tb.appendChild(tr);
            });
            $('seq-empty').style.display = SEQ.length ? 'none' : 'block';
        }
        $('seq-body').addEventListener('click', (e) => {
            if (SR.on) return;
            const up = e.target.getAttribute('data-up'), dn = e.target.getAttribute('data-dn'), rm = e.target.getAttribute('data-rm');
            if (up !== null && +up > 0) { const i = +up; [SEQ[i - 1], SEQ[i]] = [SEQ[i], SEQ[i - 1]]; }
            else if (dn !== null && +dn < SEQ.length - 1) { const i = +dn; [SEQ[i + 1], SEQ[i]] = [SEQ[i], SEQ[i + 1]]; }
            else if (rm !== null) SEQ.splice(+rm, 1); else return;
            renderSeq(); markDirty();
        });
        $('seq-body').addEventListener('change', (e) => {
            const i = e.target.getAttribute('data-seq'); if (i === null) return; const st = SEQ[+i];
            if (e.target.type === 'checkbox') st.r[+e.target.getAttribute('data-r')] = e.target.checked ? 1 : 0;
            else if (e.target.getAttribute('data-p') === 'deg') st.deg = parseFloat(e.target.value) || 0; else st.s = Math.max(0, parseFloat(e.target.value) || 0);
            markDirty();
        });
        $('seq-add').addEventListener('click', () => {
            const k = $('seq-kind').value, r = stationRobots.map(R => R === activeR ? 1 : 0);
            SEQ.push(k === 'run' || k === 'home' ? { k: k, r: r } : k === 'e1' ? { k: k, deg: 90 } : { k: k, s: 1 });
            renderSeq(); markDirty();
        });
        $('seq-auto').addEventListener('click', () => {
            commitRobot(activeR); SEQ = [];
            stationRobots.forEach(R => { if (R.traj.length >= 2) SEQ.push({ k: 'run', r: stationRobots.map(Q => Q === R ? 1 : 0) }); });
            if (!SEQ.length) toast('Record at least 2 steps for a robot first.'); else toast('Sequence built: robots run one after another.');
            renderSeq(); markDirty();
        });
        $('seq-clear').addEventListener('click', () => { if (SR.on) return; SEQ = []; renderSeq(); markDirty(); });

        function applyPose(R, transforms, e1) {
            const ex = (e1 === undefined) ? e1Global : e1;
            withRobot(R, () => updateSceneTransforms(transforms, R.tool.ox, R.tool.oy, R.tool.oz, R.tool.rx, R.tool.ry, R.tool.rz, data.jigX, data.jigY, data.jigZ, ex));
        }
        function setE1(v) { e1Global = v; internalJigContent.rotation.z = v; }
        function lerpAngles(a, b, f) { return a.map((v, i) => v + (b[i] - v) * f); }
        function seqNext() {
            SR.i++;
            if (SR.i >= SEQ.length) { if ($('chk-sloop').checked && SR.skips <= SEQ.length) { SR.i = 0; } else { seqStop(); toast('Station sequence finished'); return; } }
            const s = SEQ[SR.i]; SR.tasks = []; SR.t = 0;
            if (s.k === 'run') stationRobots.forEach(R => { if (s.r[R.idx] && R.traj.length >= 2) { const es = R.traj.map(p => p.angles[7]); SR.tasks.push({ k: 'run', R: R, i: 0, f: 0, e1: Math.max(...es) - Math.min(...es) > 0.01 }); } });
            else if (s.k === 'home') stationRobots.forEach(R => { if (s.r[R.idx]) SR.tasks.push({ k: 'home', R: R, from: [...R.angles], f: 0 }); });
            else if (s.k === 'e1') SR.tasks.push({ k: 'e1', from: e1Global, to: s.deg * D2R, f: 0 });
            if (s.k !== 'wait' && !SR.tasks.length) { SR.skips++; seqNext(); return; }
            SR.skips = 0; renderSeq();
            $('seq-status').innerText = 'Step ' + (SR.i + 1) + ' of ' + SEQ.length;
        }
        function seqStart() {
            if (runSimulation) return;
            if (!SEQ.length) { toast('Add at least one step, or press "Auto".'); return; }
            commitRobot(activeR); stationRobots.forEach(R => { R.angles[7] = e1Global; });
            SR.on = true; SR.i = -1; SR.skips = 0; SR.last = performance.now(); selectPart(null); seqNext();
        }
        function seqStop() {
            if (!SR.on) return;
            SR.on = false;
            SR.tasks.forEach(T => {      // keep the pose where it stopped
                if (T.done) return;
                if (T.k === 'run') { const a = T.R.traj[T.i], b = T.R.traj[Math.min(T.i + 1, T.R.traj.length - 1)]; T.R.angles = lerpAngles(a.angles, b.angles, T.f); }
                else if (T.k === 'home') T.R.angles = lerpAngles(T.from, new Array(8).fill(0), T.f);
            });
            stationRobots.forEach(R => { R.angles[7] = e1Global; });
            const act = activeR;
            stationRobots.forEach(R => { if (R !== act) applyPose(R, withRobot(R, () => computeForwardKinematics(R.angles))); });
            bindRobot(act); localJointAngles = act.angles; embeddedTrajectory = act.traj; refreshSceneDisplay(true);
            $('seq-status').innerText = 'Idle'; renderSeq(); renderStationRobots(); markDirty();
        }
        function seqTick() {
            if (!SR.on) return;
            const now = performance.now(), dt = Math.min(0.1, (now - SR.last) / 1000); SR.last = now;
            const f60 = dt * 60, sp = parseFloat($('sld-speed').value) / 100, s = SEQ[SR.i];
            if (s.k === 'wait') { SR.t += dt; if (SR.t >= s.s) seqNext(); return; }
            SR.tasks.forEach(T => {
                if (T.done) return;
                if (T.k === 'run') {
                    const tr = T.R.traj; T.f += sp * 0.04 * f60;
                    if (T.f >= 1) { T.f = 0; T.i++; if (T.i >= tr.length - 1) { T.done = true; const last = tr[tr.length - 1]; T.R.angles = [...last.angles]; if (T.e1) setE1(last.angles[7]); T.R.angles[7] = e1Global; applyPose(T.R, last.transforms); return; } }
                    const a = tr[T.i], b = tr[T.i + 1], bl = [];
                    for (let i = 0; i < 7; i++) {
                        bl.push({ pos: new THREE.Vector3().lerpVectors(new THREE.Vector3().fromArray(a.transforms[i].pos), new THREE.Vector3().fromArray(b.transforms[i].pos), T.f).toArray(),
                                  quat: new THREE.Quaternion().fromArray(a.transforms[i].quat).slerp(new THREE.Quaternion().fromArray(b.transforms[i].quat), T.f).toArray() });
                    }
                    if (T.e1) setE1((1 - T.f) * a.angles[7] + T.f * b.angles[7]);
                    applyPose(T.R, bl);
                } else if (T.k === 'home') {
                    T.f += sp * 0.02 * f60; const f = Math.min(1, T.f), z = new Array(8).fill(0); z[7] = e1Global;
                    const ang = lerpAngles(T.from, z, f); ang[7] = e1Global;
                    applyPose(T.R, withRobot(T.R, () => computeForwardKinematics(ang)));
                    if (f >= 1) { T.done = true; T.R.angles = ang; }
                } else if (T.k === 'e1') {
                    T.f += sp * 0.02 * f60; const f = Math.min(1, T.f); setE1(T.from + (T.to - T.from) * f); if (f >= 1) T.done = true;
                }
            });
            if (SR.tasks.every(T => T.done)) seqNext();
        }
        $('btn-seq-run').addEventListener('click', () => { if (SR.on) seqStop(); else seqStart(); });
        (function sloop() { requestAnimationFrame(sloop); seqTick(); })();

        /* ======================= SAVE / LOAD (browser localStorage, per station set-up) ======================= */
        const SKEY = 'olp.station.v3';
                function saveStation() {
            try {
                commitRobot(activeR); const m = cadMesh();
                let old = []; try { const o = JSON.parse(localStorage.getItem(SKEY) || '{}'); old = o.robots || []; } catch (e) { }
                const robots = stationRobots.map(R => ({ p: R.profile, tn: R.toolName, base: { x: R.root.position.x, y: R.root.position.y, z: R.root.position.z, yaw: R.root.rotation.z }, tool: R.tool, angles: R.angles, traj: R.traj }));
                for (let i = robots.length; i < old.length; i++) robots.push(old[i]);      // keep robots that are switched off in Setup, so they come back as they were
                localStorage.setItem(SKEY, JSON.stringify({
                    e1: e1Global, seq: SEQ, robots: robots,
                    objs: stationObjs.map(o => ({ n: o.name, p: o.root.position.toArray(), q: o.root.quaternion.toArray(), s: o.sc, k: o.keep })),
                    jig: { p: [data.jigX, data.jigY, data.jigZ], q: jigMesh.quaternion.toArray(), cx: m ? m.rotation.x : null, cy: m ? m.rotation.y : null }
                }));
            } catch (e) { }
        }
        function loadStation() {
            try {
                const s = localStorage.getItem(SKEY); if (!s) return; const d = JSON.parse(s);
                if (d.jig) { [data.jigX, data.jigY, data.jigZ] = d.jig.p; jigMesh.quaternion.fromArray(d.jig.q); const m = cadMesh(); if (m && d.jig.cx !== null) { m.rotation.x = d.jig.cx; m.rotation.y = d.jig.cy; } }
                e1Global = d.e1 || 0;
                (d.objs || []).forEach((os, i) => { const o = stationObjs[i]; if (!o || o.name !== os.n) return; o.root.position.fromArray(os.p); o.root.quaternion.fromArray(os.q); o.sc = os.s; o.mesh.scale.setScalar(os.s); o.keep = !!os.k; applyKeep(o); });
                (d.robots || []).forEach((rs, i) => {            // each robot is restored on its own, so adding robot 2 does not disturb robot 1
                    const R = stationRobots[i]; if (!R || rs.p !== R.profile) return;
                    R.root.position.set(rs.base.x, rs.base.y, rs.base.z); R.root.rotation.z = rs.base.yaw;
                    if (rs.tn === R.toolName) { R.tool = rs.tool; R.angles = rs.angles; R.traj = rs.traj; }   // tool offset and program only if the same tool is still chosen
                });
                SEQ = (d.seq || []).map(st => { if (st.r) { const r = stationRobots.map((_, k) => st.r[k] ? 1 : 0); st.r = r; } return st; });
                bindRobot(activeR);
            } catch (e) { }
        }
        setInterval(() => { if (dirty) { dirty = false; saveStation(); } }, 1500);
        window.addEventListener('pagehide', saveStation);
        $('btn-lay-reset').addEventListener('click', () => {
            try { localStorage.removeItem(SKEY); } catch (e) { }
            stationRobots.forEach((R, i) => { const b = data.robots[i].base; R.root.position.set(b.x, b.y, b.z || 0); R.root.rotation.z = b.yaw || 0; R.tool = { ox: 0, oy: 0, oz: 0, rx: 0, ry: 0, rz: 0 }; });
            bindRobot(activeR);
            stationObjs.forEach(o => { o.root.position.fromArray(o.def); o.root.quaternion.set(0, 0, 0, 1); });
            data.jigX = data.jigX0; data.jigY = data.jigY0; data.jigZ = data.jigZ0; jigMesh.quaternion.set(0, 0, 0, 1);
            const m = cadMesh(); if (m) { m.rotation.x = data.rotX; m.rotation.y = data.rotY; }
            refreshSceneDisplay(false); toast('Layout reset (robot programs kept)');
        });
        data.jigX0 = data.jigX; data.jigY0 = data.jigY; data.jigZ0 = data.jigZ;

        // keep the 3D canvas sized to its panel
        if (window.ResizeObserver) new ResizeObserver(() => {
            if (!container.clientWidth || !container.clientHeight) return;
            camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }).observe(container);


        /* ======================= RIGHT-CLICK MENU (move / rotate on all axes or one axis) ======================= */
        const ctx = document.createElement('div'); ctx.id = 'ctxmenu'; $('view').appendChild(ctx);
        function closeCtx() { ctx.style.display = 'none'; }
        function openCtx(x, y) {
            const is = (m, a) => gMode === m && axisSel === a ? ' on' : '';
            const items = [['h', selPart === 'tool' ? '🔧 Tool' : selPart === 'robot' ? '🦾 ' + rName(activeR) : selPart === 'obj' ? '🧱 ' + stationObjs[selObj].name : '⚙ Jig'],
                ['translate', 'all', '⤢ Move: all axes'], ['translate', 'x', 'Move along X', '#ff6b6b'], ['translate', 'y', 'Move along Y', '#6bff6b'], ['translate', 'z', 'Move along Z', '#6b8bff'],
                ['sep'],
                ['rotate', 'all', '⟳ Rotate: all axes'], ['rotate', 'x', 'Rotate about X', '#ff6b6b'], ['rotate', 'y', 'Rotate about Y', '#6bff6b'], ['rotate', 'z', 'Rotate about Z', '#6b8bff'],
                ['sep'], ['space', '🌐 Space: ' + (gSpace === 'world' ? 'World → switch to Local' : 'Local → switch to World')], ['desel', '✕ Deselect']];
            ctx.innerHTML = '';
            items.forEach(it => {
                if (it[0] === 'sep') { ctx.appendChild(Object.assign(document.createElement('div'), { className: 'csep' })); return; }
                if (it[0] === 'h') { ctx.appendChild(Object.assign(document.createElement('div'), { className: 'chead', innerText: it[1] })); return; }
                const b = document.createElement('button'); b.className = 'citem' + ((it[0] === 'translate' || it[0] === 'rotate') ? is(it[0], it[1]) : '');
                b.innerHTML = (it[3] ? '<i style="background:' + it[3] + '"></i>' : '') + (it[0] === 'space' || it[0] === 'desel' ? it[1] : it[2]);
                b.addEventListener('click', () => {
                    if (it[0] === 'space') toggleSpace(); else if (it[0] === 'desel') selectPart(null); else setGizmoMode(it[0], it[1]);
                    closeCtx();
                });
                ctx.appendChild(b);
            });
            const vr = $('view').getBoundingClientRect();
            ctx.style.display = 'block';
            ctx.style.left = Math.max(4, Math.min(x - vr.left, vr.width - ctx.offsetWidth - 4)) + 'px';
            ctx.style.top = Math.max(4, Math.min(y - vr.top, vr.height - ctx.offsetHeight - 4)) + 'px';
        }
        let rDown = null;
        renderer.domElement.addEventListener('pointerdown', (e) => { if (e.button === 2) rDown = { x: e.clientX, y: e.clientY }; else closeCtx(); });
        renderer.domElement.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (busy()) return;
            if (rDown && Math.hypot(e.clientX - rDown.x, e.clientY - rDown.y) > 5) return;   // it was a right-drag (pan), not a click
            if (!selPart) {
                selectPicked(pickAt(e));
                if (!selPart) { toast('Select an object first (left-click the robot, tool, jig or an uploaded object), then right-click.'); return; }
            }
            openCtx(e.clientX, e.clientY);
        });
        window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeCtx(); });

        /* ---------- hide / show the Simulation Setup panel (it lives in the Streamlit page above this frame) ---------- */
        function setupEl() { try { return window.parent.document.querySelector('[data-testid="stExpander"]'); } catch (e) { return null; } }
        function setSetupHidden(h, quiet) {
            const el = setupEl(), fr = window.frameElement;
            if (!el) { if (!quiet) toast('Click the "Simulation Setup" bar above this window to hide or show it.'); return; }
            el.style.display = h ? 'none' : '';
            if (fr) { try { fr.style.height = h ? Math.max(620, window.parent.innerHeight - 40) + 'px' : '820px'; } catch (e) { } }
            $('btn-setup').innerText = h ? '⚙ Show setup' : '⚙ Hide setup';
            try { localStorage.setItem('olp.hideSetup', h ? '1' : '0'); } catch (e) { }
            if (!h) { try { window.parent.scrollTo(0, 0); } catch (e) { } }
        }
        $('btn-setup').addEventListener('click', () => setSetupHidden($('btn-setup').innerText.indexOf('Hide') >= 0));
        try { if (localStorage.getItem('olp.hideSetup') === '1') setSetupHidden(true, true); } catch (e) { }

        // ---------- init ----------
        buildTree(); setGizmoMode('translate'); applySnap();
        loadStation();
        stationRobots.forEach(R => { R.angles[7] = e1Global; applyPose(R, withRobot(R, () => computeForwardKinematics(R.angles))); });   // put every robot in its configured home pose
        buildJogRows(); updateChips(); updateTreeClasses(); renderSeq();
        setView('iso');
        setInterval(updateReadouts, 120);
        setInterval(updateRunUI, 150);

        refreshSceneDisplay(true); updateUIElements(); animate();
    </script>
</body>
</html>"""
    
    html_source = html_source.replace("__PAYLOAD_STREAM__", json_stream)
    components.html(html_source, height=820, scrolling=False)

# --- 9. DYNAMIC HARDWARE FILE BINDING LAYER (several robots share meshes of the same model) ---
path_jig = os.path.join(TEMP_DIR, "jig.stl")

profiles_payload, tools_payload, robots_payload = {}, {}, []
for k, cfg in enumerate(robot_cfgs):
    prof = cfg["profile"]
    if prof not in profiles_payload:
        pcfg = ROBOT_REGISTRY.get(prof, ROBOT_REGISTRY["ABB_6700"])
        link_b64s = []
        for i in range(7):
            mesh_filename = f"link_{i}.stl" if i > 0 else "base_link.stl"
            target_mesh_path = os.path.join(BASE_DIR, "assets", "robots", prof, mesh_filename)
            if not os.path.exists(target_mesh_path):
                target_mesh_path = os.path.join(BASE_DIR, "assets", "meshes", mesh_filename)
            link_b64s.append(get_file_base64_cached(target_mesh_path))
        profiles_payload[prof] = {
            "linkGeometries": link_b64s,
            "fallbackHeights": pcfg["fallback_heights"],
            "dhConfig": pcfg["links"],
            "jointLimits": pcfg["limits"],
        }
    tool_key, tool_name = "", ""
    tp = cfg["tool_path"]
    if tp and os.path.exists(tp):
        tool_key, tool_name = tp, os.path.basename(tp)
        if tool_key not in tools_payload:
            tools_payload[tool_key] = get_file_base64_cached(tp, get_file_hash(tp))
    bx, by, bz, byaw = DEFAULT_BASES[k]
    robots_payload.append({"profile": prof, "toolKey": tool_key, "toolName": tool_name,
                           "base": {"x": bx, "y": by, "z": bz, "yaw": float(byaw)}})

objects_payload = []
for uf in (up_objs or [])[:8]:
    try:
        objects_payload.append({"name": os.path.splitext(uf.name)[0], "data": base64.b64encode(uf.getvalue()).decode("utf-8")})
    except Exception:
        pass

scene_payload = {
    "objects": objects_payload,
    "profiles": profiles_payload,
    "tools": tools_payload,
    "robots": robots_payload,
    "jigData": get_file_base64_cached(path_jig, get_file_hash(path_jig)),
    "jigX": jx_pos, "jigY": jy_pos, "jigZ": jz_pos,
    "rotX": float(j_rot_x) * (np.pi / 180.0),
    "rotY": float(j_rot_y) * (np.pi / 180.0),
    "jigScale": js_scale,
}

build_embedded_viewport(scene_payload)
