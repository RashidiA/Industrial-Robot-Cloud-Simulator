import streamlit as st
import streamlit.components.v1 as components
import os
import json
import base64

# --- 1. SYSTEM INITIALIZATION & CLOUD HOOKS ---
st.set_page_config(page_title="Cloud Robot OLP Platform", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
ROBOTS_DIR = os.path.join(ASSETS_DIR, "robots")
GUNS_DIR = os.path.join(ASSETS_DIR, "guns")

for d in [ROBOTS_DIR, GUNS_DIR]:
    os.makedirs(d, exist_ok=True)

if 'program' not in st.session_state:
    st.session_state.program = []
if 'j_angles' not in st.session_state:
    st.session_state.j_angles = [0.0] * 8

def get_directories_list(directory_path):
    if not os.path.exists(directory_path):
        return []
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

def get_base64_mesh(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            return ""
    return ""

# --- 2. DYNAMIC KINEMATIC DIMENSION PROFILES ---
# Dictates exact physical offsets to prevent link disconnection during WebGL assembly matrix builds
ROBOT_DIMENSIONS = {
    "ABB_6700":     {"L0": 0.78,  "L1": 0.32, "L2": 1.28, "L3": 1.142, "L3_Z": 0.2,  "L4": 0.2, "L5": 0.2},
    "ABB_6600":     {"L0": 0.715, "L1": 0.32, "L2": 1.075, "L3": 1.142, "L3_Z": 0.2,  "L4": 0.2, "L5": 0.2},
    "ABB_4400":     {"L0": 0.68,  "L1": 0.15, "L2": 0.88,  "L3": 0.78,  "L3_Z": 0.15, "L4": 0.15, "L5": 0.1},
    "KUKA_KR150":   {"L0": 0.75,  "L1": 0.35, "L2": 1.25,  "L3": 1.10,  "L3_Z": 0.22, "L4": 0.21, "L5": 0.19},
    "default":      {"L0": 0.78,  "L1": 0.32, "L2": 1.28,  "L3": 1.142, "L3_Z": 0.2,  "L4": 0.2, "L5": 0.2}
}

# --- 3. RE-ROUTE STREAM CONTROLLER ---
query_params = st.query_params
if "event" in query_params:
    event_type = query_params.get("event")
    if event_type == "sync_sequence":
        try:
            st.session_state.program = json.loads(query_params.get("program_data", "[]"))
            if len(st.session_state.program) > 0:
                st.session_state.j_angles = st.session_state.program[-1]["angles"]
        except Exception as e:
            st.error(f"Sync error: {e}")
    elif event_type == "clear_sequence":
        st.session_state.program = []
        st.session_state.j_angles = [0.0] * 8
    st.query_params.clear()

# --- 4. CONTROL INTERFACE SIDEBAR ---
with st.sidebar:
    st.title("📟 Cloud Pendant Pro")
    st.caption("Edge-Computing Kinematics Engine.")
    
    with st.expander("🤖 Hardware Directories Profile", expanded=True):
        robot_options = get_directories_list(ROBOTS_DIR)
        selected_robot = st.selectbox("Select Robot Library Profile", options=robot_options if robot_options else list(ROBOT_DIMENSIONS.keys())[:-1])
        
        gun_options = get_directories_list(GUNS_DIR)
        selected_gun = st.selectbox("Select End-Effecter Gun Library", options=gun_options if gun_options else ["None / Custom Tool Profile"])

    with st.expander("🏗️ Workcell Jig Setup", expanded=False):
        jx_pos = st.number_input("Jig Base Position X", value=1.6, step=0.1)
        jy_pos = st.number_input("Jig Base Position Y", value=0.0, step=0.1)
        jz_pos = st.number_input("Jig Base Elevation Z", value=0.55, step=0.01)
        st.divider()
        j_rot_x = st.slider("CAD Adjust Rotate X", -180, 180, 0, step=90)
        j_rot_y = st.slider("CAD Adjust Rotate Y", -180, 180, 0, step=90)
        js_scale = st.number_input("Jig Local Scale Factor multiplier", value=0.001, format="%.5f")

    with st.expander("🔫 Tool Orientation Adjustment", expanded=False):
        g_off_x = st.slider("Gun Offset Axis (TCP)", -0.5, 0.5, 0.0, step=0.01)
        g_rot_z = st.slider("Gun Tool Orientation Vector Rotation", -180, 180, 180, step=90)

# Extract matching geometric configuration array based on chosen profile
dimensions_profile = ROBOT_DIMENSIONS.get(selected_robot, ROBOT_DIMENSIONS["default"])

robot_geometry_streams = []
if os.path.exists(os.path.join(ROBOTS_DIR, selected_robot)):
    target_r_path = os.path.join(ROBOTS_DIR, selected_robot)
    for i in range(7):
        mesh_filename = f"link_{i}.stl" if i > 0 else "base_link.stl"
        robot_geometry_streams.append(get_base64_mesh(os.path.join(target_r_path, mesh_filename)))

gun_geometry_stream = ""
if selected_gun != "None / Custom Tool Profile":
    gun_geometry_stream = get_base64_mesh(os.path.join(GUNS_DIR, selected_gun, "tool.stl"))

payload_data = {
    "trajectory": st.session_state.program,
    "initialAngles": st.session_state.j_angles,
    "robotMeshes": robot_geometry_streams,
    "dimensions": dimensions_profile,
    "gunMesh": gun_geometry_stream,
    "jigConfig": {
        "x": jx_pos, "y": jy_pos, "z": jz_pos,
        "rotX": float(j_rot_x) * 0.017453292519943295,
        "rotY": float(j_rot_y) * 0.017453292519943295,
        "scale": js_scale
    },
    "gunConfig": {
        "offset": g_off_x,
        "rotZ": float(g_rot_z) * 0.017453292519943295
    }
}

# --- 5. COMPACTED GRAPHICS SYSTEM ENGINE ---
def render_cloud_viewport(payload):
    json_payload = json.dumps(payload)
    
    html_source = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body { margin: 0; background-color: #111; overflow: hidden; font-family: Arial, sans-serif; user-select: none; }
            #canvas-viewport { width: 100vw; height: 100vh; position: absolute; top:0; left:0; z-index:1; }
            #telemetry-bar { position: absolute; top: 10px; left: 10px; color: #00ffcc; font-size: 11px; font-family: monospace; background: rgba(15,15,15,0.75); padding: 5px 10px; border-radius:4px; border: 1px solid #333; z-index: 10; }
            
            /* UI COMPACTION MODIFICATIONS: Reduced width, padding, font sizing, and increased transparency */
            #pendant-container { position: absolute; top: 10px; right: 10px; background: rgba(20, 20, 20, 0.75); border: 1px solid #ff9800; border-radius: 5px; width: 175px; padding: 8px; color: white; z-index: 10; box-shadow: 0 6px 15px rgba(0,0,0,0.6); backdrop-filter: blur(3px); }
            .pendant-header { font-size: 9px; font-weight: bold; letter-spacing: 0.5px; color: #ff9800; text-align: center; border-bottom: 1px solid #333; padding-bottom: 4px; margin-bottom: 6px; }
            .jog-axis-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
            .axis-label { font-size: 10px; font-weight: bold; font-family: monospace; color: #ffb74d; }
            .axis-btn { background: #222; border: 1px solid #444; color: white; width: 32px; height: 22px; font-size: 12px; font-weight: bold; cursor: pointer; border-radius: 3px; }
            .axis-btn:active { background: #ff9800; color: #000; border-color: #ff9800; }
            .axis-readout { font-family: monospace; font-size: 10px; color: #00ffcc; width: 50px; text-align: center; }
            
            .jig-dropzone { border: 1px dashed #00e5ff; background: rgba(0, 229, 255, 0.03); border-radius: 3px; padding: 4px; font-size: 9px; text-align: center; color: #00e5ff; margin-bottom: 6px; }
            .control-actions { margin-top: 4px; border-top: 1px solid #333; padding-top: 6px; display: flex; flex-direction: column; gap: 4px; }
            .btn-core { width: 100%; border: none; font-weight: bold; height: 26px; border-radius: 3px; cursor: pointer; font-size: 10px; display: flex; align-items: center; justify-content: center; }
            #btn-rec { background: #ff9800; color: #000; }
            #btn-play { background: #4caf50; color: #fff; }
            #btn-del { background: #f44336; color: #fff; }
            .status-counter { font-size: 9px; text-align: center; color: #aaa; font-family: monospace; margin-top: 2px; }
        </style>
    </head>
    <body>
        <div id="telemetry-bar">EDGE KINEMATICS ENGINE: ONLINE</div>
        
        <div id="pendant-container">
            <div class="pendant-header">⚡ WEBGL DIRECT JOG</div>
            <div id="axes-container"></div>
            <div class="jig-dropzone" id="jig-drop">📂 DROP STL HERE</div>
            <input type="file" id="jig-file-hidden" accept=".stl" style="display:none;">

            <div class="control-actions">
                <div class="jog-axis-row" style="margin-bottom: 2px;">
                    <input type="range" id="sim-speed" min="5" max="100" value="50" style="flex-grow:1; margin:0 4px; height:3px; accent-color:#ff9800;">
                    <div id="speed-label" style="font-size:9px; color:#ff9800; width:22px; font-family:monospace; text-align:right;">50%</div>
                </div>
                <button class="btn-core" id="btn-rec">💾 RECORD POSITION</button>
                <button class="btn-core" id="btn-play">▶️ RUN INTERPOLATION</button>
                <button class="btn-core" id="btn-del">🗑️ WIPE SEQUENCE</button>
                <div class="status-counter" id="steps-counter">Steps: 0</div>
            </div>
        </div>

        <div id="canvas-viewport"></div>

        <script>
            const coreData = __INJECTED_PAYLOAD__;
            const dims = coreData.dimensions; // Read structural profile dimensions dynamically
            
            let currentAngles = [...coreData.initialAngles];
            let memoryBuffer = [...coreData.trajectory];
            const J_INCREMENT = 5 * (Math.PI / 180);

            THREE.Object3D.DefaultUp.set(0, 0, 1);
            const host = document.getElementById('canvas-viewport');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x141414);

            const camera = new THREE.PerspectiveCamera(45, host.clientWidth / host.clientHeight, 0.01, 100);
            camera.position.set(3.5, -3.5, 2.5);

            const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
            renderer.setSize(host.clientWidth, host.clientHeight);
            host.appendChild(renderer.domElement);

            const orbit = new THREE.OrbitControls(camera, renderer.domElement);
            orbit.target.set(0.6, 0, 0.8);

            scene.add(new THREE.AmbientLight(0x666666));
            const sunLight = new THREE.DirectionalLight(0xffffff, 0.8); sunLight.position.set(5, 5, 8); scene.add(sunLight);
            
            const grid = new THREE.GridHelper(16, 32, 0x444444, 0x222222);
            grid.rotation.x = Math.PI / 2;
            scene.add(grid);

            const stlLoader = new THREE.STLLoader();
            const kinematicNodes = [];
            
            let assemblyGunGroup = new THREE.Group();
            let assemblyJigGroup = new THREE.Group();
            let jigInternalMeshGroup = new THREE.Group();
            
            assemblyJigGroup.add(jigInternalMeshGroup);
            scene.add(assemblyJigGroup);

            function base64ToBuffer(b64) {
                const bString = window.atob(b64);
                const len = bString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) { bytes[i] = bString.charCodeAt(i); }
                return bytes.buffer;
            }

            const defaultMaterials = [0x252525, 0xecb214, 0xecb214, 0xecb214, 0xecb214, 0xecb214, 0xecb214];

            let attachmentParent = scene;
            for(let i=0; i<7; i++) {
                const jointPivotNode = new THREE.Group();
                attachmentParent.add(jointPivotNode);
                kinematicNodes.push(jointPivotNode);

                if(coreData.robotMeshes && coreData.robotMeshes[i] && coreData.robotMeshes[i].length > 0) {
                    try {
                        const geo = stlLoader.parse(base64ToBuffer(coreData.robotMeshes[i]));
                        const mat = new THREE.MeshStandardMaterial({ color: defaultMaterials[i], roughness: 0.4, metalness: 0.2 });
                        jointPivotNode.add(new THREE.Mesh(geo, mat));
                    } catch(err) {
                        applyLocalCylinderFallback(i, jointPivotNode);
                    }
                } else {
                    applyLocalCylinderFallback(i, jointPivotNode);
                }
                attachmentParent = jointPivotNode;
            }

            kinematicNodes[6].add(assemblyGunGroup);

            function applyLocalCylinderFallback(idx, container) {
                let dLen = dims.L2;
                if(idx===0) dLen=dims.L0; if(idx===1) dLen=dims.L1; if(idx Sil==4 || idx===5) dLen=dims.L4;
                const geometry = new THREE.CylinderGeometry(0.12, 0.15, dLen, 16);
                if (idx === 0) { geometry.rotateX(Math.PI / 2); geometry.translate(0, 0, dLen / 2); }
                else if (idx === 1) { geometry.rotateY(Math.PI / 2); geometry.translate(dLen / 2, 0, 0); }
                else if (idx === 2) { geometry.rotateX(Math.PI / 2); geometry.translate(0, 0, dLen / 2); }
                else { geometry.rotateY(Math.PI / 2); geometry.translate(dLen / 2, 0, 0); }
                container.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: defaultMaterials[idx], roughness: 0.5 })));
            }

            if(coreData.gunMesh && coreData.gunMesh.length > 0) {
                const geo = stlLoader.parse(base64ToBuffer(coreData.gunMesh));
                geo.center(); geo.rotateY(Math.PI/2);
                const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.3 }));
                mesh.scale.set(0.001, 0.001, 0.001);
                mesh.rotation.y = coreData.gunConfig.rotZ;
                assemblyGunGroup.add(mesh);
            }

            // DYNAMIC SOLVER LAYER: Replaces the old hardcoded metrics with profile parameters
            function executeMatrixSolver(angles, configTCP) {
                kinematicNodes[0].position.set(0, 0, 0);
                kinematicNodes[0].rotation.set(0, 0, 0);

                kinematicNodes[1].position.set(0, 0, dims.L0);
                kinematicNodes[1].rotation.set(0, 0, angles[1]);

                kinematicNodes[2].position.set(dims.L1, 0, 0);
                kinematicNodes[2].rotation.set(0, angles[2], 0);

                kinematicNodes[3].position.set(0, 0, dims.L2);
                kinematicNodes[3].rotation.set(0, angles[3], 0);

                kinematicNodes[4].position.set(dims.L3, 0, dims.L3_Z);
                kinematicNodes[4].rotation.set(angles[4], 0, 0);

                kinematicNodes[5].position.set(dims.L4, 0, 0);
                kinematicNodes[5].rotation.set(0, angles[5], 0);

                kinematicNodes[6].position.set(dims.L5, 0, 0);
                kinematicNodes[6].rotation.set(angles[6], 0, 0);

                assemblyGunGroup.position.set(configTCP.offset, 0, 0);
            }

            function syncEnvironmentTransforms() {
                executeMatrixSolver(currentAngles, coreData.gunConfig);
                assemblyJigGroup.position.set(coreData.jigConfig.x, coreData.jigConfig.y, coreData.jigConfig.z);
                jigInternalMeshGroup.rotation.z = currentAngles[7]; 
            }

            function extractMatrixWorldCoordinates() {
                const computedSnapshot = [];
                renderer.render(scene, camera);
                for(let i=0; i<7; i++) {
                    const absPos = new THREE.Vector3(); const absRot = new THREE.Quaternion();
                    kinematicNodes[i].matrixWorld.decompose(absPos, absRot, new THREE.Vector3());
                    computedSnapshot.push({ pos: absPos.toArray(), quat: absRot.toArray() });
                }
                return computedSnapshot;
            }

            const UIHost = document.getElementById('axes-container');
            for(let i=1; i<=6; i++) {
                const el = document.createElement('div'); el.className = 'jog-axis-row';
                el.innerHTML = `
                    <button class="axis-btn" id="j-dec-${i}">-</button>
                    <div class="axis-label">A${i}</div>
                    <div class="axis-readout" id="readout-${i}">${(currentAngles[i] * (180/Math.PI)).toFixed(1)}°</div>
                    <button class="axis-btn" id="j-inc-${i}">+</button>
                `;
                UIHost.appendChild(el);
                document.getElementById(`j-dec-${i}`).addEventListener('click', () => pulseAxis(i, -1));
                document.getElementById(`j-inc-${i}`).addEventListener('click', () => pulseAxis(i, 1));
            }

            const E1Container = document.createElement('div'); E1Container.className = 'jog-axis-row';
            E1Container.style.cssText = 'margin-top:4px; border-top:1px solid #333; padding-top:4px;';
            E1Container.innerHTML = `
                <button class="axis-btn" id="e-dec">-</button>
                <div class="axis-label" style="color:#00e5ff;">E1</div>
                <div class="axis-readout" id="readout-7">${(currentAngles[7] * (180/Math.PI)).toFixed(1)}°</div>
                <button class="axis-btn" id="e-inc">+</button>
            `;
            UIHost.appendChild(E1Container);
            document.getElementById('e-dec').addEventListener('click', () => pulseAxis(7, -1));
            document.getElementById('e-inc').addEventListener('click', () => pulseAxis(7, 1));

            function pulseAxis(index, scalar) {
                if(isActivePlayback) return;
                currentAngles[index] += scalar * J_INCREMENT;
                document.getElementById(`readout-${index}`).innerText = `${(currentAngles[index] * (180/Math.PI)).toFixed(1)}°`;
                syncEnvironmentTransforms();
            }

            const hiddenInput = document.getElementById('jig-file-hidden');
            const dropzone = document.getElementById('jig-drop');
            dropzone.addEventListener('click', () => hiddenInput.click());
            hiddenInput.addEventListener('change', (e) => {
                if(e.target.files.length > 0) {
                    const reader = new FileReader();
                    reader.onload = function(evt) {
                        while(jigInternalMeshGroup.children.length > 0){ jigInternalMeshGroup.remove(jigInternalMeshGroup.children[0]); }
                        const parsedGeo = stlLoader.parse(evt.target.result); parsedGeo.center();
                        const targetMesh = new THREE.Mesh(parsedGeo, new THREE.MeshStandardMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.55 }));
                        targetMesh.scale.set(coreData.jigConfig.scale, coreData.jigConfig.scale, coreData.jigConfig.scale);
                        targetMesh.rotation.x = coreData.jigConfig.rotX; targetMesh.rotation.y = coreData.jigConfig.rotY;
                        jigInternalMeshGroup.add(targetMesh);
                        dropzone.innerText = "✓ LOADED";
                    };
                    reader.readAsArrayBuffer(e.target.files[0]);
                }
            });

            document.getElementById('sim-speed').addEventListener('input', (e) => {
                document.getElementById('speed-label').innerText = e.target.value + "%";
            });

            document.getElementById('btn-rec').addEventListener('click', () => {
                if(isActivePlayback) return;
                const matricesCoordinates = extractMatrixWorldCoordinates();
                memoryBuffer.push({ angles: [...currentAngles], transforms: matricesCoordinates });
                document.getElementById('steps-counter').innerText = "Steps: " + memoryBuffer.length;
                
                const targetUrl = new URL(window.parent.location.href);
                targetUrl.searchParams.set("event", "sync_sequence");
                targetUrl.searchParams.set("program_data", JSON.stringify(memoryBuffer));
                window.parent.history.replaceState({}, '', targetUrl.toString());
            });

            document.getElementById('btn-play').addEventListener('click', () => {
                if(memoryBuffer.length < 2) return;
                indexPointer = 0; lerpFactor = 0; isActivePlayback = true;
            });

            document.getElementById('btn-del').addEventListener('click', () => {
                memoryBuffer = []; document.getElementById('steps-counter').innerText = "Steps: 0";
                const targetUrl = new URL(window.parent.location.href);
                targetUrl.searchParams.set("event", "clear_sequence");
                window.parent.location.href = targetUrl.toString();
            });

            let indexPointer = 0, lerpFactor = 0, isActivePlayback = false;

            function runLoop() {
                requestAnimationFrame(runLoop);
                orbit.update();

                if(isActivePlayback && memoryBuffer.length >= 2) {
                    let activeFrame = memoryBuffer[indexPointer];
                    let futureFrame = memoryBuffer[indexPointer + 1];
                    let selectedVelocity = parseFloat(document.getElementById('sim-speed').value);
                    lerpFactor += (selectedVelocity / 100) * 0.04;

                    if(lerpFactor >= 1.0) {
                        lerpFactor = 0; indexPointer++;
                        if(indexPointer >= memoryBuffer.length - 1) { isActivePlayback = false; indexPointer = 0; }
                    }

                    let interpolatedAngles = [];
                    for(let i=0; i<8; i++) {
                        interpolatedAngles.push((1 - lerpFactor) * activeFrame.angles[i] + lerpFactor * futureFrame.angles[i]);
                    }
                    executeMatrixSolver(interpolatedAngles, coreData.gunConfig);
                    jigInternalMeshGroup.rotation.z = interpolatedAngles[7];
                }
                renderer.render(scene, camera);
            }

            window.addEventListener('resize', () => {
                camera.aspect = host.clientWidth / host.clientHeight; camera.updateProjectionMatrix();
                renderer.setSize(host.clientWidth, host.clientHeight);
            });

            document.getElementById('steps-counter').innerText = "Steps: " + memoryBuffer.length;
            syncEnvironmentTransforms();
            runLoop();
        </script>
    </body>
    </html>
    """.replace("__INJECTED_PAYLOAD__", json_payload)
    components.html(html_source, height=740, scrolling=False)

render_cloud_viewport(payload_data)
