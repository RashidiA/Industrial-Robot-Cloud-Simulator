import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import os
import json
import base64

# --- 1. SYSTEM INITIALIZATION ---
st.set_page_config(page_title="Cloud Pendant Pro-Simulator", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

if 'j_angles' not in st.session_state:
    st.session_state.j_angles = [0.0] * 8 
if 'program' not in st.session_state:
    st.session_state.program = []

# --- 2. KINEMATIC COEFFICIENT PROFILES ---
ROBOT_DIMENSIONS = {
    "ABB_6700":   {"L0": 0.78,  "L1": 0.32, "L2": 1.28,  "L3": 1.142, "L3_Z": 0.2,  "L4": 0.2,  "L5": 0.2},
    "ABB_6600":   {"L0": 0.715, "L1": 0.32, "L2": 1.075, "L3": 1.142, "L3_Z": 0.2,  "L4": 0.2,  "L5": 0.2},
    "ABB_4400":   {"L0": 0.68,  "L1": 0.15, "L2": 0.88,  "L3": 0.78,  "L3_Z": 0.15, "L4": 0.15, "L5": 0.1},
    "KUKA_KR150": {"L0": 0.75,  "L1": 0.35, "L2": 1.25,  "L3": 1.10,  "L3_Z": 0.22, "L4": 0.21, "L5": 0.19}
}

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

# --- 3. HYBRID PARAMETER ROUTER ---
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
            st.error(f"Pendant data storage sync failure: {e}")
    elif event_type == "clear_sequence":
        st.session_state.program = []
        st.session_state.j_angles = [0.0] * 8
    st.query_params.clear()

# --- 4. OPERATOR SIDEBAR INTERFACE ---
with st.sidebar:
    st.title("📟 Teach Pendant Pro")

    with st.expander("🛠️ Layout Setup", expanded=True):
        if st.button("🔴 RESET GUN & JIG", use_container_width=True):
            for f in [os.path.join(TEMP_DIR, "gun.stl"), os.path.join(TEMP_DIR, "jig.stl")]:
                if os.path.exists(f): 
                    try: os.remove(f)
                    except: pass
            st.session_state.program = []
            st.session_state.j_angles = [0.0] * 8
            st.cache_data.clear()      
            st.query_params.clear()
            st.rerun()
            
        st.divider()
        selected_robot = st.selectbox(
            "Select Robot Library Profile", 
            options=list(ROBOT_DIMENSIONS.keys()), 
            index=0
        )
        
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

# --- 5. ENGINE VIRTUAL WEBGL CONTAINER ---
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
            
            #jog-pendant { position: absolute; top: 10px; right: 10px; background: rgba(20, 20, 20, 0.85); border: 1px solid #ff9800; border-radius: 6px; width: 180px; padding: 8px; color: white; z-index: 10; box-shadow: 0 4px 15px rgba(0,0,0,0.5); backdrop-filter: blur(3px); }
            .pendant-title { font-size: 10px; font-weight: bold; text-transform: uppercase; color: #ff9800; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 4px; margin-bottom: 8px; text-align: center; }
            .jog-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
            .jog-label { font-size: 11px; font-weight: bold; font-family: monospace; color: #bbb; }
            .jog-btn { background: #222; border: 1px solid #444; color: white; width: 36px; height: 22px; font-size: 13px; font-weight: bold; cursor: pointer; border-radius: 4px; }
            .jog-btn:active { background: #ff9800; color: black; border-color: #ff9800; }
            .val-display { font-family: monospace; font-size: 11px; color: #00ffcc; width: 55px; text-align: center; }
            
            .action-block { margin-top: 8px; border-top: 1px solid #333; padding-top: 8px; display: flex; flex-direction: column; gap: 4px; }
            .btn-action { width: 100%; border: none; font-weight: bold; height: 28px; border-radius: 4px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center; }
            
            #btn-save-step { background: #ff9800; color: black; }
            #btn-run-sim { background: #4caf50; color: white; }
            #btn-clear-seq { background: #f44336; color: white; }
            .step-counter { font-size: 11px; font-family: monospace; text-align: center; color: #aaa; margin-top: 2px; }
        </style>
    </head>
    <body>
        <div id="status">WebGL Processing...</div>
        
        <div id="jog-pendant">
            <div class="pendant-title">⚡ WEBGL DIRECT JOG</div>
            <div id="jog-rows-container"></div>
            <div class="action-block">
                <div class="jog-row" style="margin-bottom: 4px;">
                    <input type="range" id="sld-speed" min="5" max="100" value="50" step="5" style="flex-grow: 1; margin: 0 6px; accent-color: #ff9800;">
                    <div class="val-display" id="val-speed" style="width: 30px; color: #ff9800; font-weight: bold; font-size:10px;">50%</div>
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
            const dims = data.dimensions;

            let localJointAngles = [...data.initialAngles];
            let lastComputedTransforms = [];
            let embeddedTrajectory = [...data.trajectory];
            
            const J_STEP = 5 * (Math.PI / 180);

            THREE.Object3D.DefaultUp.set(0, 0, 1);
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x141414);

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

            const linkColors = [0x222222, 0xecb214, 0xecb214, 0xecb214, 0xecb214, 0xecb214, 0xecb214];

            // PARSE SOLID MESHES OR COMPUTE THE INDEPENDENT CYLINDRICAL BACKBONE STATIONS
            for(let i=0; i<7; i++) {
                let mesh;
                if(data.linkGeometries && data.linkGeometries[i] && data.linkGeometries[i].length > 0) {
                    try {
                        const geometry = loader.parse(base64ToArrayBuffer(data.linkGeometries[i]));
                        const material = new THREE.MeshStandardMaterial({ color: linkColors[i], roughness: 0.4 });
                        mesh = new THREE.Mesh(geometry, material);
                    } catch(err) {
                        mesh = generateFallbackMesh(i);
                    }
                } else {
                    mesh = generateFallbackMesh(i);
                }
                scene.add(mesh);
                links.push(mesh);
            }

            function generateFallbackMesh(idx) {
                let dLen = 0.4;
                if(idx===0) dLen = dims.L0; 
                if(idx===1) dLen = dims.L1; 
                if(idx===2) dLen = dims.L2; 
                if(idx===3) dLen = dims.L3;
                if(idx===4) dLen = dims.L4; 
                if(idx===5) dLen = dims.L5;

                const geometry = new THREE.CylinderGeometry(0.14, 0.18, dLen, 24);
                geometry.rotateX(Math.PI / 2); 
                if(idx === 1 || idx === 2 || idx === 3) { geometry.translate(0, 0, dLen / 2); }
                return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: linkColors[idx], roughness: 0.4 }));
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

            // FLAT COORDINATE MATHEMATICAL MATRIX TRANSFORMATIONS
            function computeForwardKinematics(angles) {
                const computedTransforms = [];
                let currentMatrix = new THREE.Matrix4();
                
                computedTransforms.push({
                    pos: new THREE.Vector3(0,0,0).toArray(),
                    quat: new THREE.Quaternion().toArray()
                });

                let m1 = new THREE.Matrix4().makeTranslation(0, 0, dims.L0);
                m1.multiply(new THREE.Matrix4().makeRotationZ(angles[1]));
                currentMatrix.multiply(m1);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m2 = new THREE.Matrix4().makeTranslation(dims.L1, 0, 0);
                m2.multiply(new THREE.Matrix4().makeRotationY(angles[2]));
                currentMatrix.multiply(m2);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m3 = new THREE.Matrix4().makeTranslation(0, 0, dims.L2);
                m3.multiply(new THREE.Matrix4().makeRotationY(angles[3]));
                currentMatrix.multiply(m3);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m4 = new THREE.Matrix4().makeTranslation(dims.L3, 0, dims.L3_Z);
                m4.multiply(new THREE.Matrix4().makeRotationX(angles[4]));
                currentMatrix.multiply(m4);
                
                let correctionMatrix = currentMatrix.clone();
                let directionVector = new THREE.Vector3(1, 0, 0).applyQuaternion(new THREE.Quaternion().setFromRotationMatrix(correctionMatrix));
                let fixedPos = new THREE.Vector3().setFromMatrixPosition(correctionMatrix).add(directionVector.multiplyScalar(-1.0));
                
                computedTransforms.push({
                    pos: fixedPos.toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m5 = new THREE.Matrix4().makeTranslation(dims.L4, 0, 0);
                m5.multiply(new THREE.Matrix4().makeRotationY(angles[5]));
                currentMatrix.multiply(m5);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                let m6 = new THREE.Matrix4().makeTranslation(dims.L5, 0, 0);
                m6.multiply(new THREE.Matrix4().makeRotationX(angles[6]));
                currentMatrix.multiply(m6);
                computedTransforms.push({
                    pos: new THREE.Vector3().setFromMatrixPosition(currentMatrix).toArray(),
                    quat: new THREE.Quaternion().setFromRotationMatrix(currentMatrix).toArray()
                });

                return computedTransforms;
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
                    gunMesh.translateX(gunOffset);
                }
                jigMesh.position.set(jigX, jigY, jigZ);
                internalJigContent.rotation.z = e1RotAngle;
            }

            const rowsContainer = document.getElementById('jog-rows-container');
            for(let i=1; i<=6; i++) {
                const row = document.createElement('div');
                row.className = 'jog-row';
                row.innerHTML = `
                    <button class="jog-btn" id="btn-m-${i}">-</button>
                    <div class="jog-label">A${i}</div>
                    <div class="val-display" id="val-${i}">${(localJointAngles[i] * (180 / Math.PI)).toFixed(1)}°</div>
                    <button class="jog-btn" id="btn-p-${i}">+</button>
                `;
                rowsContainer.appendChild(row);
                
                (function(idx) {
                    document.getElementById(`btn-m-${idx}`).addEventListener('click', () => jogJoint(idx, -1));
                    document.getElementById(`btn-p-${idx}`).addEventListener('click', () => jogJoint(idx, 1));
                })(i);
            }
            
            const e1Row = document.createElement('div');
            e1Row.className = 'jog-row';
            e1Row.style.marginTop = '6px';
            e1Row.style.borderTop = '1px solid #333';
            e1Row.style.paddingTop = '4px';
            e1Row.innerHTML = `
                <button class="jog-btn" id="btn-m-7">-</button>
                <div class="jog-label" style="color:#ff9800;">E1</div>
                <div class="val-display" id="val-7">${(localJointAngles[7] * (180 / Math.PI)).toFixed(1)}°</div>
                <button class="jog-btn" id="btn-p-7">+</button>
            `;
            rowsContainer.appendChild(e1Row);
            document.getElementById('btn-m-7').addEventListener('click', () => jogJoint(7, -1));
            document.getElementById('btn-p-7').addEventListener('click', () => jogJoint(7, 1));

            function jogJoint(jointIdx, direction) {
                if(runSimulation) return; 
                localJointAngles[jointIdx] += direction * J_STEP;
                document.getElementById(`val-${jointIdx}`).innerText = `${(localJointAngles[jointIdx] * (180 / Math.PI)).toFixed(1)}°`;
                lastComputedTransforms = computeForwardKinematics(localJointAngles);
                updateSceneTransforms(lastComputedTransforms, data.gunOffset, data.jigX, data.jigY, data.jigZ, localJointAngles[7]);
            }

            function updateUIElements() {
                document.getElementById('lbl-steps').innerText = "Steps: " + embeddedTrajectory.length;
            }
            updateUIElements();

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
                updateUIElements();
                
                const targetUrl = new URL(window.parent.location.href);
                targetUrl.searchParams.set("event", "sync_sequence");
                targetUrl.searchParams.set("program_data", JSON.stringify(embeddedTrajectory));
                window.parent.history.replaceState({}, '', targetUrl.toString());
            });

            document.getElementById('btn-run-sim').addEventListener('click', () => {
                if(embeddedTrajectory.length < 2) {
                    alert("Please record at least 2 structural step points to interpolate trajectory paths.");
                    return;
                }
                simStepIndex = 0;
                interpolationFraction = 0;
                runSimulation = true;
            });

            document.getElementById('btn-clear-seq').addEventListener('click', () => {
                embeddedTrajectory = [];
                updateUIElements();
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
            animate();
        </script>
    </body>
    </html>
    """.replace("__PAYLOAD_OBJECT__", json_stream)
    
    components.html(html_source, height=750, scrolling=False)

# --- 6. EXECUTION ENGINE HARDWARE REBINDING LAYER ---
path_gun = os.path.join(TEMP_DIR, "gun.stl")
path_jig = os.path.join(TEMP_DIR, "jig.stl")

chosen_dimensions = ROBOT_DIMENSIONS[selected_robot]

link_b64s = []
for i in range(7):
    specific_folder_path = os.path.join(BASE_DIR, "assets", "meshes", selected_robot)
    name = f"link_{i}.stl" if i > 0 else "base_link.stl"
    
    if os.path.exists(os.path.join(specific_folder_path, name)):
        target_file = os.path.join(specific_folder_path, name)
    else:
        target_file = os.path.join(BASE_DIR, "assets", "meshes", name)
        
    link_b64s.append(get_file_base64_cached(target_file))

scene_payload = {
    "trajectory": st.session_state.program,
    "initialAngles": st.session_state.j_angles,
    "linkGeometries": link_b64s,
    "dimensions": chosen_dimensions,
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
