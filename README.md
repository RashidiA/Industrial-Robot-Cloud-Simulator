# 📟 Industrial Robot Cloud Simulator & OLP Platform
[![Release](https://img.shields.io/badge/Release-v1.0.1-orange.svg)](https://github.com/RashidiA/Industrial-Robot-Cloud-Simulator)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Engine](https://img.shields.io/badge/Graphics-WebGL%20%2F%20Three.js-000000.svg)](https://threejs.org/)

An advanced, cloud-native Industrial Robot Offline Programming (OLP) simulator engineered for automotive manufacturing contexts (such as robotic resistance spot welding and material handling assembly lines). The application features multi-robot hardware registry, forward/inverse kinematics handling, tooling calibration matrices, and a cutting-edge, client-side **MediaPipe AI gesture tracking system** for real-time camera-driven teleoperation.

---

## 🚀 Key Features & System Architecture

* **Multi-Hardware Robot Profiles:** Built-in kinematic dimensions and joint constraints for standard heavy-payload industrial arms out-of-the-box (`ABB_6700`, `ABB_4400`, `ABB_6600`, `KUKA_KR150`, `Yaskawa_3500`).
* **Hybrid WebGL Inverse Kinematics:** Uses a localized client-side Cyclic Coordinate Descent (CCD) engine running in JavaScript inside the user's browser, enabling ultra-fast, smooth $60\text{ FPS}$ positional updates bypassing server lag.
* **CAD & Tooling Calibration Layer:** Interactive parameters to translate, scale, and rotate custom end-effectors (welding guns, grippers) or imported fixture CAD models (`.stl`) to match global plant floor coordinates.
* **AI Gesture-Controlled Teach Pendant:** Translates a user's real-world hand positions and orientations into live robotic commands via webcam tracking. Includes an automated "pinch tracking" trigger threshold to dynamically record sequential structural step locations.

---

## 📂 System Directory Layout

To run the simulator alongside its physical 3D mesh components, ensure your repository files are organized as follows:

```text
Industrial-Robot-Cloud-Simulator/
├── app.py                      # Main Python Streamlit Application
├── README.md                   # System Documentation
└── assets/
    ├── robots/                 # Robotic Arm CAD Components
    │   ├── ABB_6700/
    │   │   ├── base_link.stl
    │   │   ├── link_1.stl ... link_6.stl
    │   ├── ABB_4400/
    │   └── Yaskawa_3500/
    └── robot_tools/            # End-Effector Library Assets
        ├── welding_guns/
        ├── grippers/
        └── welding_torches/


⚙️ Installation & Local Implementation
1. Environment Setup
Clone the platform codebase and transition into the active environment:

git clone [https://github.com/RashidiA/Industrial-Robot-Cloud-Simulator.git](https://github.com/RashidiA/Industrial-Robot-Cloud-Simulator.git)
cd Industrial-Robot-Cloud-Simulator

2. Dependency Management
Install the core application packages via pip:

pip install streamlit numpy ikpy

3. Running the Server Instance
Execute the software directly from your local development console terminal:

streamlit run app.py

📖 Operational Guide: Hybrid Gesture Mode
The implementation couples Streamlit with interactive query data intercepts to route calculations around continuous page reruns.

Tracking Interlock Process

1. Selection: Open the application viewport sidebar and choose your target arm hardware and welding gun tool files.
2. Engagement: Switch the software interface pendant option to ✋ Gesture.
3. Initialization: Position your palm facing forward inside the highlighted orange boundary on your webcam view. Hold completely still for 2 seconds while the system automatically samples a deep structural baseline vector framework.
4. Teleoperation: Once the status banner shifts to green, move your hand horizontally or vertically. The WebGL model shifts its Tool Center Point (TCP) synchronously with your motion.
5. Timeline Recording: Bring your thumb and index finger together inside the capture frame. The system registers a "pinch event," triggers a status update notification flash, and appends the precise joint angle set directly back to your Streamlit programming script dataset.

🔒 Deployment & Plant Security Configuration

When publishing your container or setting it up within strict corporate IT intranets:

1. Webcam Authorization: Because all computer vision processes occur strictly inside the browser sandbox client-side via secure MediaPipe CDN networks, ensure that browser camera security settings grant access permissions to your exact application host domain address.
2. Session Persistence: Toggling the 🔴 RESET TOOL & JIG interface component instantly purges cached memory configurations, sweeps internal dynamic script histories, resets rotary positioning jig alignments back to $0^\circ$, and schedules clean runtime rendering loops.

🤝 Citation & Open Source License

Distributed under open-source project standards for automotive manufacturing simulation, design verification, and diagnostic research applications.
