# 📟 Cloud-Based Industrial Robot Simulator (Multi-Robot Station + AI Hand-Gesture Interface)

[![Release](https://img.shields.io/badge/Release-v2.0.0-orange.svg)](https://github.com/RashidiA/Industrial-Robot-Cloud-Simulator)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Engine](https://img.shields.io/badge/Graphics-WebGL%20%2F%20Three.js-000000.svg)](https://threejs.org/)

A cloud-native Industrial Robot Offline Programming (OLP) simulator for automotive manufacturing contexts (robotic spot welding, material handling, assembly stations). It has a **RoboDK-style interface**, supports a **full station of up to 4 robots**, lets you move robots, tools, jigs and fixtures directly in the 3D view, and includes a client-side **MediaPipe hand-gesture teach pendant**.

All simulation work (kinematics, inverse kinematics, animation, editing) runs **in your own browser** (laptop or phone). Streamlit only serves the page and prepares the CAD data, so the robots move smoothly without server reruns.

**Live simulator:** https://industrial-robot-cloud-simulator-54jggbftztagnuxosqtvzr.streamlit.app/

---

## 🚀 Key Features

### Robots & kinematics
* **Multi-hardware robot profiles:** `ABB_6700`, `ABB_4400`, `ABB_6600`, `KUKA_KR150`, `Yaskawa_3500`, plus any robot folder you add under `assets/robots/`. Each profile keeps its own link CAD, joint limits and kinematic dimensions.
* **Client-side forward & inverse kinematics:** a Cyclic Coordinate Descent (CCD) solver in JavaScript for fast, lag-free motion.
* **Up to 4 robots in one station.** Every robot has its own profile, tool and base position, and starts in its home pose.

### RoboDK-style 3D interface
* **Station tree** (left) with robots, tools, jig and uploaded objects. Click an item to select it.
* **Right-hand tabs:** **Jog** (sliders and ± buttons per joint, robot selector), **Program** (recorded points, export), **Layout** (exact numeric position/rotation), **Station** (sequence builder), **Learn**.
* **Hide the panels to maximise the cell view:** use **◧ Panel** (left), **Panel ◨** (right) or **⛶ Focus** (both) in the view toolbar, or press **B** / **N**. Your choice is remembered in the browser.
* View buttons (Iso etc.) centre the camera on the whole station.

### Move, rotate and match parts directly in 3D
* **Click to select** a robot, its tool, the jig or an uploaded object. An XYZ gizmo appears on the selected part.
* **Floating toolbar:** **Move / Rotate / Space** (world or local).
* **Right-click menu** on a selected part: move or rotate on all axes or on X / Y / Z only, switch space, deselect.
* **Easy tool-to-flange matching:** **Snap 90°**, **+90° X / Y / Z** and **Round** buttons snap the tool orientation to the robot's axis 6 flange. Jigs and objects rotate freely.
* **Movable robot base:** move or rotate a robot and its attached tool follows.
* **Numeric entry** in the Layout tab for exact positions, rotations and object scale.

### Fixtures, fences and other CAD
* Upload **the rotary positioning jig** (shared by all robots) and **up to 8 extra STL objects** (fixtures, fences, tables, etc.).
* Objects can be selected, moved, rotated and scaled. Units (mm or m) are detected automatically, and objects with "fence" in the name are shown transparent.
* Tools can come from the internal library (`welding_guns`, `grippers`, `welding_torches`) or from an external STL upload, per robot.

### Station sequences (multi-robot)
Build the cell cycle in the **Station** tab:
* Run steps **sequentially** or **in parallel**
* **Home** step, **E1 turn** of the shared positioner, **Wait**, **Loop**, and an **Auto** run
* One **Run** button executes the whole sequence

### Programming & export
* Record points per robot in the Program tab, run/stop playback, and **Export program** for all robots.

### AI gesture-controlled teach pendant
* Hand tracking through the webcam (MediaPipe, in the browser) moves the TCP in real time.
* A **pinch** (thumb + index finger) records the current joint angles as a program step.

### Persistence
* Station layout (robot bases, per-robot tool placement, jig, objects, sequence) is saved in the **browser's local storage**. Adding a robot 2 does not disturb the tool you already attached to robot 1.
* Settings in the setup panel are kept even when the panel is hidden.

---

## 📂 Repository Layout

```text
Industrial-Robot-Cloud-Simulator/
├── app.py                      # Streamlit application (UI + WebGL simulator)
├── README.md                   # Documentation
└── assets/
    ├── robots/                 # Robot CAD (one folder per robot profile)
    │   ├── ABB_6700/
    │   │   ├── base_link.stl
    │   │   └── link_1.stl ... link_6.stl
    │   ├── ABB_4400/
    │   └── Yaskawa_3500/
    └── robot_tools/            # End-effector library
        ├── welding_guns/
        ├── grippers/
        └── welding_torches/
```

Any new folder in `assets/robots/` with `base_link.stl` and `link_1.stl … link_6.stl` appears automatically in the robot profile list. Tool `.stl` files placed in the `robot_tools` sub-folders appear in the tool list.

---

## ⚙️ Installation (local)

```bash
git clone https://github.com/RashidiA/Industrial-Robot-Cloud-Simulator.git
cd Industrial-Robot-Cloud-Simulator
pip install streamlit numpy ikpy
streamlit run app.py
```

---

## 📖 Quick Start

1. **Simulation Setup (top of the page):**
   * Set **Number of robots** (1–4) and choose a **hardware profile** and **tool** for each robot in its tab.
   * Optionally upload a **jig** and extra **fixtures / fences** (STL).
   * When finished, switch **⚙️ Show Simulation Setup** off to hide the panel and enlarge the cell.
2. **Position the parts:** click a robot, tool, jig or object in the 3D view and use the gizmo, the right-click menu or the Layout tab. For a tool, use **Snap 90° / +90°** to match the flange.
3. **Jog and teach:** use the Jog tab to move joints, record points in the Program tab.
4. **Run the station:** build the sequence in the Station tab and press Run.
5. **Maximise the view:** press **⛶ Focus** (or **B** / **N**) to hide the side panels.

### Keyboard & mouse
| Action | How |
|---|---|
| Select a part | Left-click it in the 3D view or the station tree |
| Context menu | Right-click the selected part |
| Hide/show left panel | **B** or **◧ Panel** |
| Hide/show right panel | **N** or **Panel ◨** |
| Hide both panels | **⛶ Focus** |

### Hybrid gesture mode
1. Open the Jog panel and switch the pendant to **✋ Gesture**.
2. Allow camera access for the app's domain.
3. Hold your palm facing forward inside the orange boundary for 2 seconds while the baseline is sampled.
4. When the banner turns green, move your hand to move the TCP.
5. Pinch thumb and index finger to record a program step.

---

## 🔒 Deployment & Security Notes

1. **Webcam:** all vision processing runs in the browser (MediaPipe via CDN). Grant camera permission to your app's host domain. On strict intranets, allow the CDNs used by three.js and MediaPipe.
2. **Reset:** **🔴 RESET TOOLS & JIG** clears uploaded tool/jig/object files, program data and cached state, and reloads a clean setup.
3. **Browser storage:** the station layout is stored per browser. Clearing site data resets it.
4. **Uploads:** uploaded STL files are written to a temporary folder on the server for the current session only.

---

## 🤝 Citation & License

Distributed under open-source project standards for automotive manufacturing simulation, design verification, and teaching applications.
