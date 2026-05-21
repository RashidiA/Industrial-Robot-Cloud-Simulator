# Multi-Robot OLP Pro-Simulator 🤖⚡

An interactive, web-based Offline Programming (OLP) robotic simulator built with **Streamlit** and **Three.js (WebGL)**. This application provides real-time forward kinematics calculation, an intuitive virtual jog pendant, welding gun tooling offset adjustments, and rotatable fixture positioner configuration—all within a single responsive, wide-screen interface.

---

## 📺 Demonstration Video

Watch the simulation workflow in real-time, showcasing multi-robot profile loading, tool adjustments, and full horizontal workspace maximization:

<g-emoji class="g-emoji" alias="arrow_forward" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/25b6.png">▶️</g-emoji> **Click to preview execution setup below:**

https://github.com/user-attachments/assets/Simulasi Robot ABB Welding.mp4

> 💡 **Tip:** To embed your own recording on GitHub, simply edit this `README.md` on the GitHub website and **drag-and-drop** your `.mp4` or `.mov` file directly into the markdown code window. GitHub will host it securely and generate the play link automatically!

---

## 🌟 Key Features

* **Widescreen Mode with Minimizable Controls:** Utilizes the Streamlit sidebar for robot model selection, which can be collapsed/expanded with a single click to grant the WebGL viewport **100% full horizontal workspace**.
* **Multi-Robot Profiles:** Fully integrated configurations for industrial manipulators:
  * ABB IRB 6700
  * ABB IRB 4400
  * ABB IRB 6600
  * KUKA KR150
  * Yaskawa MH3500
* **Interactive WebGL Viewport:** High-performance 3D visualization using Three.js with robust camera orbit controls, ambient/directional lighting, and coordinate space locking to prevent robot base detachment or drifting.
* **Instant Jog Pendant:** On-the-fly joint control (`A1` through `A6`) with instantaneous forward kinematics recalculation and step trajectory recording.
* **Tooling & Fixture Alignment:** * Custom STL mesh upload support for the Welding Gun (TCP) and Rotary Positioner Jig.
  * Real-time adjustment sliders for tool offsets, gun twist, jig elevation levels, and spatial scale coordinates.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/multi-robot-olp-simulator.git](https://github.com/yourusername/multi-robot-olp-simulator.git)
cd multi-robot-olp-simulator

### 3. Install Required Dependencies
pip install streamlit numpy ikpy scipy

### 4. Project Directory Structure
multi-robot-olp-simulator/
│
├── app.py                 # Core application script
├── README.md              # Project documentation
├── temp/                  # Auto-generated runtime storage for uploaded gun/jig STLs
└── assets/
    ├── meshes/            # Fallback global STL meshes (base_link.stl, link_1.stl, etc.)
    └── robots/
        ├── ABB_6700/      # Profile-specific robotic link STL geometries
        ├── ABB_4400/
        ├── ABB_6600/
        ├── KUKA_KR150/
        └── Yaskawa_3500/

### 5. Running the Simulator
streamlit run app.py
