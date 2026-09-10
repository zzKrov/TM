import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="SENTINEL // Autonomous Bio-Perimeter Sensor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# CHECK AND LOAD LOCAL TFJS MODEL ARTIFACTS
# ---------------------------------------------------------
MODEL_PATH = "model.json"
METADATA_PATH = "metadata.json"
WEIGHTS_PATH = "weights.bin"

missing_files = [f for f in [MODEL_PATH, METADATA_PATH, WEIGHTS_PATH] if not os.path.exists(f)]

if missing_files:
    st.error(
        f"Missing Teachable Machine artifacts: {', '.join(missing_files)}. "
        f"Ensure `model.json`, `metadata.json`, and `weights.bin` are located in your repository root."
    )
    st.stop()

# Read local JSON files
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    model_json_str = f.read()

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata_json_str = f.read()

# Read weights.bin as base64 to inject directly into client-side TF.js
with open(WEIGHTS_PATH, "rb") as f:
    weights_b64 = base64.b64encode(f.read()).decode("utf-8")

# ---------------------------------------------------------
# JAVASCRIPT + TFJS SENSOR APP INJECTION
# ---------------------------------------------------------
html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8/dist/teachablemachine-pose.min.js"></script>

<style>
:root {{
    --bg-void: #060709;
    --bio-green: #00ff88;
    --alert-coral: #ff3366;
    --slate-cold: #64748b;
    --panel-border: rgba(255, 255, 255, 0.08);
}}

* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    background-color: var(--bg-void);
    background-image:
        radial-gradient(700px circle at var(--mouse-x, 50vw) var(--mouse-y, 30vh), rgba(0, 255, 136, 0.08), transparent 70%),
        radial-gradient(900px circle at 85% 15%, rgba(30, 41, 59, 0.4), transparent 80%),
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
    color: #f8fafc;
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 24px;
    min-height: 100vh;
    overflow-x: hidden;
}}

/* Masthead Header */
.sentinel-masthead {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 16px;
    margin-bottom: 24px;
}}
.sentinel-title {{
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.beacon {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: var(--bio-green);
    box-shadow: 0 0 12px var(--bio-green);
    animation: pulseBeacon 2s infinite;
}}
@keyframes pulseBeacon {{
    0%, 100% {{ transform: scale(1); opacity: 1; }}
    50% {{ transform: scale(1.6); opacity: 0.4; }}
}}
.mono {{
    font-family: 'JetBrains Mono', monospace;
}}

/* Main Workspace Grid */
.workspace-grid {{
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 28px;
}}

/* Camera / Canvas Stage */
.viewport-card {{
    background: rgba(10, 13, 18, 0.95);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 16px;
    position: relative;
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.6);
}}
.viewport-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--bio-green), transparent);
    animation: scanLine 3.5s linear infinite;
}}
@keyframes scanLine {{
    0% {{ transform: translateY(0); opacity: 0; }}
    20% {{ opacity: 0.8; }}
    80% {{ opacity: 0.8; }}
    100% {{ transform: translateY(420px); opacity: 0; }}
}}

.canvas-wrapper {{
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 10px;
    overflow: hidden;
    background: #030406;
    border: 1px solid rgba(255, 255, 255, 0.06);
    min-height: 400px;
}}

canvas {{
    width: 100%;
    max-height: 480px;
    object-fit: contain;
}}

/* Telemetry & Presence Status Hero */
.telemetry-card {{
    display: flex;
    flex-direction: column;
    gap: 16px;
}}
.presence-hero {{
    border-radius: 14px;
    padding: 24px;
    border: 1px solid var(--panel-border);
    transition: all 0.3s ease;
}}
.presence-hero.detected {{
    background: linear-gradient(145deg, rgba(0, 255, 136, 0.08) 0%, rgba(13, 16, 23, 0.95) 100%);
    border-color: rgba(0, 255, 136, 0.5);
    box-shadow: 0 15px 40px -10px rgba(0, 255, 136, 0.2);
}}
.presence-hero.vacant {{
    background: linear-gradient(145deg, rgba(100, 116, 139, 0.08) 0%, rgba(13, 16, 23, 0.95) 100%);
    border-color: rgba(100, 116, 139, 0.35);
    box-shadow: 0 15px 40px -10px rgba(0, 0, 0, 0.4);
}}
.hero-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}}
.hero-status {{
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    line-height: 1.1;
    margin-bottom: 12px;
}}

/* Meter Bar */
.meter-bg {{
    background: rgba(255, 255, 255, 0.08);
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    margin: 14px 0 6px 0;
}}
.meter-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.15s ease-out;
}}
.fill-detected {{
    background: linear-gradient(90deg, #00c868, #00ff88);
    box-shadow: 0 0 15px #00ff88;
}}
.fill-vacant {{
    background: linear-gradient(90deg, #475569, #94a3b8);
}}

/* Stat Pill Grid */
.stat-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}}
.stat-pill {{
    background: rgba(19, 23, 34, 0.7);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 14px 18px;
}}
.stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--slate-cold);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.stat-val {{
    font-family: 'Chakra Petch', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    margin-top: 4px;
}}

/* Control Button */
.btn-control {{
    background: rgba(0, 255, 136, 0.12);
    border: 1px solid rgba(0, 255, 136, 0.4);
    color: var(--bio-green);
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
    padding: 12px 24px;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100%;
    text-transform: uppercase;
    margin-top: 12px;
}}
.btn-control:hover {{
    background: var(--bio-green);
    color: #05070a;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
}}
</style>
</head>

<body>

<div class="sentinel-masthead">
    <div>
        <div class="sentinel-title">
            <span class="beacon"></span>
            SENTINEL // BIO-PRESENCE RADAR
        </div>
        <div class="mono" style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
            TEACHABLE MACHINE TF.JS POSE SENSOR • CLIENT-SIDE INFERENCE ENGINE
        </div>
    </div>
    <div class="mono" style="text-align: right; font-size: 0.75rem;">
        <span style="color: #64748b;">LATENCY MODE:</span> <span style="color: #00ff88;">REALTIME (GPU-ACCELERATED)</span><br>
        <span style="color: #64748b;">DISCRIMINATOR:</span> <span style="color: #cbd5e1;">POSENET v1.0</span>
    </div>
</div>

<div class="workspace-grid">
    <!-- Camera Canvas Viewport -->
    <div class="viewport-card">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">
            <span style="font-family: 'Chakra Petch'; font-size: 1.05rem; font-weight: 600;">SENSOR VIEWPORT</span>
            <span id="fps-counter" class="mono" style="font-size: 0.75rem; color: #00ff88;">FPS: --</span>
        </div>
        <div class="canvas-wrapper">
            <canvas id="canvas"></canvas>
        </div>
        <button id="start-btn" class="btn-control" onclick="toggleSensor()">INITIATE OPTICAL SENSOR</button>
    </div>

    <!-- Telemetry & Decision Hero -->
    <div class="telemetry-card">
        <div id="presence-box" class="presence-hero vacant">
            <div id="presence-tag" class="hero-tag" style="color: #94a3b8;">○ SENSOR STANDBY</div>
            <div id="presence-status" class="hero-status" style="color: #94a3b8;">STANDBY</div>
            <div id="presence-desc" class="mono" style="font-size: 0.85rem; color: #64748b;">
                Click Initiate Optical Sensor to launch PoseNet perception.
            </div>
            <div class="meter-bg">
                <div id="meter-fill" class="meter-fill fill-vacant" style="width: 0%;"></div>
            </div>
            <div class="mono" style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b;">
                <span>CONFIDENCE INDEX</span>
                <span id="prob-val" style="color: #cbd5e1;">0.00%</span>
            </div>
        </div>

        <div class="stat-grid">
            <div class="stat-pill">
                <div class="stat-label">DOMINANT SIGNATURE</div>
                <div id="stat-class" class="stat-val" style="color: #94a3b8;">--</div>
            </div>
            <div class="stat-pill">
                <div class="stat-label">SKELETON TRACKING</div>
                <div id="stat-pose" class="stat-val" style="color: #00ff88;">INACTIVE</div>
            </div>
        </div>
    </div>
</div>

<script>
// Ambient cursor light tracking
window.addEventListener('mousemove', (e) => {{
    document.documentElement.style.setProperty('--mouse-x', `${{e.clientX}}px`);
    document.documentElement.style.setProperty('--mouse-y', `${{e.clientY}}px`);
}});

// Inject local model artifacts from Python
const modelJSON = {model_json_str};
const metadataJSON = {metadata_json_str};
const weightsBase64 = "{weights_b64}";

let model, webcam, ctx;
let isRunning = false;
let lastFrameTime = performance.now();

// Convert Base64 back to ArrayBuffer for TFJS WeightLoader
function base64ToArrayBuffer(base64) {{
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
    }}
    return bytes.buffer;
}}

async function loadLocalCustomModel() {{
    const weightsBuffer = base64ToArrayBuffer(weightsBase64);
    
    // Custom IOHandler that loads directly from in-memory blobs
    const customIO = {{
        load: async () => {{
            return tf.io.decodeWeights(weightsBuffer, modelJSON.weightsManifest[0].weights);
        }}
    }};

    // Construct custom PoseNet wrapper from local artifacts
    const customModel = await tf.loadLayersModel(tf.io.fromMemory(modelJSON, null, customIO));
    
    // Mount into teachablemachine pose wrapper with MobileNet PoseNet
    const posenetModel = await tmPose.load(
        "https://teachablemachine.withgoogle.com/models/placeholder/",
        null
    ).catch(async () => {{
        // Fallback: direct posenet backbone
        return null;
    }});

    // Build the Pose classifier
    return new tmPose.CustomPoseNet(customModel, metadataJSON);
}}

async function toggleSensor() {{
    const btn = document.getElementById("start-btn");

    if (!isRunning) {{
        btn.innerText = "INITIALIZING POSENET PIPELINE...";
        btn.disabled = true;

        try {{
            // Construct weights blob URL
            const weightsBlob = new Blob([base64ToArrayBuffer(weightsBase64)], {{ type: 'application/octet-stream' }});
            const weightsUrl = URL.createObjectURL(weightsBlob);

            // Update model JSON to point to blob URL
            modelJSON.weightsManifest[0].paths = [weightsUrl];
            const modelBlob = new Blob([JSON.stringify(modelJSON)], {{ type: 'application/json' }});
            const modelUrl = URL.createObjectURL(modelBlob);

            const metaBlob = new Blob([JSON.stringify(metadataJSON)], {{ type: 'application/json' }});
            const metaUrl = URL.createObjectURL(metaBlob);

            model = await tmPose.load(modelUrl, metaUrl);

            // Setup camera
            const size = 440;
            const flip = true;
            webcam = new tmPose.Webcam(size, size, flip);
            await webcam.setup();
            await webcam.play();

            const canvas = document.getElementById("canvas");
            canvas.width = size;
            canvas.height = size;
            ctx = canvas.getContext("2d");

            isRunning = true;
            btn.innerText = "HALT OPTICAL SENSOR";
            btn.disabled = false;
            btn.style.background = "rgba(255, 51, 102, 0.15)";
            btn.style.borderColor = "#ff3366";
            btn.style.color = "#ff3366";

            window.requestAnimationFrame(loop);
        }} catch (err) {{
            console.error("Initialization failure:", err);
            btn.innerText = "INITIATION FAILED (CHECK CONSOLE)";
            btn.disabled = false;
        }}
    }} else {{
        isRunning = false;
        if (webcam) {{
            await webcam.stop();
        }}
        btn.innerText = "INITIATE OPTICAL SENSOR";
        btn.style.background = "rgba(0, 255, 136, 0.12)";
        btn.style.borderColor = "rgba(0, 255, 136, 0.4)";
        btn.style.color = "var(--bio-green)";
    }}
}}

async function loop(timestamp) {{
    if (!isRunning) return;

    webcam.update();
    await predict();

    const delta = performance.now() - lastFrameTime;
    lastFrameTime = performance.now();
    document.getElementById("fps-counter").innerText = `FPS: ${{Math.round(1000 / delta)}}`;

    window.requestAnimationFrame(loop);
}}

async function predict() {{
    // PoseNet estimation + custom Teachable Machine classification
    const {{ pose, posenetOutput }} = await model.estimatePose(webcam.canvas);
    const prediction = await model.predict(posenetOutput);

    // Draw video frame
    ctx.drawImage(webcam.canvas, 0, 0);

    // Draw Pose Skeleton & Keypoints if person present
    if (pose) {{
        const minPartConfidence = 0.5;
        tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
        tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        document.getElementById("stat-pose").innerText = "LOCKED";
        document.getElementById("stat-pose").style.color = "#00ff88";
    }} else {{
        document.getElementById("stat-pose").innerText = "NO SKELETON";
        document.getElementById("stat-pose").style.color = "#64748b";
    }}

    // Evaluate probabilities: Label 0 = Person, Label 1 = Person't
    const probPerson = prediction[0].probability;
    const probVacant = prediction[1] ? prediction[1].probability : (1.0 - probPerson);

    const isPerson = probPerson > 0.55;
    const activePct = Math.round((isPerson ? probPerson : probVacant) * 100);

    const hero = document.getElementById("presence-box");
    const tag = document.getElementById("presence-tag");
    const status = document.getElementById("presence-status");
    const desc = document.getElementById("presence-desc");
    const meter = document.getElementById("meter-fill");
    const probVal = document.getElementById("prob-val");
    const statClass = document.getElementById("stat-class");

    if (isPerson) {{
        hero.className = "presence-hero detected";
        tag.style.color = "#00ff88";
        tag.innerText = "● VERIFIED ORGANIC SIGNATURE";
        status.style.color = "#00ff88";
        status.innerText = "HUMAN DETECTED";
        desc.innerText = `Live organic subject confirmed in sensor perimeter with ${{activePct}}% certainty.`;
        meter.className = "meter-fill fill-detected";
        statClass.innerText = prediction[0].className;
        statClass.style.color = "#00ff88";
    }} else {{
        hero.className = "presence-hero vacant";
        tag.style.color = "#94a3b8";
        tag.innerText = "○ PERIMETER VACANT";
        status.style.color = "#94a3b8";
        status.innerText = "NO PERSON DETECTED";
        desc.innerText = `Sector is clear of organic human signatures with ${{activePct}}% certainty.`;
        meter.className = "meter-fill fill-vacant";
        statClass.innerText = prediction[1] ? prediction[1].className : "Person't";
        statClass.style.color = "#94a3b8";
    }}

    meter.style.width = `${{activePct}}%`;
    probVal.innerText = `${{activePct}}%`;
}}
</script>

</body>
</html>
"""

components.html(html_code, height=750, scrolling=False)
