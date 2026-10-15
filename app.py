import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Detector de Presencia // Krov Industries",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# LÍNEA BASE VISUAL: GRADIENTES ESMERALDA Y CIBERNÉTICOS
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #060709 !important;
    color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    overflow-x: hidden;
}

[data-testid="stHeader"] { background: transparent !important; }

/* 1. Gradiente animado continuo */
.stApp {
    background: linear-gradient(135deg, #060709 0%, #081410 30%, #0d1e18 60%, #05090a 100%) !important;
    background-size: 300% 300% !important;
    animation: matrixShift 18s ease infinite alternate !important;
}

@keyframes matrixShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 2. Fluidos orgánicos bio-luminiscentes */
.fluid-matrix-1 {
    position: fixed;
    top: 15%;
    left: -120px;
    width: 520px;
    height: 520px;
    background: radial-gradient(circle, rgba(0, 255, 136, 0.12) 0%, rgba(6, 78, 59, 0.04) 60%, transparent 75%);
    border-radius: 40% 60% 70% 30% / 50% 60% 40% 50%;
    filter: blur(85px);
    animation: matrixDrift1 22s infinite alternate ease-in-out;
    pointer-events: none;
    z-index: 0;
}

.fluid-matrix-2 {
    position: fixed;
    bottom: 5%;
    right: -100px;
    width: 560px;
    height: 560px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, rgba(0, 255, 136, 0.03) 60%, transparent 75%);
    border-radius: 60% 40% 50% 50% / 40% 50% 60% 50%;
    filter: blur(90px);
    animation: matrixDrift2 25s infinite alternate ease-in-out;
    pointer-events: none;
    z-index: 0;
}

@keyframes matrixDrift1 {
    0% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(120px, 70px) scale(1.15); }
    100% { transform: translate(40px, 140px) scale(0.95); }
}

@keyframes matrixDrift2 {
    0% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(-100px, -70px) scale(1.2); }
    100% { transform: translate(-30px, -120px) scale(0.9); }
}
</style>

<div class="fluid-matrix-1"></div>
<div class="fluid-matrix-2"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# VERIFICACIÓN DE MODELOS TFJS LOCALES
# ---------------------------------------------------------
required_files = ["model.json", "metadata.json", "weights.bin"]
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"Archivos de modelo faltantes: {', '.join(missing)}. Colócalos en el directorio raíz.")
    st.stop()

with open("model.json", "r", encoding="utf-8") as f:
    model_json_data = json.load(f)

if "weightsManifest" in model_json_data and len(model_json_data["weightsManifest"]) > 0:
    model_json_data["weightsManifest"][0]["paths"] = ["weights.bin"]

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata_json_data = json.load(f)

with open("weights.bin", "rb") as f:
    weights_b64 = base64.b64encode(f.read()).decode("utf-8")

labels = metadata_json_data.get("labels", ["Persona", "Sin presencia"])

# ---------------------------------------------------------
# COMPONENTE JAVASCRIPT: DETECCIÓN FACIAL Y CORPORAL EXACTA
# ---------------------------------------------------------
html_payload = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8/dist/teachablemachine-pose.min.js"></script>

    <style>
        :root {{
            --bg: transparent;
            --bio-green: #00ff88;
            --coral-alert: #ef4444;
            --slate: #64748b;
            --border: rgba(255, 255, 255, 0.08);
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            background-color: var(--bg);
            color: #f8fafc;
            font-family: 'Plus Jakarta Sans', sans-serif;
            padding: 10px;
            min-height: 100vh;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}

        .brand-signature {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            color: var(--slate);
            text-transform: uppercase;
        }}

        .title {{
            font-family: 'Chakra Petch', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
        }}

        .layout {{
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 20px;
        }}

        .viewport {{
            background: rgba(13, 16, 24, 0.9);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        }}

        .canvas-box {{
            width: 100%;
            height: 360px;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.06);
            position: relative;
        }}

        canvas {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .btn-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 12px;
        }}

        .btn {{
            width: 100%;
            padding: 12px;
            background: rgba(0, 255, 136, 0.12);
            border: 1px solid rgba(0, 255, 136, 0.4);
            color: var(--bio-green);
            font-family: 'Chakra Petch', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.05em;
            border-radius: 6px;
            cursor: pointer;
            text-transform: uppercase;
            transition: 0.2s all;
            text-align: center;
        }}

        .btn:hover {{
            background: var(--bio-green);
            color: #000;
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);
        }}

        .btn-secondary {{
            background: rgba(100, 116, 139, 0.15);
            border-color: rgba(100, 116, 139, 0.4);
            color: #cbd5e1;
        }}
        .btn-secondary:hover {{
            background: #cbd5e1;
            color: #000;
        }}

        .status-hero {{
            border-radius: 10px;
            padding: 20px;
            border: 1px solid var(--border);
            transition: 0.25s all;
            margin-bottom: 14px;
        }}

        .status-hero.detected {{
            background: linear-gradient(145deg, rgba(0, 255, 136, 0.12) 0%, rgba(13, 16, 24, 0.95) 100%);
            border-color: rgba(0, 255, 136, 0.6);
            box-shadow: 0 8px 30px rgba(0, 255, 136, 0.2);
        }}

        .status-hero.vacant {{
            background: linear-gradient(145deg, rgba(239, 68, 68, 0.12) 0%, rgba(13, 16, 24, 0.95) 100%);
            border-color: rgba(239, 68, 68, 0.6);
            box-shadow: 0 8px 30px rgba(239, 68, 68, 0.2);
        }}

        .count-badge {{
            font-family: 'Chakra Petch', sans-serif;
            font-size: 3.4rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 4px;
        }}

        .meter {{
            background: rgba(255, 255, 255, 0.08);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 12px 0 6px 0;
        }}

        .meter-bar {{
            height: 100%;
            transition: width 0.15s ease, background-color 0.2s ease;
        }}

        .bar-green {{
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }}

        .bar-red {{
            background: #ef4444;
            box-shadow: 0 0 10px #ef4444;
        }}

        .telemetry-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}

        .telemetry-card {{
            background: rgba(13, 16, 24, 0.7);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
        }}

        #file-input {{ display: none; }}
    </style>
</head>

<body>
    <div class="header">
        <div>
            <div class="brand-signature">Krov Industries · Sensores Ópticos</div>
            <div class="title">Detector de Presencia y Postura</div>
        </div>
        <div id="status-pill" style="font-family:'JetBrains Mono'; font-size:0.75rem; color:var(--slate);">
            ○ SENSOR EN ESPERA
        </div>
    </div>

    <div class="layout">
        <div class="viewport">
            <div class="canvas-box">
                <canvas id="canvas"></canvas>
            </div>

            <div class="btn-row">
                <button id="btn" class="btn" onclick="toggleWebcam()">Iniciar cámara</button>
                <label class="btn btn-secondary" for="file-input">Subir imagen</label>
                <input type="file" id="file-input" accept="image/*" onchange="handleFileUpload(event)">
            </div>
        </div>

        <div>
            <div id="hero-box" class="status-hero vacant">
                <div style="font-family:'JetBrains Mono'; font-size:0.75rem; letter-spacing:0.1em; color:#ef4444;" id="hero-tag">
                    ● CONTEO CONFIRMADO
                </div>
                <div id="count-display" class="count-badge" style="color:#ef4444;">0</div>
                <div id="hero-title" style="font-family:'Chakra Petch'; font-size:1.4rem; font-weight:700; color:#ef4444; margin-bottom:4px;">
                    SIN PERSONA DETECTADA
                </div>
                <div id="hero-desc" style="font-size:0.82rem; color:#94a3b8;">
                    No se registra rostro ni presencia humana válida en el encuadre óptico.
                </div>
                <div class="meter">
                    <div id="meter-bar" class="meter-bar bar-red" style="width: 0%;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; font-family:'JetBrains Mono'; color:#64748b;">
                    <span>CONFIANZA DE SEÑAL</span>
                    <span id="conf-txt">0.0%</span>
                </div>
            </div>

            <div class="telemetry-grid">
                <div class="telemetry-card">
                    <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:var(--slate);">CANAL FACIAL</div>
                    <div id="stat-face" style="font-family:'Chakra Petch'; font-size:1.1rem; font-weight:700; color:#ef4444; margin-top:2px;">
                        NO DETECTADO
                    </div>
                </div>
                <div class="telemetry-card">
                    <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:var(--slate);">ESTRUCTURA ÓSEA</div>
                    <div id="stat-pose" style="font-family:'Chakra Petch'; font-size:1.1rem; font-weight:700; color:#64748b; margin-top:2px;">
                        INACTIVA
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const modelJSON = {json.dumps(model_json_data)};
        const metadataJSON = {json.dumps(metadata_json_data)};
        const weightsB64 = "{weights_b64}";

        let model = null;
        let webcam = null;
        let ctx = null;
        let active = false;
        let consecutiveFrames = 0;
        const MIN_CONFIRM_FRAMES = 2;

        function base64ToUint8(b64) {{
            const bin = window.atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            return bytes;
        }}

        async function ensureModelLoaded() {{
            if (model) return model;
            const modelFile = new File([JSON.stringify(modelJSON)], "model.json", {{ type: "application/json" }});
            const weightsFile = new File([base64ToUint8(weightsB64)], "weights.bin", {{ type: "application/octet-stream" }});
            const metadataFile = new File([JSON.stringify(metadataJSON)], "metadata.json", {{ type: "application/json" }});
            model = await tmPose.loadFromFiles(modelFile, weightsFile, metadataFile);
            return model;
        }}

        async function toggleWebcam() {{
            const btn = document.getElementById("btn");
            if (!active) {{
                btn.innerText = "Iniciando sensor...";
                btn.disabled = true;
                try {{
                    await ensureModelLoaded();
                    const size = 360;
                    webcam = new tmPose.Webcam(size, size, true);
                    await webcam.setup();
                    await webcam.play();

                    const canvas = document.getElementById("canvas");
                    canvas.width = size;
                    canvas.height = size;
                    ctx = canvas.getContext("2d");

                    active = true;
                    btn.disabled = false;
                    btn.innerText = "Detener cámara";
                    btn.style.color = "#ef4444";
                    btn.style.borderColor = "#ef4444";
                    btn.style.background = "rgba(239, 68, 68, 0.12)";
                    document.getElementById("status-pill").innerText = "● SENSOR ACTIVO";
                    document.getElementById("status-pill").style.color = "#00ff88";

                    window.requestAnimationFrame(loop);
                }} catch (err) {{
                    btn.disabled = false;
                    btn.innerText = "Iniciar cámara";
                    alert("Error al iniciar cámara: " + err.message);
                }}
            }} else {{
                active = false;
                if (webcam) webcam.stop();
                btn.innerText = "Iniciar cámara";
                btn.style.color = "var(--bio-green)";
                btn.style.borderColor = "rgba(0, 255, 136, 0.4)";
                btn.style.background = "rgba(0, 255, 136, 0.12)";
                document.getElementById("status-pill").innerText = "○ SENSOR EN ESPERA";
                document.getElementById("status-pill").style.color = "var(--slate)";
                setNoPersonState(0);
            }}
        }}

        async function loop() {{
            if (!active) return;
            webcam.update();
            await evaluateFrame(webcam.canvas);
            window.requestAnimationFrame(loop);
        }}

        async function handleFileUpload(event) {{
            const file = event.target.files[0];
            if (!file) return;
            if (active) toggleWebcam();

            await ensureModelLoaded();
            const img = new Image();
            img.onload = async () => {{
                const canvas = document.getElementById("canvas");
                canvas.width = 360;
                canvas.height = 360;
                ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0, 360, 360);
                await evaluateFrame(canvas);
            }};
            img.src = URL.createObjectURL(file);
        }}

        function setNoPersonState(conf) {{
            document.getElementById("hero-box").className = "status-hero vacant";
            document.getElementById("count-display").innerText = "0";
            document.getElementById("count-display").style.color = "#ef4444";
            document.getElementById("hero-tag").style.color = "#ef4444";
            document.getElementById("hero-title").innerText = "SIN PERSONA DETECTADA";
            document.getElementById("hero-title").style.color = "#ef4444";
            document.getElementById("hero-desc").innerText = "No se registra presencia humana ni rostro válido en el encuadre.";
            document.getElementById("meter-bar").className = "meter-bar bar-red";
            document.getElementById("meter-bar").style.width = `${{conf}}%`;
            document.getElementById("conf-txt").innerText = `${{conf}}%`;
            document.getElementById("stat-face").innerText = "NO DETECTADO";
            document.getElementById("stat-face").style.color = "#ef4444";
            document.getElementById("stat-pose").innerText = "INACTIVA";
            document.getElementById("stat-pose").style.color = "#64748b";
        }}

        async function evaluateFrame(inputSource) {{
            const {{ pose, posenetOutput }} = await model.estimatePose(inputSource);
            const prediction = await model.predict(posenetOutput);

            if (inputSource !== document.getElementById("canvas")) {{
                ctx.drawImage(inputSource, 0, 0);
            }}

            // EXTRACCIÓN DE PUNTOS FACIALES (0: Nariz, 1: Ojo Izq, 2: Ojo Der, 3: Oreja Izq, 4: Oreja Der)
            let faceDetected = false;
            let faceCenterX = 0, faceCenterY = 0, validFacePts = 0;

            if (pose && pose.keypoints) {{
                for (let i = 0; i <= 4; i++) {{
                    const kp = pose.keypoints[i];
                    if (kp && kp.score > 0.38) {{
                        faceDetected = true;
                        faceCenterX += kp.position.x;
                        faceCenterY += kp.position.y;
                        validFacePts++;
                    }}
                }}
            }}

            const hasValidPose = pose && pose.score > 0.42;
            const personProb = prediction[0] ? prediction[0].probability : 0;
            const rawScore = Math.round(personProb * 100);

            // Se confirma presencia si hay postura corporal VÁLIDA O rostro VÁLIDO
            const confirmedInFrame = (faceDetected && validFacePts >= 2) || (hasValidPose && personProb > 0.60);

            if (confirmedInFrame) {{
                consecutiveFrames++;
            }} else {{
                consecutiveFrames = 0;
            }}

            if (consecutiveFrames >= MIN_CONFIRM_FRAMES) {{
                // Dibujar esqueleto si está presente
                if (pose) {{
                    tmPose.drawKeypoints(pose.keypoints, 0.5, ctx);
                    tmPose.drawSkeleton(pose.keypoints, 0.5, ctx);
                }}

                // Dibujar anillo facial luminoso si se identificó rostro
                if (faceDetected && validFacePts > 0) {{
                    const cx = faceCenterX / validFacePts;
                    const cy = faceCenterY / validFacePts;
                    ctx.beginPath();
                    ctx.arc(cx, cy, 32, 0, Math.PI * 2);
                    ctx.strokeStyle = '#00ff88';
                    ctx.lineWidth = 2.5;
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = '#00ff88';
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                }}

                document.getElementById("hero-box").className = "status-hero detected";
                document.getElementById("count-display").innerText = "1";
                document.getElementById("count-display").style.color = "#00ff88";
                document.getElementById("hero-tag").style.color = "#00ff88";
                document.getElementById("hero-title").innerText = "PERSONA DETECTADA";
                document.getElementById("hero-title").style.color = "#00ff88";
                document.getElementById("hero-desc").innerText = faceDetected 
                    ? "Presencia confirmada (Rostro detectado en encuadre)."
                    : "Presencia humana confirmada mediante puntos articulares corporales.";
                document.getElementById("meter-bar").className = "meter-bar bar-green";
                document.getElementById("meter-bar").style.width = `${{Math.max(rawScore, 75)}}%`;
                document.getElementById("conf-txt").innerText = `${{Math.max(rawScore, 75)}}%`;
                document.getElementById("stat-face").innerText = faceDetected ? "CONFIRMADO" : "FUERA DE ENCUADRE";
                document.getElementById("stat-face").style.color = faceDetected ? "#00ff88" : "#94a3b8";
                document.getElementById("stat-pose").innerText = hasValidPose ? "ACTIVA" : "PARCIAL";
                document.getElementById("stat-pose").style.color = "#00ff88";
            }} else {{
                setNoPersonState(rawScore < 50 ? 0 : rawScore);
            }}
        }}
    </script>
</body>
</html>
"""

components.html(html_payload, height=640, scrolling=False)
