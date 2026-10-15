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
# VERIFICACIÓN DE MODELOS TFJS LOCALES
# ---------------------------------------------------------
required_files = ["model.json", "metadata.json", "weights.bin"]
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"Faltan los siguientes archivos del modelo: {', '.join(missing)}. Colócalos en la raíz del proyecto.")
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
# COMPONENTE JAVASCRIPT POSE PIPELINE
# ---------------------------------------------------------
html_payload = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@600;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8/dist/teachablemachine-pose.min.js"></script>

    <style>
        :root {{
            --bg: #060709;
            --bio-green: #00ff88;
            --coral-alert: #ff3366;
            --slate: #64748b;
            --border: rgba(255, 255, 255, 0.08);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg);
            color: #f8fafc;
            font-family: 'Plus Jakarta Sans', sans-serif;
            padding: 20px;
            min-height: 100vh;
        }}

        .brand-signature {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            color: var(--bio-green);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}

        .title {{
            font-family: 'Chakra Petch', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .dot {{
            width: 10px;
            height: 10px;
            background: var(--bio-green);
            border-radius: 50%;
            box-shadow: 0 0 12px var(--bio-green);
        }}

        .mono {{
            font-family: 'JetBrains Mono', monospace;
        }}

        .layout {{
            display: grid;
            grid-template-columns: 1.25fr 1fr;
            gap: 24px;
        }}

        .viewport {{
            background: rgba(13, 16, 24, 0.9);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }}

        .canvas-box {{
            width: 100%;
            height: 380px;
            background: #000;
            border-radius: 8px;
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
            gap: 12px;
            margin-top: 14px;
        }}

        .btn {{
            width: 100%;
            padding: 13px;
            background: rgba(0, 255, 136, 0.12);
            border: 1px solid rgba(0, 255, 136, 0.4);
            color: var(--bio-green);
            font-family: 'Chakra Petch', sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.05em;
            border-radius: 8px;
            cursor: pointer;
            text-transform: uppercase;
            transition: 0.2s all;
            text-align: center;
        }}

        .btn:hover {{
            background: var(--bio-green);
            color: #000;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
        }}

        .btn-secondary {{
            background: rgba(100, 116, 139, 0.15);
            border-color: rgba(100, 116, 139, 0.4);
            color: #cbd5e1;
        }}
        .btn-secondary:hover {{
            background: #cbd5e1;
            color: #000;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        }}

        .status-hero {{
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--border);
            transition: 0.3s all;
            margin-bottom: 16px;
        }}

        .status-hero.detected {{
            background: linear-gradient(145deg, rgba(0, 255, 136, 0.08) 0%, rgba(13, 16, 24, 0.95) 100%);
            border-color: rgba(0, 255, 136, 0.5);
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.15);
        }}

        .status-hero.vacant {{
            background: linear-gradient(145deg, rgba(100, 116, 139, 0.08) 0%, rgba(13, 16, 24, 0.95) 100%);
            border-color: rgba(100, 116, 139, 0.35);
        }}

        .hero-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }}

        .hero-title {{
            font-family: 'Chakra Petch', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .meter {{
            background: rgba(255, 255, 255, 0.08);
            height: 10px;
            border-radius: 999px;
            overflow: hidden;
            margin: 16px 0 8px 0;
        }}

        .meter-bar {{
            height: 100%;
            border-radius: 999px;
            transition: width 0.15s ease;
        }}

        .bar-green {{
            background: linear-gradient(90deg, #00c868, #00ff88);
            box-shadow: 0 0 12px #00ff88;
        }}

        .bar-gray {{
            background: #64748b;
        }}

        .telemetry-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }}

        .telemetry-card {{
            background: rgba(13, 16, 24, 0.7);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
        }}

        #file-input {{
            display: none;
        }}
    </style>
</head>

<body>
    <div class="brand-signature">Krov Industries · Sensores y Visión</div>
    <div class="header">
        <div>
            <div class="title"><span class="dot"></span> Detector de Presencia y Postura</div>
            <div class="mono" style="font-size: 0.75rem; color: var(--slate); margin-top: 4px;">
                Clasificación de estados: {labels[0]} vs {labels[1]}
            </div>
        </div>
        <div class="mono" style="font-size: 0.75rem; text-align: right; color: var(--bio-green);">
            ● Sensor activo
        </div>
    </div>

    <div class="layout">
        <div class="viewport">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-family: 'Chakra Petch'; font-size: 0.95rem;">Área de captura óptica</span>
                <span id="fps" class="mono" style="font-size: 0.75rem; color: var(--bio-green);">FPS: --</span>
            </div>
            
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
                <div id="hero-tag" class="hero-tag" style="color: var(--slate);">○ En espera</div>
                <div id="hero-title" class="hero-title" style="color: var(--slate);">En espera</div>
                <div id="hero-desc" class="mono" style="font-size: 0.8rem; color: var(--slate);">
                    Activa la cámara o sube una imagen de prueba para evaluar la presencia.
                </div>
                <div class="meter">
                    <div id="meter-bar" class="meter-bar bar-gray" style="width: 0%;"></div>
                </div>
                <div class="mono" style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--slate);">
                    <span>Certeza de detección</span>
                    <span id="conf-txt">0.0%</span>
                </div>
            </div>

            <div class="telemetry-grid">
                <div class="telemetry-card">
                    <div class="mono" style="font-size: 0.7rem; color: var(--slate);">Estado clasificado</div>
                    <div id="stat-label" style="font-family: 'Chakra Petch'; font-size: 1.3rem; font-weight: 700; margin-top: 4px; color: var(--slate);">--</div>
                </div>
                <div class="telemetry-card">
                    <div class="mono" style="font-size: 0.7rem; color: var(--slate);">Seguimiento de postura</div>
                    <div id="stat-pose" style="font-family: 'Chakra Petch'; font-size: 1.3rem; font-weight: 700; margin-top: 4px; color: var(--slate);">Inactivo</div>
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
        let lastTime = performance.now();

        function base64ToUint8(b64) {{
            const bin = window.atob(b64);
            const len = bin.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {{
                bytes[i] = bin.charCodeAt(i);
            }}
            return bytes;
        }}

        async function ensureModelLoaded() {{
            if (model) return model;

            const desc = document.getElementById("hero-desc");
            desc.innerText = "Cargando modelo neuronal y pesos locales...";

            const modelFile = new File([JSON.stringify(modelJSON)], "model.json", {{ type: "application/json" }});
            const weightsFile = new File([base64ToUint8(weightsB64)], "weights.bin", {{ type: "application/octet-stream" }});
            const metadataFile = new File([JSON.stringify(metadataJSON)], "metadata.json", {{ type: "application/json" }});

            model = await tmPose.loadFromFiles(modelFile, weightsFile, metadataFile);
            return model;
        }}

        async function toggleWebcam() {{
            const btn = document.getElementById("btn");
            const desc = document.getElementById("hero-desc");

            if (!active) {{
                btn.innerText = "Iniciando sensor...";
                btn.disabled = true;

                try {{
                    await ensureModelLoaded();

                    const size = 380;
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
                    btn.style.color = "#ff3366";
                    btn.style.borderColor = "#ff3366";
                    btn.style.background = "rgba(255, 51, 102, 0.1)";

                    desc.innerText = "Cámara activa. Procesando puntos articulares...";
                    window.requestAnimationFrame(loop);
                }} catch (err) {{
                    console.error("Error al iniciar cámara:", err);
                    btn.disabled = false;
                    btn.innerText = "Iniciar cámara";
                    desc.style.color = "#ff3366";
                    desc.innerText = "Error: " + (err.message || err);
                }}
            }} else {{
                active = false;
                if (webcam) webcam.stop();
                btn.innerText = "Iniciar cámara";
                btn.style.color = "var(--bio-green)";
                btn.style.borderColor = "rgba(0, 255, 136, 0.4)";
                btn.style.background = "rgba(0, 255, 136, 0.12)";
                desc.style.color = "var(--slate)";
                desc.innerText = "Sensor óptico en reposo.";
            }}
        }}

        async function loop() {{
            if (!active) return;
            webcam.update();
            await evaluateFrame(webcam.canvas);

            const delta = performance.now() - lastTime;
            lastTime = performance.now();
            document.getElementById("fps").innerText = `FPS: ${{Math.round(1000 / delta)}}`;

            window.requestAnimationFrame(loop);
        }}

        async function handleFileUpload(event) {{
            const file = event.target.files[0];
            if (!file) return;

            if (active) toggleWebcam();

            const desc = document.getElementById("hero-desc");
            try {{
                await ensureModelLoaded();

                const img = new Image();
                img.onload = async () => {{
                    const canvas = document.getElementById("canvas");
                    canvas.width = 380;
                    canvas.height = 380;
                    ctx = canvas.getContext("2d");
                    ctx.drawImage(img, 0, 0, 380, 380);

                    desc.innerText = "Evaluando imagen cargada con el clasificador...";
                    await evaluateFrame(canvas);
                }};
                img.src = URL.createObjectURL(file);
            }} catch (err) {{
                desc.style.color = "#ff3366";
                desc.innerText = "Error al procesar la imagen: " + (err.message || err);
            }}
        }}

        async function evaluateFrame(inputSource) {{
            const {{ pose, posenetOutput }} = await model.estimatePose(inputSource);
            const prediction = await model.predict(posenetOutput);

            if (inputSource !== document.getElementById("canvas")) {{
                ctx.drawImage(inputSource, 0, 0);
            }}

            if (pose) {{
                tmPose.drawKeypoints(pose.keypoints, 0.5, ctx);
                tmPose.drawSkeleton(pose.keypoints, 0.5, ctx);
                document.getElementById("stat-pose").innerText = "Detectado";
                document.getElementById("stat-pose").style.color = "#00ff88";
            }} else {{
                document.getElementById("stat-pose").innerText = "Sin registro";
                document.getElementById("stat-pose").style.color = "#64748b";
            }}

            const personProb = prediction[0].probability;
            const absentProb = prediction[1] ? prediction[1].probability : (1.0 - personProb);
            const isPerson = personProb > 0.55;

            const displayScore = Math.round((isPerson ? personProb : absentProb) * 100);

            const heroBox = document.getElementById("hero-box");
            const heroTag = document.getElementById("hero-tag");
            const heroTitle = document.getElementById("hero-title");
            const heroDesc = document.getElementById("hero-desc");
            const meterBar = document.getElementById("meter-bar");
            const confTxt = document.getElementById("conf-txt");
            const statLabel = document.getElementById("stat-label");

            if (isPerson) {{
                heroBox.className = "status-hero detected";
                heroTag.style.color = "#00ff88";
                heroTag.innerText = "● Detección confirmada";
                heroTitle.style.color = "#00ff88";
                heroTitle.innerText = "Persona identificada";
                heroDesc.style.color = "#cbd5e1";
                heroDesc.innerText = `Presencia verificada con un ${{displayScore}}% de certeza.`;
                meterBar.className = "meter-bar bar-green";
                statLabel.innerText = "{labels[0]}";
                statLabel.style.color = "#00ff88";
            }} else {{
                heroBox.className = "status-hero vacant";
                heroTag.style.color = "#64748b";
                heroTag.innerText = "○ Zona despejada";
                heroTitle.style.color = "#64748b";
                heroTitle.innerText = "Sin presencia detectada";
                heroDesc.style.color = "#64748b";
                heroDesc.innerText = `Espacio libre verificado con un ${{displayScore}}% de certeza.`;
                meterBar.className = "meter-bar bar-gray";
                statLabel.innerText = "{labels[1]}";
                statLabel.style.color = "#64748b";
            }}

            meterBar.style.width = `${{displayScore}}%`;
            confTxt.innerText = `${{displayScore}}%`;
        }}
    </script>
</body>
</html>
"""

components.html(html_payload, height=660, scrolling=False)
