import io
import json
import time
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# SENSOR SUITE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SENTINEL // Autonomous Presence Sensor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# INTERACTIVE RADAR & MOUSE FOLLOWER
# ---------------------------------------------------------
components.html(
    """
    <script>
    const pDoc = window.parent.document;
    const root = pDoc.documentElement;

    // Track mouse coordinates for dynamic radar illumination
    pDoc.addEventListener('mousemove', (e) => {
        root.style.setProperty('--mouse-x', `${e.clientX}px`);
        root.style.setProperty('--mouse-y', `${e.clientY}px`);
    });
    </script>
    """,
    height=0,
    width=0
)

# ---------------------------------------------------------
# BIO-PERIMETER STYLESHEET (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    :root {
        --bg-void: #060709;
        --panel-bg: rgba(13, 16, 23, 0.82);
        --panel-border: rgba(255, 255, 255, 0.08);
        --bio-green: #00ff88;
        --bio-green-dim: rgba(0, 255, 136, 0.15);
        --alert-coral: #ff3366;
        --alert-dim: rgba(255, 51, 102, 0.15);
        --slate-cold: #64748b;
        --text-pure: #f8fafc;
    }

    /* Interactive Sonar/Radar Canvas */
    .stApp {
        background-color: var(--bg-void);
        background-image:
            radial-gradient(700px circle at var(--mouse-x, 50vw) var(--mouse-y, 30vh), rgba(0, 255, 136, 0.06), transparent 70%),
            radial-gradient(1000px circle at 80% 20%, rgba(30, 41, 59, 0.5), transparent 80%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
        color: var(--text-pure);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Headings & Monospace */
    h1, h2, h3 {
        font-family: 'Chakra Petch', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.04em !important;
    }
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Header Masthead */
    .sentinel-masthead {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--panel-border);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .sentinel-title {
        font-family: 'Chakra Petch', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        letter-spacing: 0.06em;
        color: #fff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .beacon {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: var(--bio-green);
        box-shadow: 0 0 12px var(--bio-green);
        animation: pulseBeacon 2s infinite;
    }
    @keyframes pulseBeacon {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.6); opacity: 0.4; }
    }

    /* Sensor Capture Viewport Frame */
    .sensor-viewport {
        position: relative;
        background: rgba(10, 13, 18, 0.95);
        border: 1px solid var(--panel-border);
        border-radius: 12px;
        padding: 12px;
        backdrop-filter: blur(16px);
        transition: border-color 0.3s ease;
    }
    .sensor-viewport:hover {
        border-color: rgba(0, 255, 136, 0.35);
    }
    .corner-tl, .corner-tr, .corner-bl, .corner-br {
        position: absolute;
        width: 14px;
        height: 14px;
        border-color: var(--bio-green);
        pointer-events: none;
    }
    .corner-tl { top: 6px; left: 6px; border-top: 2px solid; border-left: 2px solid; }
    .corner-tr { top: 6px; right: 6px; border-top: 2px solid; border-right: 2px solid; }
    .corner-bl { bottom: 6px; left: 6px; border-bottom: 2px solid; border-left: 2px solid; }
    .corner-br { bottom: 6px; right: 6px; border-bottom: 2px solid; border-right: 2px solid; }

    /* Tactical Hero Presence Status Card */
    .presence-hero-box {
        border-radius: 14px;
        padding: 26px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        border: 1px solid;
    }
    .presence-detected {
        background: linear-gradient(145deg, rgba(0, 255, 136, 0.08) 0%, rgba(13, 16, 23, 0.95) 100%);
        border-color: rgba(0, 255, 136, 0.5);
        box-shadow: 0 15px 40px -10px rgba(0, 255, 136, 0.18), inset 0 0 20px rgba(0, 255, 136, 0.04);
    }
    .presence-vacant {
        background: linear-gradient(145deg, rgba(100, 116, 139, 0.08) 0%, rgba(13, 16, 23, 0.95) 100%);
        border-color: rgba(100, 116, 139, 0.4);
        box-shadow: 0 15px 40px -10px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(100, 116, 139, 0.03);
    }
    .presence-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .presence-status {
        font-family: 'Chakra Petch', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.01em;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    /* Gauge / Meter styling */
    .meter-container {
        background: rgba(255, 255, 255, 0.05);
        height: 10px;
        border-radius: 999px;
        overflow: hidden;
        margin: 14px 0 6px 0;
    }
    .meter-bar-detected {
        height: 100%;
        background: linear-gradient(90deg, #00c868, #00ff88);
        box-shadow: 0 0 15px #00ff88;
        border-radius: 999px;
    }
    .meter-bar-vacant {
        height: 100%;
        background: linear-gradient(90deg, #475569, #94a3b8);
        border-radius: 999px;
    }

    /* Telemetry Pill Grid */
    .telemetry-cell {
        background: rgba(19, 23, 34, 0.7);
        border: 1px solid var(--panel-border);
        border-radius: 10px;
        padding: 14px 18px;
        transition: all 0.2s ease;
    }
    .telemetry-cell:hover {
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    .cell-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--slate-cold);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .cell-val {
        font-family: 'Chakra Petch', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LOAD METADATA & MODEL (TEACHABLE MACHINE ENGINE)
# ---------------------------------------------------------
@st.cache_resource
def load_teachable_metadata():
    try:
        with open("metadata.json", "r") as f:
            data = json.load(f)
            return data.get("labels", ["Person", "Person't"])
    except Exception:
        return ["Person", "Person't"]

@st.cache_resource
def load_teachable_model():
    try:
        from keras.models import load_model
        model = load_model("keras_model.h5", compile=False)
        return model
    except Exception as e:
        return None

labels = load_teachable_metadata()
model = load_teachable_model()

# ---------------------------------------------------------
# MASTHEAD / TOP DECK
# ---------------------------------------------------------
st.markdown(
    """
    <div class="sentinel-masthead">
        <div>
            <div class="sentinel-title">
                <span class="beacon"></span>
                SENTINEL // PRESENCE DETECTION SUITE
            </div>
            <div class="mono" style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                TEACHABLE MACHINE NEURAL CLASSIFIER • OPTICAL PERIMETER MONITOR
            </div>
        </div>
        <div class="mono" style="text-align: right; font-size: 0.75rem;">
            <span style="color: #64748b;">CHANNEL:</span> <span style="color: #00ff88;">REALTIME-OPTICS</span><br>
            <span style="color: #64748b;">SENSOR NODE:</span> <span style="color: #cbd5e1;">SN-804-FLIR</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SENSOR CONTROL BAR & WORKSPACE
# ---------------------------------------------------------
col_settings, col_mode = st.columns([2, 1])

with col_settings:
    st.markdown("<div class='mono' style='font-size: 0.75rem; color: #94a3b8;'>ACQUISITION APERTURE</div>", unsafe_allow_html=True)
    input_method = st.segmented_control(
        "Sensor Source",
        options=["Optical Sensor (Webcam)", "Digital Ingestion (Upload)"],
        default="Optical Sensor (Webcam)",
        label_visibility="collapsed"
    )

with col_mode:
    sensitivity_threshold = st.slider(
        "Detection Certainty Trigger",
        min_value=0.50,
        max_value=0.99,
        value=0.65,
        step=0.01,
        help="Minimum confidence threshold required to trigger an affirmative human signature alert."
    )

st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# FRAME CAPTURE ZONE
# ---------------------------------------------------------
sensor_image = None

col_viewport, col_telemetry = st.columns([1.2, 1], gap="large")

with col_viewport:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
            <span style="font-family: 'Chakra Petch'; font-size: 1.05rem; font-weight: 600;">SENSOR VIEWPORT</span>
            <span class="mono" style="font-size: 0.7rem; color: #64748b;">TARGET RES: 224x224 RGB</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if input_method == "Optical Sensor (Webcam)":
        cam_buffer = st.camera_input("Optical Sensor Feed", label_visibility="collapsed")
        if cam_buffer:
            sensor_image = Image.open(cam_buffer).convert("RGB")
    else:
        file_buffer = st.file_uploader("Ingest Reference Frame", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if file_buffer:
            sensor_image = Image.open(file_buffer).convert("RGB")

    # Corner bracket aesthetic wrapper
    if sensor_image:
        st.markdown(
            """
            <div class="sensor-viewport">
                <div class="corner-tl"></div>
                <div class="corner-tr"></div>
                <div class="corner-bl"></div>
                <div class="corner-br"></div>
            """,
            unsafe_allow_html=True
        )
        st.image(sensor_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="border: 1px dashed rgba(255,255,255,0.12); border-radius: 12px; padding: 4.5rem 1rem; text-align: center;">
                <div style="font-family: 'Chakra Petch'; font-size: 1.15rem; color: #94a3b8;">NO INGESTED SIGNAL</div>
                <div class="mono" style="font-size: 0.75rem; color: #475569; margin-top: 6px;">Activate the optical sensor above to evaluate subject presence.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# INFERENCE & PRESENCE EVALUATION
# ---------------------------------------------------------
with col_telemetry:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
            <span style="font-family: 'Chakra Petch'; font-size: 1.05rem; font-weight: 600;">PRESENCE EVALUATION</span>
            <span class="mono" style="font-size: 0.7rem; color: #64748b;">DISCRIMINATIVE INFERENCE</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if sensor_image is not None:
        if model is None:
            st.error("Missing model artifact: `keras_model.h5` could not be initialized.")
            st.stop()

        # Teachable Machine Preprocessing Pipeline
        t_start = time.perf_counter()
        resized_img = sensor_image.resize((224, 224))
        image_array = np.asarray(resized_img, dtype=np.float32)
        
        # Standard Teachable Machine normalization: scale from [0, 255] to [-1, 1]
        normalized_img = (image_array / 127.5) - 1.0
        data_payload = np.expand_dims(normalized_img, axis=0)

        # Execute classification
        predictions = model.predict(data_payload, verbose=0)[0]
        eval_time_ms = (time.perf_counter() - t_start) * 1000

        # Map predictions to metadata labels: Index 0 = Person, Index 1 = Person't
        prob_person = float(predictions[0])
        prob_absent = float(predictions[1]) if len(predictions) > 1 else (1.0 - prob_person)

        is_person_detected = prob_person >= sensitivity_threshold

        # High-Impact Tactical Status Hero
        if is_person_detected:
            st.markdown(
                f"""
                <div class="presence-hero-box presence-detected">
                    <div class="presence-tag" style="color: #00ff88;">● VERIFIED ORGANIC SIGNATURE</div>
                    <div class="presence-status" style="color: #00ff88;">HUMAN DETECTED</div>
                    <div class="mono" style="font-size: 0.85rem; color: #cbd5e1;">
                        Subject confirmed within sensor aperture with {prob_person * 100:.1f}% confidence.
                    </div>
                    <div class="meter-container">
                        <div class="meter-bar-detected" style="width: {prob_person * 100}%;"></div>
                    </div>
                    <div class="mono" style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b;">
                        <span>TRIGGER: {sensitivity_threshold * 100:.0f}%</span>
                        <span style="color: #00ff88;">INDEX: {prob_person:.3f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="presence-hero-box presence-vacant">
                    <div class="presence-tag" style="color: #94a3b8;">○ SECTOR UNINHABITED</div>
                    <div class="presence-status" style="color: #94a3b8;">NO PERSON DETECTED</div>
                    <div class="mono" style="font-size: 0.85rem; color: #64748b;">
                        Absence certainty is verified at {prob_absent * 100:.1f}%. Perimeter is vacant.
                    </div>
                    <div class="meter-container">
                        <div class="meter-bar-vacant" style="width: {prob_absent * 100}%;"></div>
                    </div>
                    <div class="mono" style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b;">
                        <span>VACANCY PROBABILITY</span>
                        <span style="color: #cbd5e1;">INDEX: {prob_absent:.3f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Telemetry Data Grid
        cell_1, cell_2 = st.columns(2)
        with cell_1:
            st.markdown(
                f"""
                <div class="telemetry-cell">
                    <div class="cell-title">Latency Speed</div>
                    <div class="cell-val">{eval_time_ms:.1f} <span class="mono" style="font-size: 0.8rem; color: #64748b;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cell_2:
            st.markdown(
                f"""
                <div class="telemetry-cell">
                    <div class="cell-title">Dominant Class</div>
                    <div class="cell-val" style="color: {'#00ff88' if is_person_detected else '#94a3b8'};">
                        {labels[0] if is_person_detected else labels[1]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

        # Probability Breakdown Breakdown
        chart_df = pd.DataFrame({
            "Classification Target": [str(l) for l in labels],
            "Probability": [prob_person, prob_absent]
        })
        st.bar_chart(
            chart_df.set_index("Classification Target"),
            color="#00ff88" if is_person_detected else "#64748b"
        )

    else:
        st.markdown(
            """
            <div class="telemetry-cell" style="padding: 2.5rem 1.5rem; text-align: center;">
                <div class="mono" style="font-size: 0.8rem; color: #64748b;">STANDBY MODE</div>
                <div style="font-family: 'Chakra Petch'; font-size: 1.1rem; color: #f8fafc; margin-top: 6px;">
                    AWAITING FRAME CAPTURE
                </div>
                <div class="mono" style="font-size: 0.75rem; color: #475569; margin-top: 6px;">
                    Acquire an optical sample to populate presence indices and latency readouts.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
