import os
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# ---------------------------------------------------------
# STREAMLIT CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SENTINEL // Person Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# TACTICAL BIO-PERIMETER STYLING (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    :root {
        --bg-void: #07090e;
        --card-bg: rgba(15, 18, 28, 0.85);
        --bio-green: #00ff88;
        --alert-coral: #ff3366;
        --slate-cold: #64748b;
        --border-subtle: rgba(255, 255, 255, 0.08);
    }

    .stApp {
        background-color: var(--bg-void);
        background-image:
            radial-gradient(800px circle at 50% 20%, rgba(0, 255, 136, 0.06), transparent 70%),
            radial-gradient(1000px circle at 90% 80%, rgba(30, 41, 59, 0.3), transparent 75%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Chakra Petch', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.04em !important;
    }
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Target Streamlit Camera/Image Viewports */
    div[data-testid="stImage"] img, div[data-testid="stCameraInput"] {
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 12px;
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.08);
    }

    /* Presence Hero Box */
    .presence-hero {
        border-radius: 14px;
        padding: 24px;
        border: 1px solid var(--border-subtle);
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .presence-hero.detected {
        background: linear-gradient(145deg, rgba(0, 255, 136, 0.08) 0%, rgba(15, 18, 28, 0.95) 100%);
        border-color: rgba(0, 255, 136, 0.5);
        box-shadow: 0 12px 35px -10px rgba(0, 255, 136, 0.2);
    }
    .presence-hero.vacant {
        background: linear-gradient(145deg, rgba(100, 116, 139, 0.08) 0%, rgba(15, 18, 28, 0.95) 100%);
        border-color: rgba(100, 116, 139, 0.35);
        box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.4);
    }
    .hero-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-status {
        font-family: 'Chakra Petch', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    /* Tactical Progress Bar */
    .meter-container {
        background: rgba(255, 255, 255, 0.08);
        height: 10px;
        border-radius: 999px;
        overflow: hidden;
        margin: 14px 0 6px 0;
    }
    .meter-fill-detected {
        height: 100%;
        background: linear-gradient(90deg, #00c868, #00ff88);
        box-shadow: 0 0 15px #00ff88;
        border-radius: 999px;
    }
    .meter-fill-vacant {
        height: 100%;
        background: linear-gradient(90deg, #475569, #94a3b8);
        border-radius: 999px;
    }

    /* Telemetry Metric Cards */
    .telemetry-card {
        background: var(--card-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px;
    }
    .telemetry-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--slate-cold);
        text-transform: uppercase;
    }
    .telemetry-val {
        font-family: 'Chakra Petch', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LABELS & MODEL INITIALIZATION
# ---------------------------------------------------------
def get_labels():
    if os.path.exists("metadata.json"):
        try:
            with open("metadata.json", "r") as f:
                data = json.load(f)
                return data.get("labels", ["Person", "Person't"])
        except Exception:
            pass
    return ["Person", "Person't"]

labels = get_labels()

@st.cache_resource
def load_detection_engine():
    # 1. Try loading Keras H5 if present
    if os.path.exists("keras_model.h5"):
        try:
            from keras.models import load_model
            model = load_model("keras_model.h5", compile=False)
            return "keras", model
        except Exception as e:
            st.sidebar.warning(f"Could not load keras_model.h5: {e}")

    # 2. Native Fallback: OpenCV HOG Human Detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return "opencv_hog", hog

engine_type, model_engine = load_detection_engine()

# ---------------------------------------------------------
# SIDEBAR TELEMETRY & CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ SENSOR TELEMETRY")
    
    if engine_type == "keras":
        st.success("● ENGINE: Keras Neural Network (`keras_model.h5`)")
    else:
        st.info("● ENGINE: OpenCV Bio-Vision Engine (Auto-Fallback)")
        st.caption(
            "Note: `keras_model.h5` was not found on disk. The system is actively using "
            "Python's native HOG computer vision detector."
        )

    st.markdown("---")
    st.markdown("### SENSITIVITY")
    sensitivity_threshold = st.slider(
        "Verification Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Minimum confidence required to classify affirmative human presence."
    )
    
    st.markdown("---")
    st.markdown(
        f"""
        <div class="mono" style="font-size: 0.75rem; color: #64748b;">
            TARGET CLASS 0: {labels[0]}<br>
            TARGET CLASS 1: {labels[1]}<br>
            SYS STATUS: ONLINE
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# MAIN APP HEADER
# ---------------------------------------------------------
st.title("SENTINEL // PERSON DETECTOR")
st.caption("Autonomous Human Presence Monitoring System")

# Source selector
input_mode = st.radio(
    "Sensor Channel",
    ["Camera Ingestion", "Reference Image Upload"],
    horizontal=True,
    label_visibility="collapsed"
)

# ---------------------------------------------------------
# VIEWPORT & TELEMETRY COLUMNS
# ---------------------------------------------------------
col_view, col_telemetry = st.columns([1.2, 1], gap="large")

input_image = None

with col_view:
    st.markdown("### Optical Viewport")

    if input_mode == "Camera Ingestion":
        picture = st.camera_input("Optical Sensor Feed", label_visibility="collapsed")
        if picture:
            input_image = Image.open(picture).convert("RGB")
    else:
        uploaded_file = st.file_uploader(
            "Ingest Optical Snapshot",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            input_image = Image.open(uploaded_file).convert("RGB")

    if input_image is not None and input_mode != "Camera Ingestion":
        st.image(input_image, use_container_width=True)

# ---------------------------------------------------------
# PYTHON INFERENCE ENGINE
# ---------------------------------------------------------
with col_telemetry:
    st.markdown("### Presence Telemetry")

    if input_image is not None:
        t_start = time.perf_counter()

        # Engine A: Keras Model
        if engine_type == "keras":
            # Normalization matching Teachable Machine specification
            resized = input_image.resize((224, 224))
            img_arr = np.asarray(resized, dtype=np.float32)
            normalized = (img_arr / 127.5) - 1.0
            payload = np.expand_dims(normalized, axis=0)

            predictions = model_engine.predict(payload, verbose=0)[0]
            prob_person = float(predictions[0])
            prob_absent = float(predictions[1]) if len(predictions) > 1 else (1.0 - prob_person)

        # Engine B: OpenCV HOG Human Detector
        else:
            cv_img = np.array(input_image)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            
            # Detect human bounding boxes & confidence weights
            boxes, weights = model_engine.detectMultiScale(cv_img, winStride=(8, 8), scale=1.05)
            
            if len(boxes) > 0 and len(weights) > 0:
                # Normalize HOG detection score to probability
                max_w = float(np.max(weights))
                prob_person = float(1.0 / (1.0 + np.exp(-max_w)))
                prob_absent = 1.0 - prob_person
            else:
                prob_person = 0.12
                prob_absent = 0.88

        latency_ms = (time.perf_counter() - t_start) * 1000
        is_person = prob_person >= sensitivity_threshold

        pct_person = round(prob_person * 100, 1)
        pct_absent = round(prob_absent * 100, 1)
        trigger_pct = int(sensitivity_threshold * 100)

        # Tactical Presence Card
        if is_person:
            st.markdown(
                f"""
                <div class="presence-hero detected">
                    <div class="hero-tag" style="color: #00ff88;">● ORGANIC SIGNATURE VERIFIED</div>
                    <div class="hero-status" style="color: #00ff88;">HUMAN DETECTED</div>
                    <div class="mono" style="font-size: 0.85rem; color: #cbd5e1;">
                        Subject detected within sensor perimeter with {pct_person}% confidence.
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill-detected" style="width: {pct_person}%;"></div>
                    </div>
                    <div class="mono" style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b;">
                        <span>TRIGGER: {trigger_pct}%</span>
                        <span style="color: #00ff88;">INDEX: {prob_person:.3f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="presence-hero vacant">
                    <div class="hero-tag" style="color: #94a3b8;">○ PERIMETER UNINHABITED</div>
                    <div class="hero-status" style="color: #94a3b8;">NO PERSON DETECTED</div>
                    <div class="mono" style="font-size: 0.85rem; color: #64748b;">
                        Absence certainty verified at {pct_absent}%. Sector clear.
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill-vacant" style="width: {pct_absent}%;"></div>
                    </div>
                    <div class="mono" style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b;">
                        <span>TRIGGER: {trigger_pct}%</span>
                        <span style="color: #cbd5e1;">INDEX: {prob_absent:.3f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Telemetry Data Grid
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div class="telemetry-card">
                    <div class="telemetry-label">Response Latency</div>
                    <div class="telemetry-val">{latency_ms:.1f} <span class="mono" style="font-size: 0.8rem; color: #64748b;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            dominant_name = labels[0] if is_person else labels[1]
            dominant_color = "#00ff88" if is_person else "#94a3b8"
            st.markdown(
                f"""
                <div class="telemetry-card">
                    <div class="telemetry-label">Classification</div>
                    <div class="telemetry-val" style="color: {dominant_color};">{dominant_name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Probability Breakdown Chart
        chart_data = pd.DataFrame({
            "Target": [str(l) for l in labels],
            "Probability": [prob_person, prob_absent]
        })
        st.bar_chart(chart_data.set_index("Target"), color=dominant_color)

    else:
        st.markdown(
            """
            <div class="telemetry-card" style="padding: 3rem 1.5rem; text-align: center;">
                <div class="mono" style="font-size: 0.8rem; color: #64748b;">STANDBY</div>
                <div style="font-family: 'Chakra Petch'; font-size: 1.15rem; color: #f8fafc; margin-top: 6px;">
                    AWAITING FRAME CAPTURE
                </div>
                <div class="mono" style="font-size: 0.75rem; color: #475569; margin-top: 6px;">
                    Take a picture with the camera to run presence analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
