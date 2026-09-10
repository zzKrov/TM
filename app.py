import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from keras.models import load_model
from keras.layers import DepthwiseConv2D

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="SENTINEL // Person Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# NEON BIO-PERIMETER STYLESHEET
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    :root {
        --bg-void: #06070a;
        --card-bg: rgba(14, 18, 28, 0.85);
        --bio-green: #00ff88;
        --slate-cold: #64748b;
        --border-glow: rgba(0, 255, 136, 0.25);
    }

    .stApp {
        background-color: var(--bg-void);
        background-image:
            radial-gradient(750px circle at 50% 15%, rgba(0, 255, 136, 0.08), transparent 70%),
            radial-gradient(900px circle at 85% 85%, rgba(30, 41, 59, 0.35), transparent 75%),
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

    /* Hero Status Cards */
    .presence-hero {
        border-radius: 14px;
        padding: 24px;
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border: 1px solid;
    }
    .presence-hero.detected {
        background: linear-gradient(145deg, rgba(0, 255, 136, 0.08) 0%, rgba(14, 18, 28, 0.95) 100%);
        border-color: rgba(0, 255, 136, 0.5);
        box-shadow: 0 12px 35px -10px rgba(0, 255, 136, 0.2);
    }
    .presence-hero.vacant {
        background: linear-gradient(145deg, rgba(100, 116, 139, 0.08) 0%, rgba(14, 18, 28, 0.95) 100%);
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
        font-size: 2.1rem;
        line-height: 1.1;
        margin-bottom: 10px;
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
        border: 1px solid rgba(255, 255, 255, 0.08);
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

    /* Camera/Image styling */
    div[data-testid="stImage"] img, div[data-testid="stCameraInput"] {
        border: 1px solid var(--border-glow);
        border-radius: 12px;
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# CUSTOM DEPTHWISECONV2D FIX FOR TEACHABLE MACHINE
# ---------------------------------------------------------
class CustomDepthwiseConv2D(DepthwiseConv2D):
    """Overrides DepthwiseConv2D to discard obsolete 'groups' argument."""
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)

# ---------------------------------------------------------
# LOAD MODEL & LABELS
# ---------------------------------------------------------
def load_labels():
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            cleaned = []
            for line in lines:
                parts = line.split(" ", 1)
                cleaned.append(parts[1] if (len(parts) == 2 and parts[0].isdigit()) else line)
            return cleaned
    return ["Person", "Person't"]

@st.cache_resource
def load_teachable_model():
    if not os.path.exists("keras_model.h5"):
        return None
    return load_model(
        "keras_model.h5",
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
        compile=False
    )

model = load_teachable_model()
labels = load_labels()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    if os.path.exists("OIG5.jpg"):
        st.image("OIG5.jpg", use_container_width=True)

    st.markdown("### 🛡️ SENSOR TELEMETRY")
    if model:
        st.success("● MODEL: `keras_model.h5` [MOUNTED]")
    else:
        st.error("❌ ERROR: `keras_model.h5` not found.")

    st.markdown("---")
    st.markdown("### LABELS DETECTED")
    for i, lbl in enumerate(labels):
        st.markdown(f"<span class='mono' style='color:#00ff88;'>[{i}]</span> {lbl}", unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
st.title("SENTINEL // PERSON DETECTOR")
st.caption("Teachable Machine Autonomous Presence Sensor")

if model is None:
    st.error("Missing `keras_model.h5` file in repository root.")
    st.stop()

col_view, col_telemetry = st.columns([1.2, 1], gap="large")

with col_view:
    st.markdown("### Sensor Viewport")
    camera_buffer = st.camera_input("Optical Sensor Feed", label_visibility="collapsed")

with col_telemetry:
    st.markdown("### Presence Telemetry")

    if camera_buffer is not None:
        # Standard Teachable Machine (224, 224, 3) normalization
        t_start = time.perf_counter()
        img = Image.open(camera_buffer).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.asarray(img_resized, dtype=np.float32)
        
        normalized = (img_array / 127.5) - 1.0
        data = np.expand_dims(normalized, axis=0)

        # Run inference
        prediction = model.predict(data, verbose=0)[0]
        latency_ms = (time.perf_counter() - t_start) * 1000

        top_idx = int(np.argmax(prediction))
        conf = float(prediction[top_idx])
        top_label = labels[top_idx] if top_idx < len(labels) else f"Class {top_idx}"

        # Classify presence: label contains "person" and not "person't" / "no person"
        is_person = "person't" not in top_label.lower() and "no" not in top_label.lower() and "person" in top_label.lower()
        pct_conf = round(conf * 100, 1)

        # Hero presence card
        if is_person:
            st.markdown(
                f"""
                <div class="presence-hero detected">
                    <div class="hero-tag" style="color: #00ff88;">● ORGANIC SIGNATURE DETECTED</div>
                    <div class="hero-status" style="color: #00ff88;">HUMAN CONFIRMED</div>
                    <div class="mono" style="font-size: 0.85rem; color: #cbd5e1;">
                        Subject detected with {pct_conf}% model certainty.
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill-detected" style="width: {pct_conf}%;"></div>
                    </div>
                    <div class="mono" style="text-align: right; font-size: 0.75rem; color: #00ff88; margin-top: 4px;">
                        INDEX: {conf:.3f}
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
                        Absence certainty verified at {pct_conf}%. Sector clear.
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill-vacant" style="width: {pct_conf}%;"></div>
                    </div>
                    <div class="mono" style="text-align: right; font-size: 0.75rem; color: #cbd5e1; margin-top: 4px;">
                        INDEX: {conf:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Stat cards
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div class="telemetry-card">
                    <div class="telemetry-label">Response Time</div>
                    <div class="telemetry-val">{latency_ms:.1f} <span class="mono" style="font-size: 0.8rem; color: #64748b;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="telemetry-card">
                    <div class="telemetry-label">Classification</div>
                    <div class="telemetry-val" style="color: {'#00ff88' if is_person else '#94a3b8'};">{top_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Probability breakdown chart
        chart_data = pd.DataFrame({
            "Label": labels[:len(prediction)],
            "Probability": [float(p) for p in prediction]
        })
        st.bar_chart(chart_data.set_index("Label"), color="#00ff88" if is_person else "#64748b")

    else:
        st.markdown(
            """
            <div class="telemetry-card" style="padding: 3rem 1.5rem; text-align: center;">
                <div class="mono" style="font-size: 0.8rem; color: #64748b;">STANDBY</div>
                <div style="font-family: 'Chakra Petch'; font-size: 1.15rem; color: #f8fafc; margin-top: 6px;">
                    AWAITING OPTICAL FRAME
                </div>
                <div class="mono" style="font-size: 0.75rem; color: #475569; margin-top: 6px;">
                    Take a snapshot with the camera above to run presence analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
