import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import cv2
import tempfile
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Product Defect Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD CSS
# =====================================

try:
    with open("style/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

# =====================================
# LOAD YOLO MODEL
# =====================================

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🏭 AI INSPECT")

    st.caption("Industrial Quality Inspection")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📤 Upload Image",
            "📊 Statistics",
            "🕘 History",
            "⚙ Settings"
        ]
    )

    st.divider()

    st.subheader("🤖 Model Status")

    st.success("YOLOv8 Active")

    st.write("Model : best.pt")
    st.write("Device : CPU")

    st.divider()

    st.metric("Today's Scan", "1248")
    st.metric("Accuracy", "96.32%")

    st.divider()

    st.caption("© AI Inspect 2026")

# =====================================
# HEADER
# =====================================

left, right = st.columns([5,1])

with left:

    st.markdown("""
    <h1 style='color:white;'>
    🏭 AI Product Defect Detection System
    </h1>

    <p style='color:gray;'>
    Industrial Inspection using YOLOv8
    </p>
    """,
    unsafe_allow_html=True
    )

with right:

    st.metric(
        "📅 Date",
        datetime.now().strftime("%d-%m-%Y")
    )

    st.metric(
        "🕒 Time",
        datetime.now().strftime("%H:%M")
    )

st.divider()

# =====================================
# DASHBOARD CARDS
# =====================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("📷 Images", "1248", "+12")

with c2:
    st.metric("✅ Good", "842", "+8")

with c3:
    st.metric("❌ Defects", "406", "-2")

with c4:
    st.metric("🎯 Accuracy", "96.32%")

st.divider()

# =====================================
# MAIN SECTION
# =====================================

left, right = st.columns([1,1])

# =====================================
# LEFT PANEL
# =====================================

with left:

    st.subheader("📤 Upload Product Image")

    uploaded_file = st.file_uploader(
        "Select Product Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    image = None

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )
        # =====================================
# RIGHT PANEL (YOLO DETECTION)
# =====================================

with right:

    st.subheader("🔍 Detection Result")

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

            image.save(tmp.name)
            temp_path = tmp.name

        with st.spinner("Running YOLO Detection..."):

            results = model.predict(
                source=temp_path,
                conf=0.10,
                save=False,
                verbose=False
            )

        result = results[0]

        detection_count = len(result.boxes)

        plotted = result.plot()

        plotted = cv2.cvtColor(
            plotted,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            plotted,
            caption="Detection Result",
            width=300
        )

        st.metric(
            "Detections",
            detection_count
        )

        if detection_count > 0:

            confidence = float(result.boxes.conf[0]) * 100

            class_id = int(result.boxes.cls[0])

            class_name = model.names[class_id]

            st.markdown("## ❌ DEFECT DETECTED")

            st.error("Status : FAIL")

            st.progress(confidence / 100)

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with c2:
                st.metric(
                    "Defect",
                    class_name
                )

        else:

            st.markdown("## ✅ PRODUCT PASSED")

            st.success("Status : PASS")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Confidence",
                    "--"
                )

            with c2:
                st.metric(
                    "Defect",
                    "None"
                )

    else:

        st.info("Upload an image to begin inspection.")

# =====================================
# CHARTS
# =====================================

st.divider()

left_chart, right_table = st.columns([1, 1.2])

with left_chart:

    st.subheader("📊 Defect Distribution")

    labels = [
        "Scratch",
        "Bent",
        "Color",
        "Flip"
    ]

    values = [
        42,
        25,
        18,
        15
    ]

    fig = go.Figure(

        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6
            )
        ]

    )

    fig.update_layout(

        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        height=350

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# HISTORY TABLE
# =====================================

with right_table:

    st.subheader("📝 Recent Detections")

    history = pd.DataFrame({

        "Image": [

            "metal_001.png",
            "metal_002.png",
            "metal_003.png",
            "metal_004.png"

        ],

        "Result": [

            "PASS",
            "FAIL",
            "PASS",
            "FAIL"

        ],

        "Confidence": [

            "100%",
            "98.42%",
            "99.80%",
            "97.65%"

        ]

    })

    st.dataframe(

        history,

        use_container_width=True,

        hide_index=True

    )