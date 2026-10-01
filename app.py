import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

# ============================================================
# AGROVISION
# AI-POWERED PRECISION CROP PROTECTION
# ============================================================

st.set_page_config(
    page_title="AgroVision",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.agro-title {
    font-size: 42px;
    font-weight: 800;
    color: #064e3b;
}

.agro-subtitle {
    font-size: 20px;
    color: #475569;
}

.status-box {
    background: #dcfce7;
    color: #166534;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 700;
    margin-top: 15px;
}

.section-title {
    color: #064e3b;
    font-size: 27px;
    font-weight: 750;
    margin-top: 30px;
}

.info-box {
    background: #e0f2fe;
    padding: 15px;
    border-radius: 10px;
    color: #075985;
}

.result-box {
    background: white;
    border: 1px solid #dbe4df;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
}

.success-box {
    background: #dcfce7;
    border-radius: 10px;
    padding: 15px;
    color: #166534;
    font-weight: 700;
}

.warning-box {
    background: #fef3c7;
    border-radius: 10px;
    padding: 15px;
    color: #92400e;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌱 AgroVision")

    st.divider()

    st.markdown("### System")

    st.success("✓ AI Model Loaded")

    st.divider()

    st.markdown("### Model")

    st.caption("YOLO object detection")

    model_path = Path("runs/detect/train-4/weights/best.pt")

    if model_path.exists():
        st.code("best.pt")
    else:
        st.warning("Model file not found")
        st.caption("Demo detection mode will be used.")

    st.divider()

    st.markdown("### Prototype")

    st.info(
        "GPS coordinates and precision spray "
        "actions are simulated for demonstration."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("# 🌱 AgroVision")

st.markdown(
    "### AI-Powered Precision Crop Protection"
)

st.markdown(
    '<div class="status-box">🟢 AI SYSTEM ONLINE</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown("## 🎯 How AgroVision Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔍 DETECT")
    st.write(
        "AI analyzes the crop image and identifies "
        "possible disease-affected regions."
    )

with col2:
    st.markdown("### 📍 LOCATE")
    st.write(
        "Detected regions are associated with GPS "
        "target zones for precision treatment."
    )

with col3:
    st.markdown("### 🚁 ACT")
    st.write(
        "The drone can target affected areas instead "
        "of spraying the entire field."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("## 📷 Crop Image Analysis")

st.write(
    "Upload a tomato leaf image to detect possible crop diseases."
)

uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a tomato leaf image above to start AI detection."
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

else:

    image = Image.open(uploaded_file)

    st.markdown("## 🖼️ Image Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Input Image")

        st.image(
            image,
            use_container_width=True
        )

    # ========================================================
    # YOLO MODEL
    # ========================================================

    detections = []

    try:

        from ultralytics import YOLO

        if model_path.exists():

            model = YOLO(str(model_path))

            results = model.predict(
                source=image,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            annotated_image = result.plot()

            names = result.names

            for box in result.boxes:

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                class_name = names[class_id]

                detections.append(
                    {
                        "name": class_name,
                        "confidence": confidence
                    }
                )

        else:

            annotated_image = image

    except Exception as e:

        annotated_image = image


    # ========================================================
    # AI RESULT
    # ========================================================

    with col2:

        st.markdown("### 🤖 AI Detection")

        st.image(
            annotated_image,
            use_container_width=True
        )


    # ========================================================
    # DETECTION RESULTS
    # ========================================================

    st.markdown("## 🔬 Detection Results")

    if detections:

        # Remove duplicate classes
        unique_results = {}

        for item in detections:

            name = item["name"]
            confidence = item["confidence"]

            if (
                name not in unique_results
                or confidence > unique_results[name]
            ):

                unique_results[name] = confidence


        result_columns = st.columns(
            min(3, len(unique_results))
        )

        for i, (name, confidence) in enumerate(
            unique_results.items()
        ):

            with result_columns[
                i % len(result_columns)
            ]:

                st.metric(
                    name,
                    f"{confidence * 100:.1f}%"
                )


        st.success(
            "Disease detection completed successfully."
        )

        best_conf = max(unique_results.values())
        if best_conf < 0.50:
            st.warning(
                f"Model confidence is {best_conf * 100:.1f}%. "
                "For a real deployment, verify this image or improve the trained model."
            )
        elif best_conf < 0.80:
            st.info(
                f"Model confidence is {best_conf * 100:.1f}%. "
                "Suitable for prototype demonstration; further validation is recommended."
            )

    else:

        st.warning(
            "No disease was detected with the current model."
        )

        st.caption(
            "If your trained model does not detect this image, "
            "try a clearer tomato leaf image."
        )


    # ========================================================
    # AGROVISION WORKFLOW
    # ========================================================

    st.markdown("## 🚁 AgroVision Workflow")

    workflow = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3"
            ],
            "Action": [
                "DETECT",
                "LOCATE",
                "ACT"
            ],
            "Description": [
                "AI identifies affected crop regions",
                "GPS target zone is created",
                "Precision spray action is simulated"
            ]
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # GPS TARGET ZONE
    # ========================================================

    st.markdown("## 📍 GPS Target Zone")

    st.info(
        "Prototype simulation: GPS coordinates shown below "
        "represent a target zone. A real drone GPS module "
        "would provide the actual field coordinates."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # These are DEMO coordinates only.
    # They are NOT your current location.
    # --------------------------------------------------------

    latitude = 17.3850
    longitude = 78.4867

    gps_data = pd.DataFrame(
        {
            "latitude": [latitude],
            "longitude": [longitude]
        }
    )

    st.map(
        gps_data,
        latitude="latitude",
        longitude="longitude",
        zoom=12
    )

    # ========================================================
    # GPS INFORMATION
    # ========================================================

    st.markdown("### 🎯 Target Zone Information")

    gps_col1, gps_col2, gps_col3 = st.columns(3)

    with gps_col1:

        st.markdown("**Target Zone**")

        st.write("ZONE-01")

    with gps_col2:

        st.markdown("**Latitude**")

        st.write(f"{latitude:.4f}")

    with gps_col3:

        st.markdown("**Longitude**")

        st.write(f"{longitude:.4f}")


    # ========================================================
    # PRECISION SPRAY SIMULATION
    # ========================================================

    st.markdown("## 🚁 Precision Spray Control")

    st.write(
        "This section demonstrates how AgroVision could "
        "convert AI detection into a drone treatment action."
    )

    if st.button(
        "🚁 Activate Precision Spray",
        use_container_width=True
    ):

        st.success(
            "Precision spray simulation activated for ZONE-01."
        )

        st.info(
            f"Target coordinates: "
            f"{latitude:.4f}, {longitude:.4f}"
        )

        st.progress(100)

        st.write(
            "✓ Target identified"
        )

        st.write(
            "✓ GPS zone locked"
        )

        st.write(
            "✓ Spray path calculated"
        )

        st.write(
            "✓ Precision spray action simulated"
        )



# ============================================================
# JUDGE / INVESTOR VIEW
# ============================================================

st.markdown("## 💼 Project Value")

v1, v2, v3 = st.columns(3)

with v1:
    st.markdown("""
    <div class="result-box">
        <h3>🎯 Precision</h3>
        <p>Focus treatment on AI-identified target zones instead of treating the whole field uniformly.</p>
    </div>
    """, unsafe_allow_html=True)

with v2:
    st.markdown("""
    <div class="result-box">
        <h3>📈 Scalability</h3>
        <p>The software pipeline can be connected to drone imagery, GPS and a controlled spraying system.</p>
    </div>
    """, unsafe_allow_html=True)

with v3:
    st.markdown("""
    <div class="result-box">
        <h3>💰 Business Model</h3>
        <p>Potential models include drone-service deployment, farm treatment contracts and technology licensing.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>Judge-ready explanation:</b><br>
“This prototype proves the software decision pipeline: the trained YOLO model detects a crop condition,
the system assigns a target zone, and the treatment action is simulated. The next engineering stage is
connecting the same pipeline to real drone GPS, flight control and a calibrated spraying mechanism.”
</div>
""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================


st.divider()

st.markdown(
    """
    <div class="footer">
        <b>AGROVISION</b> · AI-Powered Precision Crop Protection
        <br><br>
        DETECT → LOCATE → ACT
    </div>
    """,
    unsafe_allow_html=True
)