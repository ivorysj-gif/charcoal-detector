from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from src.charcoal_detector.inference import DetectionSettings, detect_charcoal


st.set_page_config(page_title="Charcoal Detector", layout="wide")

st.title("Charcoal Detector")

with st.sidebar:
    st.header("Detection")
    threshold = st.slider("Darkness threshold", 0, 255, 95, 1)
    min_area = st.number_input("Minimum fragment area (px)", 1, 100000, 80, 10)
    max_area = st.number_input("Maximum fragment area (px)", 1, 10000000, 500000, 1000)
    closing_radius = st.slider("Boundary smoothing", 0, 15, 2, 1)
    pixel_size = st.number_input("Pixel size", min_value=0.0, value=0.0, step=0.1)
    pixel_unit = st.selectbox("Pixel unit", ["px", "um", "mm"], index=0)

uploaded = st.file_uploader(
    "Upload a microscopy image",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
)

if uploaded is None:
    st.info("Upload an image to run charcoal detection.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
settings = DetectionSettings(
    darkness_threshold=threshold,
    min_area_px=int(min_area),
    max_area_px=int(max_area),
    closing_radius=int(closing_radius),
    pixel_size=float(pixel_size) if pixel_size else None,
    pixel_unit=pixel_unit,
)

result = detect_charcoal(image, settings)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Annotated Image")
    st.image(result.annotated_image, use_container_width=True)

with right:
    st.subheader("Measurements")
    st.metric("Detected fragments", len(result.measurements))
    if result.measurements:
        table = pd.DataFrame([m.as_dict() for m in result.measurements])
        st.dataframe(table, use_container_width=True, hide_index=True)

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            file_name="charcoal_measurements.csv",
            mime="text/csv",
        )
    else:
        st.warning("No fragments detected with the current settings.")

buffer = BytesIO()
result.annotated_image.save(buffer, format="PNG")
st.download_button(
    "Download annotated image",
    buffer.getvalue(),
    file_name="charcoal_annotated.png",
    mime="image/png",
)

