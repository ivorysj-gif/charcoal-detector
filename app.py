from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.charcoal_detector.inference import (
    DetectionSettings,
    annotate_measurements,
    detect_charcoal,
    detect_charcoal_with_model,
)
from src.charcoal_detector.models import load_model


st.set_page_config(page_title="Charcoal Detector", layout="wide")

st.title("Charcoal Detector")

DEFAULT_MODEL_PATH = Path("models/charcoal_tiny_unet.pt")


@st.cache_resource
def load_cached_model(checkpoint_path: str):
    return load_model(checkpoint_path)

with st.sidebar:
    st.header("Detection")
    model_available = DEFAULT_MODEL_PATH.exists()
    default_mode_index = 0 if model_available else 1
    mode = st.radio(
        "Mode",
        ["Trained model", "Dark-particle baseline"],
        index=default_mode_index,
        disabled=not model_available,
    )
    if not model_available:
        st.caption("No trained model checkpoint found; using baseline mode.")

    model_threshold = st.slider("Model threshold", 0.05, 0.99, 0.85, 0.01)
    model_image_size = st.select_slider("Model image size", [128, 256, 512], value=256)
    threshold = st.slider("Baseline darkness threshold", 0, 255, 95, 1)
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
    model_threshold=float(model_threshold),
    model_image_size=int(model_image_size),
)

if mode == "Trained model" and model_available:
    model = load_cached_model(str(DEFAULT_MODEL_PATH))
    result = detect_charcoal_with_model(image, model, settings)
else:
    result = detect_charcoal(image, settings)

excluded_ids: list[int] = []
if result.measurements:
    measurement_options = {
        (
            f"{measurement.fragment_id}: "
            f"{measurement.area_px} px, "
            f"({measurement.centroid_x:.0f}, {measurement.centroid_y:.0f})"
        ): measurement.fragment_id
        for measurement in result.measurements
    }
    with st.sidebar:
        st.header("Review")
        excluded_labels = st.multiselect(
            "Exclude fragments",
            options=list(measurement_options),
            help="Excluded fragments are removed from the overlay, table, and CSV export.",
        )
    excluded_ids = [measurement_options[label] for label in excluded_labels]

filtered_measurements = [
    measurement
    for measurement in result.measurements
    if measurement.fragment_id not in set(excluded_ids)
]
annotated_image = annotate_measurements(image, result.mask, filtered_measurements)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Annotated Image")
    st.image(annotated_image, use_container_width=True)
    if result.probability_map is not None:
        st.subheader("Model Probability")
        st.image(result.probability_map, clamp=True, use_container_width=True)

with right:
    st.subheader("Measurements")
    count_columns = st.columns(3)
    count_columns[0].metric("Detected", len(result.measurements))
    count_columns[1].metric("Excluded", len(excluded_ids))
    count_columns[2].metric("Retained", len(filtered_measurements))

    if filtered_measurements:
        table = pd.DataFrame([m.as_dict() for m in filtered_measurements])
        st.dataframe(table, use_container_width=True, hide_index=True)

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            file_name="charcoal_measurements.csv",
            mime="text/csv",
        )
    else:
        if result.measurements:
            st.warning("All detected fragments are currently excluded.")
        else:
            st.warning("No fragments detected with the current settings.")

buffer = BytesIO()
annotated_image.save(buffer, format="PNG")
st.download_button(
    "Download annotated image",
    buffer.getvalue(),
    file_name="charcoal_annotated.png",
    mime="image/png",
)
