from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

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


def draw_lycopodium_points(
    image: Image.Image,
    points: list[dict[str, float]],
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    font = review_font(annotated)
    radius = max(8, min(annotated.size) // 80)

    for index, point in enumerate(points, start=1):
        x = int(point["x_norm"] * annotated.width)
        y = int(point["y_norm"] * annotated.height)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=(40, 220, 70, 255),
            width=max(3, radius // 3),
        )
        label = f"L{index}"
        box = draw.textbbox((x + radius + 4, y - radius), label, font=font)
        draw.rectangle(
            [box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3],
            fill=(255, 255, 255, 220),
            outline=(40, 220, 70, 255),
        )
        draw.text((x + radius + 4, y - radius), label, font=font, fill=(20, 150, 45, 255))

    return annotated


def review_font(image: Image.Image) -> ImageFont.ImageFont:
    font_size = max(16, min(34, min(image.size) // 32))
    for font_path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_for_click_review(image: Image.Image, max_width: int = 950) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / image.width
    return image.resize((max_width, int(image.height * scale)), Image.Resampling.BILINEAR)


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
    pixel_size = st.number_input(
        "Microns per pixel",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
    )
    pixel_unit = "um"

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

left, right = st.columns([1.2, 1])

with right:
    st.subheader("Measurements")
    excluded_ids: list[int] = []
    filtered_measurements = result.measurements

    if result.measurements:
        review_table = pd.DataFrame([m.as_dict() for m in result.measurements])
        review_table.insert(0, "include", True)
        edited_table = st.data_editor(
            review_table,
            use_container_width=True,
            hide_index=True,
            disabled=[column for column in review_table.columns if column != "include"],
            column_config={
                "include": st.column_config.CheckboxColumn(
                    "Include",
                    help="Uncheck false positives before exporting.",
                    default=True,
                ),
                "fragment_id": st.column_config.NumberColumn("ID", format="%d"),
            },
            key=(
                f"measurement_review_table_{uploaded.name}_{mode}_"
                f"{model_threshold}_{threshold}_{min_area}_{max_area}_{closing_radius}_"
                f"{len(result.measurements)}"
            ),
        )
        included_ids = set(
            edited_table.loc[edited_table["include"], "fragment_id"].astype(int).tolist()
        )
        excluded_ids = [
            measurement.fragment_id
            for measurement in result.measurements
            if measurement.fragment_id not in included_ids
        ]
        filtered_measurements = [
            measurement
            for measurement in result.measurements
            if measurement.fragment_id in included_ids
        ]

    count_columns = st.columns(3)
    count_columns[0].metric("Detected", len(result.measurements))
    count_columns[1].metric("Excluded", len(excluded_ids))
    count_columns[2].metric("Retained", len(filtered_measurements))

    if filtered_measurements:
        table = pd.DataFrame([m.as_dict() for m in filtered_measurements])

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

annotated_image = annotate_measurements(image, result.mask, filtered_measurements)
image_session_key = f"{uploaded.name}_{image.width}x{image.height}"
points_key = f"lycopodium_points_{image_session_key}"
last_click_key = f"last_lycopodium_click_{image_session_key}"
if points_key not in st.session_state:
    st.session_state[points_key] = []

with left:
    st.subheader("Annotated Image")
    if streamlit_image_coordinates is not None:
        lycopodium_image = draw_lycopodium_points(
            annotated_image,
            st.session_state[points_key],
        )
        click_image = resize_for_click_review(lycopodium_image)
        clicked = streamlit_image_coordinates(
            click_image,
            key=f"lycopodium_click_{image_session_key}",
        )
        if clicked is not None:
            click_signature = (clicked["x"], clicked["y"])
            if st.session_state.get(last_click_key) != click_signature:
                st.session_state[last_click_key] = click_signature
                st.session_state[points_key].append(
                    {
                        "x_norm": clicked["x"] / click_image.width,
                        "y_norm": clicked["y"] / click_image.height,
                    }
                )
                st.rerun()
    else:
        lycopodium_image = annotated_image
        st.image(lycopodium_image, use_container_width=True)

    review_columns = st.columns(3)
    review_columns[0].metric("Lycopodium", len(st.session_state[points_key]))
    if review_columns[1].button("Undo Lycopodium", disabled=not st.session_state[points_key]):
        st.session_state[points_key].pop()
        st.rerun()
    if review_columns[2].button("Clear Lycopodium", disabled=not st.session_state[points_key]):
        st.session_state[points_key] = []
        st.session_state.pop(last_click_key, None)
        st.rerun()

    if streamlit_image_coordinates is None:
        manual_lycopodium_count = st.number_input(
            "Manual Lycopodium count",
            min_value=0,
            value=len(st.session_state[points_key]),
            step=1,
        )
        st.session_state[points_key] = [
            {"x_norm": 0.0, "y_norm": 0.0}
            for _ in range(int(manual_lycopodium_count))
        ]

    if result.probability_map is not None:
        st.subheader("Model Probability")
        st.image(result.probability_map, clamp=True, use_container_width=True)

buffer = BytesIO()
download_image = draw_lycopodium_points(annotated_image, st.session_state[points_key])
download_image.save(buffer, format="PNG")
st.download_button(
    "Download annotated image",
    buffer.getvalue(),
    file_name="charcoal_annotated.png",
    mime="image/png",
)

summary_table = pd.DataFrame(
    [
        {
            "charcoal_retained_count": len(filtered_measurements),
            "charcoal_excluded_count": len(excluded_ids),
            "lycopodium_count": len(st.session_state[points_key]),
            "microns_per_pixel": float(pixel_size) if pixel_size else None,
        }
    ]
)
st.download_button(
    "Download summary CSV",
    summary_table.to_csv(index=False).encode("utf-8"),
    file_name="charcoal_summary.csv",
    mime="text/csv",
)
