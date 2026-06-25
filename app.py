from __future__ import annotations

import math
import random
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
    FragmentMeasurement,
    annotate_measurements,
    detect_charcoal,
    detect_charcoal_with_model,
)
from src.charcoal_detector.models import load_model


Image.MAX_IMAGE_PIXELS = None
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
        draw.text(
            (x + radius + 4, y - radius),
            label,
            font=font,
            fill=(20, 150, 45, 255),
        )

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


def generate_tiles(width: int, height: int, tile_size: int) -> list[dict[str, int]]:
    tiles: list[dict[str, int]] = []
    tile_number = 1
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tiles.append(
                {
                    "tile_id": tile_number,
                    "x": x,
                    "y": y,
                    "width": min(tile_size, width - x),
                    "height": min(tile_size, height - y),
                }
            )
            tile_number += 1
    return tiles


def sample_tiles(
    tiles: list[dict[str, int]],
    review_percent: int,
    method: str,
    seed: int,
) -> list[dict[str, int]]:
    if review_percent >= 100:
        return tiles

    count = max(1, math.ceil(len(tiles) * review_percent / 100))
    if count >= len(tiles):
        return tiles

    if method == "Random":
        rng = random.Random(seed)
        return sorted(rng.sample(tiles, count), key=lambda tile: tile["tile_id"])

    if count == 1:
        return [tiles[len(tiles) // 2]]
    indices = sorted(
        {round(index * (len(tiles) - 1) / (count - 1)) for index in range(count)}
    )
    return [tiles[index] for index in indices]


def detect_image(
    image: Image.Image,
    mode: str,
    model_available: bool,
    settings: DetectionSettings,
):
    if mode == "Trained model" and model_available:
        model = load_cached_model(str(DEFAULT_MODEL_PATH))
        return detect_charcoal_with_model(image, model, settings)
    return detect_charcoal(image, settings)


def reviewed_measurements_table(
    measurements: list[FragmentMeasurement],
    table_key: str,
) -> tuple[list[FragmentMeasurement], list[int]]:
    if not measurements:
        return [], []

    review_table = pd.DataFrame([measurement.as_dict() for measurement in measurements])
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
        key=table_key,
    )
    included_ids = set(
        edited_table.loc[edited_table["include"], "fragment_id"].astype(int).tolist()
    )
    excluded_ids = [
        measurement.fragment_id
        for measurement in measurements
        if measurement.fragment_id not in included_ids
    ]
    filtered = [
        measurement
        for measurement in measurements
        if measurement.fragment_id in included_ids
    ]
    return filtered, excluded_ids


def lycopodium_click_review(
    annotated_image: Image.Image,
    points_key: str,
    click_key: str,
    last_click_key: str,
) -> Image.Image:
    if points_key not in st.session_state:
        st.session_state[points_key] = []

    if streamlit_image_coordinates is not None:
        lycopodium_image = draw_lycopodium_points(
            annotated_image,
            st.session_state[points_key],
        )
        click_image = resize_for_click_review(lycopodium_image)
        clicked = streamlit_image_coordinates(click_image, key=click_key)
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
        return lycopodium_image

    st.image(annotated_image, use_container_width=True)
    manual_lycopodium_count = st.number_input(
        "Manual Lycopodium count",
        min_value=0,
        value=len(st.session_state[points_key]),
        step=1,
        key=f"manual_{points_key}",
    )
    st.session_state[points_key] = [
        {"x_norm": 0.0, "y_norm": 0.0}
        for _ in range(int(manual_lycopodium_count))
    ]
    return annotated_image


def lycopodium_controls(points_key: str, last_click_key: str) -> None:
    review_columns = st.columns(3)
    review_columns[0].metric("Lycopodium", len(st.session_state.get(points_key, [])))
    if review_columns[1].button(
        "Undo Lycopodium",
        disabled=not st.session_state.get(points_key, []),
        key=f"undo_{points_key}",
    ):
        st.session_state[points_key].pop()
        st.rerun()
    if review_columns[2].button(
        "Clear Lycopodium",
        disabled=not st.session_state.get(points_key, []),
        key=f"clear_{points_key}",
    ):
        st.session_state[points_key] = []
        st.session_state.pop(last_click_key, None)
        st.rerun()


def global_fragment_rows(
    measurements: list[FragmentMeasurement],
    tile: dict[str, int] | None,
    slide_id: str,
) -> list[dict[str, object]]:
    rows = []
    for measurement in measurements:
        row = measurement.as_dict()
        if tile is not None:
            row.update(
                {
                    "slide_id": slide_id,
                    "tile_id": tile["tile_id"],
                    "tile_x": tile["x"],
                    "tile_y": tile["y"],
                    "global_fragment_id": f"{tile['tile_id']}_{measurement.fragment_id}",
                    "centroid_global_x": measurement.centroid_x + tile["x"],
                    "centroid_global_y": measurement.centroid_y + tile["y"],
                    "bbox_global_min_x": measurement.bbox_min_x + tile["x"],
                    "bbox_global_min_y": measurement.bbox_min_y + tile["y"],
                    "bbox_global_max_x": measurement.bbox_max_x + tile["x"],
                    "bbox_global_max_y": measurement.bbox_max_y + tile["y"],
                }
            )
        else:
            row.update({"slide_id": slide_id})
        rows.append(row)
    return rows


def tile_summary_row(
    image: Image.Image,
    tile: dict[str, int],
    measurements: list[FragmentMeasurement],
    excluded_ids: list[int],
    lycopodium_count: int,
    microns_per_pixel: float | None,
) -> dict[str, object]:
    charcoal_area_px = sum(measurement.area_px for measurement in measurements)
    reviewed_area_px = tile["width"] * tile["height"]
    row: dict[str, object] = {
        "tile_id": tile["tile_id"],
        "tile_x": tile["x"],
        "tile_y": tile["y"],
        "tile_width": tile["width"],
        "tile_height": tile["height"],
        "reviewed_area_px": reviewed_area_px,
        "charcoal_retained_count": len(measurements),
        "charcoal_excluded_count": len(excluded_ids),
        "charcoal_area_px": charcoal_area_px,
        "lycopodium_count": lycopodium_count,
    }
    if microns_per_pixel:
        row["reviewed_area_um2"] = reviewed_area_px * microns_per_pixel * microns_per_pixel
        row["charcoal_area_um2"] = charcoal_area_px * microns_per_pixel * microns_per_pixel
    return row


with st.sidebar:
    st.header("Detection")
    workflow = st.radio("Workflow", ["Single image review", "Slide tile review"])
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

    if workflow == "Slide tile review":
        st.header("Tile Sampling")
        tile_size = st.select_slider(
            "Tile size",
            [1024, 2048, 4096, 8192],
            value=4096,
        )
        review_percent = st.select_slider(
            "Tiles to review",
            [5, 10, 25, 50, 100],
            value=10,
            format_func=lambda value: f"{value}%",
        )
        sampling_method = st.radio("Sampling", ["Systematic grid", "Random"])
        random_seed = st.number_input("Random seed", 0, 999999, 13, 1)

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

if workflow == "Slide tile review":
    all_tiles = generate_tiles(image.width, image.height, int(tile_size))
    selected_tiles = sample_tiles(
        all_tiles,
        int(review_percent),
        "Random" if sampling_method == "Random" else "Systematic grid",
        int(random_seed),
    )
    session_key = (
        f"slide_{uploaded.name}_{image.width}x{image.height}_{tile_size}_"
        f"{review_percent}_{sampling_method}_{random_seed}"
    )
    index_key = f"{session_key}_index"
    reviews_key = f"{session_key}_reviews"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    if reviews_key not in st.session_state:
        st.session_state[reviews_key] = {}

    st.caption(
        f"{len(all_tiles)} total tiles; {len(selected_tiles)} selected for review "
        f"({review_percent}%)."
    )

    current_index = min(st.session_state[index_key], len(selected_tiles) - 1)
    st.session_state[index_key] = current_index
    tile = selected_tiles[current_index]
    tile_image = image.crop(
        (tile["x"], tile["y"], tile["x"] + tile["width"], tile["y"] + tile["height"])
    )
    result = detect_image(tile_image, mode, model_available, settings)

    left, right = st.columns([1.2, 1])
    tile_key = f"{session_key}_tile_{tile['tile_id']}"
    points_key = f"{tile_key}_lycopodium_points"
    last_click_key = f"{tile_key}_last_click"

    with right:
        st.subheader("Tile Review")
        st.write(
            f"Tile {current_index + 1} of {len(selected_tiles)} "
            f"(ID {tile['tile_id']}, x={tile['x']}, y={tile['y']})"
        )
        filtered_measurements, excluded_ids = reviewed_measurements_table(
            result.measurements,
            f"{tile_key}_measurement_table_{len(result.measurements)}",
        )
        counts = st.columns(3)
        counts[0].metric("Detected", len(result.measurements))
        counts[1].metric("Excluded", len(excluded_ids))
        counts[2].metric("Retained", len(filtered_measurements))

        reviewed_count = len(st.session_state[reviews_key])
        st.metric("Saved reviewed tiles", f"{reviewed_count}/{len(selected_tiles)}")

        def save_tile_review() -> None:
            lycopodium_count = len(st.session_state.get(points_key, []))
            st.session_state[reviews_key][tile["tile_id"]] = {
                "fragment_rows": global_fragment_rows(
                    filtered_measurements,
                    tile,
                    uploaded.name,
                ),
                "tile_summary": tile_summary_row(
                    image,
                    tile,
                    filtered_measurements,
                    excluded_ids,
                    lycopodium_count,
                    float(pixel_size) if pixel_size else None,
                ),
            }

        nav_columns = st.columns(4)
        if nav_columns[0].button("Previous", disabled=current_index == 0):
            st.session_state[index_key] = max(0, current_index - 1)
            st.rerun()
        if nav_columns[1].button("Save tile review"):
            save_tile_review()
            st.rerun()
        if nav_columns[2].button("Save & next"):
            save_tile_review()
            st.session_state[index_key] = min(len(selected_tiles) - 1, current_index + 1)
            st.rerun()
        if nav_columns[3].button(
            "Next",
            disabled=current_index >= len(selected_tiles) - 1,
        ):
            st.session_state[index_key] = min(len(selected_tiles) - 1, current_index + 1)
            st.rerun()

    annotated_image = annotate_measurements(tile_image, result.mask, filtered_measurements)

    with left:
        st.subheader("Annotated Tile")
        lycopodium_image = lycopodium_click_review(
            annotated_image,
            points_key,
            f"{tile_key}_lycopodium_click",
            last_click_key,
        )
        lycopodium_controls(points_key, last_click_key)
        if result.probability_map is not None:
            st.subheader("Model Probability")
            st.image(result.probability_map, clamp=True, use_container_width=True)

    reviews = list(st.session_state[reviews_key].values())
    fragment_rows = [row for review in reviews for row in review["fragment_rows"]]
    tile_rows = [review["tile_summary"] for review in reviews]

    st.subheader("Slide Exports")
    fragment_table = pd.DataFrame(fragment_rows)
    tile_table = pd.DataFrame(tile_rows)
    total_reviewed_area_px = int(tile_table["reviewed_area_px"].sum()) if not tile_table.empty else 0
    total_charcoal_area_px = int(tile_table["charcoal_area_px"].sum()) if not tile_table.empty else 0
    total_lycopodium = int(tile_table["lycopodium_count"].sum()) if not tile_table.empty else 0
    summary = {
        "slide_id": uploaded.name,
        "image_width_px": image.width,
        "image_height_px": image.height,
        "total_tiles": len(all_tiles),
        "selected_tiles": len(selected_tiles),
        "reviewed_tiles": len(tile_rows),
        "review_percent_setting": int(review_percent),
        "sampling_method": sampling_method,
        "tile_size_px": int(tile_size),
        "reviewed_area_px": total_reviewed_area_px,
        "total_image_area_px": image.width * image.height,
        "charcoal_retained_count": len(fragment_rows),
        "charcoal_area_px": total_charcoal_area_px,
        "lycopodium_count": total_lycopodium,
        "microns_per_pixel": float(pixel_size) if pixel_size else None,
    }
    if pixel_size:
        summary["reviewed_area_um2"] = total_reviewed_area_px * pixel_size * pixel_size
        summary["charcoal_area_um2"] = total_charcoal_area_px * pixel_size * pixel_size
    summary_table = pd.DataFrame([summary])

    downloads = st.columns(3)
    downloads[0].download_button(
        "Download fragment CSV",
        fragment_table.to_csv(index=False).encode("utf-8"),
        file_name="slide_charcoal_fragments.csv",
        mime="text/csv",
        disabled=fragment_table.empty,
    )
    downloads[1].download_button(
        "Download tile CSV",
        tile_table.to_csv(index=False).encode("utf-8"),
        file_name="slide_tile_summary.csv",
        mime="text/csv",
        disabled=tile_table.empty,
    )
    downloads[2].download_button(
        "Download slide summary CSV",
        summary_table.to_csv(index=False).encode("utf-8"),
        file_name="slide_summary.csv",
        mime="text/csv",
        disabled=tile_table.empty,
    )
    st.stop()

result = detect_image(image, mode, model_available, settings)
left, right = st.columns([1.2, 1])

with right:
    st.subheader("Measurements")
    filtered_measurements, excluded_ids = reviewed_measurements_table(
        result.measurements,
        (
            f"measurement_review_table_{uploaded.name}_{mode}_"
            f"{model_threshold}_{threshold}_{min_area}_{max_area}_{closing_radius}_"
            f"{len(result.measurements)}"
        ),
    )

    count_columns = st.columns(3)
    count_columns[0].metric("Detected", len(result.measurements))
    count_columns[1].metric("Excluded", len(excluded_ids))
    count_columns[2].metric("Retained", len(filtered_measurements))

    if filtered_measurements:
        table = pd.DataFrame([m.as_dict() for m in filtered_measurements])
        st.download_button(
            "Download CSV",
            table.to_csv(index=False).encode("utf-8"),
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

with left:
    st.subheader("Annotated Image")
    lycopodium_image = lycopodium_click_review(
        annotated_image,
        points_key,
        f"lycopodium_click_{image_session_key}",
        last_click_key,
    )
    lycopodium_controls(points_key, last_click_key)

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
