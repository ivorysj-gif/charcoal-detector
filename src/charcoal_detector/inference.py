from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure, morphology


@dataclass(frozen=True)
class DetectionSettings:
    darkness_threshold: int = 95
    min_area_px: int = 80
    max_area_px: int = 500_000
    closing_radius: int = 2
    pixel_size: float | None = None
    pixel_unit: str = "px"


@dataclass(frozen=True)
class FragmentMeasurement:
    fragment_id: int
    area_px: int
    centroid_x: float
    centroid_y: float
    bbox_min_x: int
    bbox_min_y: int
    bbox_max_x: int
    bbox_max_y: int
    major_axis_length_px: float
    minor_axis_length_px: float
    perimeter_px: float
    area_calibrated: float | None = None
    length_calibrated: float | None = None
    unit: str = "px"

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "fragment_id": self.fragment_id,
            "area_px": self.area_px,
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "bbox_min_x": self.bbox_min_x,
            "bbox_min_y": self.bbox_min_y,
            "bbox_max_x": self.bbox_max_x,
            "bbox_max_y": self.bbox_max_y,
            "major_axis_length_px": round(self.major_axis_length_px, 2),
            "minor_axis_length_px": round(self.minor_axis_length_px, 2),
            "perimeter_px": round(self.perimeter_px, 2),
            "area_calibrated": None
            if self.area_calibrated is None
            else round(self.area_calibrated, 4),
            "length_calibrated": None
            if self.length_calibrated is None
            else round(self.length_calibrated, 4),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class DetectionResult:
    mask: np.ndarray
    annotated_image: Image.Image
    measurements: list[FragmentMeasurement]


def detect_charcoal(image: Image.Image, settings: DetectionSettings) -> DetectionResult:
    rgb = np.asarray(image.convert("RGB"))
    mask = _baseline_dark_particle_mask(rgb, settings)
    labels = measure.label(mask)
    measurements = _measure_fragments(labels, settings)
    annotated = _draw_annotations(image, labels, measurements)
    return DetectionResult(mask=mask, annotated_image=annotated, measurements=measurements)


def _baseline_dark_particle_mask(rgb: np.ndarray, settings: DetectionSettings) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray <= settings.darkness_threshold

    if settings.closing_radius > 0:
        footprint = morphology.disk(settings.closing_radius)
        mask = morphology.binary_closing(mask, footprint)

    mask = morphology.remove_small_objects(mask, settings.min_area_px)
    labels = measure.label(mask)

    filtered = np.zeros_like(mask, dtype=bool)
    for region in measure.regionprops(labels):
        if settings.min_area_px <= region.area <= settings.max_area_px:
            filtered[labels == region.label] = True

    return filtered


def _measure_fragments(
    labels: np.ndarray,
    settings: DetectionSettings,
) -> list[FragmentMeasurement]:
    measurements: list[FragmentMeasurement] = []
    scale = settings.pixel_size
    unit = settings.pixel_unit if scale else "px"

    for index, region in enumerate(measure.regionprops(labels), start=1):
        min_row, min_col, max_row, max_col = region.bbox
        area_calibrated = region.area * scale * scale if scale else None
        length_calibrated = region.major_axis_length * scale if scale else None
        measurements.append(
            FragmentMeasurement(
                fragment_id=index,
                area_px=int(region.area),
                centroid_x=float(region.centroid[1]),
                centroid_y=float(region.centroid[0]),
                bbox_min_x=int(min_col),
                bbox_min_y=int(min_row),
                bbox_max_x=int(max_col),
                bbox_max_y=int(max_row),
                major_axis_length_px=float(region.major_axis_length),
                minor_axis_length_px=float(region.minor_axis_length),
                perimeter_px=float(region.perimeter),
                area_calibrated=area_calibrated,
                length_calibrated=length_calibrated,
                unit=unit,
            )
        )

    return measurements


def _draw_annotations(
    image: Image.Image,
    labels: np.ndarray,
    measurements: list[FragmentMeasurement],
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    for measurement in measurements:
        fragment_mask = labels == measurement.fragment_id
        contours = measure.find_contours(fragment_mask.astype(np.uint8), 0.5)
        for contour in contours:
            points = [(float(col), float(row)) for row, col in contour]
            if len(points) > 1:
                draw.line(points, fill=(255, 80, 40, 230), width=2)

        draw.rectangle(
            [
                measurement.bbox_min_x,
                measurement.bbox_min_y,
                measurement.bbox_max_x,
                measurement.bbox_max_y,
            ],
            outline=(255, 180, 0, 220),
            width=1,
        )
        draw.text(
            (measurement.bbox_min_x, max(0, measurement.bbox_min_y - 12)),
            str(measurement.fragment_id),
            fill=(255, 80, 40, 255),
        )

    return annotated

