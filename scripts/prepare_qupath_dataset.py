from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairRecord:
    source_id: str
    stem: str
    raw_path: Path
    mask_path: Path
    width: int
    height: int
    nonzero_mask_pixels: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and stage QuPath raw/mask tile exports."
    )
    parser.add_argument(
        "export_root",
        type=Path,
        help="Path to QuPath charcoal_detector_export folder.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data"),
        help="Destination dataset root. Defaults to ./data.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy validated pairs into data/raw and data/masks.",
    )
    args = parser.parse_args()

    records = collect_pair_records(args.export_root)
    print_summary(records)
    write_manifest(records, args.dataset_root / "manifest.csv")

    if args.copy:
        copy_pairs(records, args.dataset_root)
        print(f"Copied {len(records)} pairs into {args.dataset_root}")
    else:
        print("Dry run only. Add --copy to copy files into the dataset folder.")


def collect_pair_records(export_root: Path) -> list[PairRecord]:
    if not export_root.exists():
        raise FileNotFoundError(f"Export root does not exist: {export_root}")

    records: list[PairRecord] = []
    for source_dir in sorted(path for path in export_root.iterdir() if path.is_dir()):
        raw_dir = source_dir / "raw"
        mask_dir = source_dir / "masks"
        if not raw_dir.exists() or not mask_dir.exists():
            continue

        raw_files = {
            path.stem: path
            for path in sorted(raw_dir.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }
        mask_files = {
            path.stem: path
            for path in sorted(mask_dir.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }

        missing_masks = sorted(set(raw_files) - set(mask_files))
        missing_raw = sorted(set(mask_files) - set(raw_files))
        if missing_masks:
            print(f"WARNING: {source_dir.name} has {len(missing_masks)} raw files without masks.")
        if missing_raw:
            print(f"WARNING: {source_dir.name} has {len(missing_raw)} masks without raw files.")

        for stem in sorted(set(raw_files) & set(mask_files)):
            raw_path = raw_files[stem]
            mask_path = mask_files[stem]
            raw_size = image_size(raw_path)
            mask_size = image_size(mask_path)
            if raw_size != mask_size:
                print(f"WARNING: size mismatch for {stem}: raw={raw_size}, mask={mask_size}")
                continue

            records.append(
                PairRecord(
                    source_id=source_dir.name,
                    stem=stem,
                    raw_path=raw_path,
                    mask_path=mask_path,
                    width=raw_size[0],
                    height=raw_size[1],
                    nonzero_mask_pixels=count_nonzero_pixels(mask_path),
                )
            )

    return records


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def count_nonzero_pixels(path: Path) -> int:
    with Image.open(path).convert("L") as image:
        return sum(1 for pixel in image.getdata() if pixel != 0)


def print_summary(records: list[PairRecord]) -> None:
    total = len(records)
    positive = sum(1 for record in records if record.nonzero_mask_pixels > 0)
    empty = total - positive
    sources = sorted({record.source_id for record in records})
    print(f"Sources: {len(sources)}")
    print(f"Paired raw/mask tiles: {total}")
    print(f"Positive masks: {positive}")
    print(f"Empty masks: {empty}")
    if records:
        widths = sorted({record.width for record in records})
        heights = sorted({record.height for record in records})
        print(f"Widths: {widths}")
        print(f"Heights: {heights}")


def write_manifest(records: list[PairRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_id",
                "stem",
                "raw_path",
                "mask_path",
                "width",
                "height",
                "nonzero_mask_pixels",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_id": record.source_id,
                    "stem": record.stem,
                    "raw_path": record.raw_path,
                    "mask_path": record.mask_path,
                    "width": record.width,
                    "height": record.height,
                    "nonzero_mask_pixels": record.nonzero_mask_pixels,
                }
            )
    print(f"Wrote manifest: {path}")


def copy_pairs(records: list[PairRecord], dataset_root: Path) -> None:
    raw_dir = dataset_root / "raw"
    mask_dir = dataset_root / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        output_stem = f"{safe_name(record.source_id)}__{record.stem}"
        shutil.copy2(record.raw_path, raw_dir / f"{output_stem}{record.raw_path.suffix.lower()}")
        shutil.copy2(record.mask_path, mask_dir / f"{output_stem}.png")


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


if __name__ == "__main__":
    main()

