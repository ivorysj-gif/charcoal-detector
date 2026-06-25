# Charcoal Detector Prototype

This is a starter prototype for detecting charcoal fragments in microscopy images.
It is designed to grow in stages:

1. Use the app immediately with a classical segmentation baseline.
2. Add annotated training data as image/mask pairs.
3. Train a segmentation model.
4. Swap the trained model into the app while keeping the same review/export workflow.

## Project Layout

```text
charcoal-detector/
  app.py                       Streamlit upload/review/export app
  requirements.txt             Python dependencies
  data/
    raw/                       Source images
    masks/                     Binary charcoal masks for training
    examples/                  Small test images
  src/
    charcoal_detector/
      inference.py             Detection and measurement pipeline
      image_io.py              Image loading helpers
      models.py                Model loading hooks
      training.py              PyTorch training scaffold
```

## Data Format

Put images in `data/raw` and matching binary masks in `data/masks`.
Use the same filename stem for each pair:

```text
data/raw/sample_001.png
data/masks/sample_001.png
```

Mask convention:

- charcoal pixels: white / non-zero
- background: black / zero

## Run The App

```powershell
pip install -r requirements.txt
streamlit run app.py
```

If `models/charcoal_tiny_unet.pt` exists, the app opens in trained-model mode.
Use the model threshold slider to trade sensitivity against false positives.
Higher thresholds are more conservative; for the first pilot model, `0.85` is a
reasonable starting point.

The app also includes a dark-particle baseline mode. It is intentionally simple:
it segments dark particles, removes tiny specks, measures connected objects, and
exports a CSV plus an annotated preview.

## Recommended Annotation Workflow

Use QuPath, CVAT, Label Studio, or napari to create binary masks for:

- charcoal fragments
- confusing dark non-charcoal debris
- lycopodium, pollen, mineral grains, bubbles, and organic matter
- out-of-focus and partially hidden fragments
- images from different slides, microscopes, magnifications, lighting, and sediment types

The most important early metric is not raw accuracy on familiar images. It is
whether the model generalizes to new slides and hard negatives.
