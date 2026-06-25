# QuPath Tile And Mask Export Workflow

This workflow is for testing export from a few annotated regions before building
a full training set.

## Goal

For each training tile, export a pair:

```text
raw/<tile_id>.png
masks/<tile_id>.png
```

The raw tile is the microscope image. The mask tile is a single-channel image:

```text
0   = background / not-charcoal
255 = charcoal
```

## QuPath Setup

1. Open your QuPath project.
2. Create an annotation class named exactly `Charcoal`.
3. Draw a rectangular annotation around a manageable training area.
4. Fully annotate every charcoal fragment inside that rectangle as `Charcoal`.
5. Select the rectangular training-region annotation.
6. Run `scripts/qupath/export_selected_region_tiles_and_masks.groovy` in QuPath's script editor.

The selected rectangle is the export area. You do not need to know the tile
boundaries while annotating. You only need to fully annotate all charcoal inside
the selected export rectangle.

## Output Location

The script writes into your QuPath project folder:

```text
charcoal_detector_export/
  <image_name>/
    raw/
    masks/
```

## First Test

For the first test, use a small rectangle and leave the script settings at:

```groovy
int tileSize = 512
int overlap = 0
double downsample = 1.0
```

Check that:

- raw and mask folders contain the same number of files
- filenames match between raw and masks
- masks are black where there is no charcoal
- charcoal annotations are white in the masks
- the mask dimensions match the raw tile dimensions

## Notes

- For training, fully labeled small regions are better than partially labeled
  large regions.
- Empty/no-charcoal tiles are useful, but only include them if you are confident
  they really do not contain charcoal.
- If the selected rectangle is huge, the script may export many tiles. Start
  small.

