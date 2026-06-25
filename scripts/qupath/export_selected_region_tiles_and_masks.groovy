/*
 * Export raw image tiles and matching binary charcoal masks from the selected
 * QuPath annotation.
 *
 * How to use:
 * 1. In QuPath, create/choose an annotation class named "Charcoal".
 * 2. Draw a rectangular "training region" annotation around a manageable area.
 * 3. Fully annotate all charcoal inside that region as "Charcoal".
 * 4. Select the training region annotation.
 * 5. Run this script from Automate > Show script editor.
 *
 * Output:
 *   <QuPath project>/charcoal_detector_export/<image name>/raw/*.png
 *   <QuPath project>/charcoal_detector_export/<image name>/masks/*.png
 *
 * Masks are single-channel labeled images:
 *   0   = background / not-charcoal
 *   255 = Charcoal
 */

import qupath.lib.common.GeneralTools
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.RegionRequest

// --------------------------
// Settings to adjust
// --------------------------
int tileSize = 512
int overlap = 0
double downsample = 1.0
String positiveClass = 'Charcoal'
String exportRootName = 'charcoal_detector_export'
String imageExtension = '.png'
String maskExtension = '.png'
boolean skipPartialTiles = true

// --------------------------
// Setup
// --------------------------
def imageData = getCurrentImageData()
def server = imageData.getServer()
def selected = getSelectedObject()

if (selected == null || selected.getROI() == null) {
    throw new IllegalArgumentException('Select one rectangular training-region annotation before running this script.')
}

def roi = selected.getROI()
def imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
def outputRoot = buildFilePath(PROJECT_BASE_DIR, exportRootName, imageName)
def rawDir = buildFilePath(outputRoot, 'raw')
def maskDir = buildFilePath(outputRoot, 'masks')
mkdirs(rawDir)
mkdirs(maskDir)

def labelServer = new LabeledImageServer.Builder(imageData)
        .backgroundLabel(0, ColorTools.BLACK)
        .downsample(downsample)
        .addLabel(positiveClass, 255)
        .multichannelOutput(false)
        .build()

int xMin = Math.max(0, Math.floor(roi.getBoundsX()) as int)
int yMin = Math.max(0, Math.floor(roi.getBoundsY()) as int)
int xMax = Math.min(server.getWidth(), Math.ceil(roi.getBoundsX() + roi.getBoundsWidth()) as int)
int yMax = Math.min(server.getHeight(), Math.ceil(roi.getBoundsY() + roi.getBoundsHeight()) as int)
int step = tileSize - overlap

if (step <= 0) {
    throw new IllegalArgumentException('overlap must be smaller than tileSize.')
}

int exported = 0

for (int y = yMin; y < yMax; y += step) {
    for (int x = xMin; x < xMax; x += step) {
        int width = Math.min(tileSize, xMax - x)
        int height = Math.min(tileSize, yMax - y)

        if (skipPartialTiles && (width < tileSize || height < tileSize)) {
            continue
        }

        String stem = String.format('%s_x%06d_y%06d', imageName, x, y)
        def rawRequest = RegionRequest.createInstance(server.getPath(), downsample, x, y, width, height)
        def maskRequest = RegionRequest.createInstance(labelServer.getPath(), downsample, x, y, width, height)

        writeImageRegion(server, rawRequest, buildFilePath(rawDir, stem + imageExtension))
        writeImageRegion(labelServer, maskRequest, buildFilePath(maskDir, stem + maskExtension))
        exported++
    }
}

print String.format('Exported %,d raw tile/mask pairs to %s', exported, outputRoot)

