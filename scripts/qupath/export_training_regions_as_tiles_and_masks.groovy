/*
 * Export each "Training Region" annotation as one raw tile plus one matching
 * binary charcoal mask.
 *
 * This script matches a workflow where red Training Region boxes are manually
 * placed around examples, and charcoal inside those boxes is annotated with the
 * class "Charcoal".
 *
 * Output:
 *   <QuPath project>/charcoal_detector_export/<image name>/raw/*.png
 *   <QuPath project>/charcoal_detector_export/<image name>/masks/*.png
 *
 * Masks:
 *   0   = background / not-charcoal
 *   255 = Charcoal
 */

import qupath.lib.common.GeneralTools
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.RegionRequest

// --------------------------
// Settings to adjust
// --------------------------
double downsample = 1.0
String trainingRegionClass = 'Training Region'
String positiveClass = 'Charcoal'
String exportRootName = 'charcoal_detector_export'
String imageExtension = '.png'
String maskExtension = '.png'

// --------------------------
// Setup
// --------------------------
def imageData = getCurrentImageData()
def server = imageData.getServer()
def imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
def outputRoot = buildFilePath(PROJECT_BASE_DIR, exportRootName, imageName)
def rawDir = buildFilePath(outputRoot, 'raw')
def maskDir = buildFilePath(outputRoot, 'masks')
mkdirs(rawDir)
mkdirs(maskDir)

def trainingRegions = getAnnotationObjects().findAll {
    it.getPathClass() != null &&
    it.getPathClass().toString() == trainingRegionClass &&
    it.getROI() != null
}

if (trainingRegions.isEmpty()) {
    throw new IllegalArgumentException("No annotations found with class '${trainingRegionClass}'.")
}

def labelServer = new LabeledImageServer.Builder(imageData)
        .backgroundLabel(0, ColorTools.BLACK)
        .downsample(downsample)
        .addLabel(positiveClass, 255)
        .multichannelOutput(false)
        .build()

int exported = 0

trainingRegions.eachWithIndex { regionObject, index ->
    def roi = regionObject.getROI()
    int x = Math.max(0, Math.floor(roi.getBoundsX()) as int)
    int y = Math.max(0, Math.floor(roi.getBoundsY()) as int)
    int width = Math.min(server.getWidth() - x, Math.ceil(roi.getBoundsWidth()) as int)
    int height = Math.min(server.getHeight() - y, Math.ceil(roi.getBoundsHeight()) as int)

    if (width <= 0 || height <= 0) {
        return
    }

    String stem = String.format('%s_training_region_%04d_x%06d_y%06d', imageName, index + 1, x, y)
    def rawRequest = RegionRequest.createInstance(server.getPath(), downsample, x, y, width, height)
    def maskRequest = RegionRequest.createInstance(labelServer.getPath(), downsample, x, y, width, height)

    writeImageRegion(server, rawRequest, buildFilePath(rawDir, stem + imageExtension))
    writeImageRegion(labelServer, maskRequest, buildFilePath(maskDir, stem + maskExtension))
    exported++
}

print String.format(
        "Exported %,d Training Region raw/mask pairs to %s",
        exported,
        outputRoot
)

