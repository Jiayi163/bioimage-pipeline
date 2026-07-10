#@ String imagePath
#@ String outputCsv
#@ double radius
#@ double threshold

import fiji.plugin.trackmate.Model
import fiji.plugin.trackmate.Settings
import fiji.plugin.trackmate.TrackMate
import fiji.plugin.trackmate.detection.LogDetectorFactory
import fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory
import fiji.plugin.trackmate.io.CSVExporter

imp = IJ.openImage(imagePath)
if (imp == null) throw new RuntimeException("Failed to open " + imagePath)

settings = new Settings(imp)
settings.detectorFactory = new LogDetectorFactory()
settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
settings.detectorSettings['RADIUS'] = radius
settings.detectorSettings['THRESHOLD'] = threshold
settings.detectorSettings['DO_SUBPIXEL_LOCALIZATION'] = true

settings.trackerFactory = new SparseLAPTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
settings.trackerSettings['LINKING_MAX_DISTANCE'] = 0.0
settings.trackerSettings['MAX_FRAME_GAP'] = 0

model = new Model()
trackmate = new TrackMate(model, settings)
if (!trackmate.checkInput() || !trackmate.process()) {
    throw new RuntimeException(trackmate.getErrorMessage())
}

CSVExporter.exportSpots(new File(outputCsv), model, false)
imp.close()
