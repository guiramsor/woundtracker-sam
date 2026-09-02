package es.us.universidad.woundtracker;

import ij.IJ;
import ij.ImagePlus;
import ij.plugin.ZProjector;
import java.util.Locale;

public class ImageProcessorService {

    public static ImagePlus performZProjection(ImagePlus inputImage, String projectionType) {
        if (inputImage == null) return null;
        
        // Solo cuenta Z: con una rodaja no hay nada que proyectar.
        if (inputImage.getNSlices() <= 1) {
            return inputImage;
        }

        IJ.showStatus("WoundTracker: Proyección Z...");

        int method = getZProjectionMethodConstant(projectionType);
        ZProjector zp = new ZProjector(inputImage);
        zp.setMethod(method);
        zp.setStartSlice(1);
        zp.setStopSlice(inputImage.getNSlices());
        zp.doHyperStackProjection(true);
        ImagePlus proj = zp.getProjection();
        proj.setTitle(inputImage.getTitle() + " - " + projectionType);

        return proj;
    }

    public static void applyAutoContrast(ImagePlus image, double saturatedPercent) {
        if (image == null) return;

        IJ.showStatus("WoundTracker: Optimizando contraste...");

        String opts = String.format(Locale.US, "saturated=%.2f stack", saturatedPercent);
        IJ.run(image, "Enhance Contrast...", opts);
    }

    private static int getZProjectionMethodConstant(String projectionTypeName) {
        switch (projectionTypeName) {
            case "Average Intensity":  return ZProjector.AVG_METHOD;
            case "Sum Slices":         return ZProjector.SUM_METHOD;
            case "Minimum Intensity":  return ZProjector.MIN_METHOD;
            case "Standard Deviation": return ZProjector.SD_METHOD;
            case "Median":             return ZProjector.MEDIAN_METHOD;
            case "Maximum Intensity":
            default:                   return ZProjector.MAX_METHOD;
        }
    }
}