package es.us.universidad.woundtracker;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import java.awt.Color;
import java.awt.Font;
import java.util.Locale;

public class CellSegmenter {

    /** Lo que produce un frame: las ROIs nuevas y, si se pidió, la binaria. */
    public static final class Resultado {
        public final Roi[] rois;
        public final ImagePlus binaria;

        Resultado(Roi[] rois, ImagePlus binaria) {
            this.rois = rois;
            this.binaria = binaria;
        }

        public boolean hasDetections() {
            return rois != null && rois.length > 0;
        }
    }

    private final WoundHealingTracker config;

    public CellSegmenter(WoundHealingTracker config) {
        this.config = config;
    }

    public Resultado segmentCells(ImagePlus imageToSegment) {
        ImagePlus segImg = imageToSegment.duplicate();
        segImg.setTitle(imageToSegment.getTitle() + " - Segmentación");

        IJ.showStatus("Convirtiendo a 8-bit...");
        IJ.run(segImg, "8-bit", "");

        if (config.getGaussianSigma() > 0) {
            IJ.showStatus("Filtro Gaussiano (sigma=" + config.getGaussianSigma() + ")...");
            IJ.run(segImg, "Gaussian Blur...", "sigma=" + config.getGaussianSigma() + " stack");
        }

        IJ.showStatus("Aplicando threshold (" + config.getThresholdMethod() + ")...");
        IJ.run(segImg, "Auto Threshold", "method=" + config.getThresholdMethod() + " white stack");

        if (config.getMedianRadius() > 0) {
            IJ.showStatus("Filtro mediano (radio=" + config.getMedianRadius() + ")...");
            IJ.run(segImg, "Median...", "radius=" + config.getMedianRadius() + " stack");
        }

        if (config.isApplyFillHoles()) {
            IJ.showStatus("Rellenando huecos...");
            IJ.run(segImg, "Fill Holes", "stack");
        }

        ImagePlus binaryImageToShow = null;
        if (config.isShowProcessingImage()) {
            binaryImageToShow = segImg.duplicate();
            binaryImageToShow.setTitle(segImg.getTitle() + " (Binaria)");
        }
        
        if (config.isShowDetailedProgress()) {
            IJ.log("     Analizando partículas...");
        }

        RoiManager rm = RoiManager.getRoiManager();
        
        int roiCountBefore = rm.getCount();

        String analyzeParams = String.format(Locale.US,
            "size=%.0f-%.0f circularity=%.2f-%.2f show=Nothing exclude add stack",
            config.getMinCellSize(), config.getMaxCellSize(), config.getMinCircularity(), config.getMaxCircularity());
        IJ.run(segImg, "Analyze Particles...", analyzeParams);

        Roi[] allRoisInManager = rm.getRoisAsArray();
        Roi[] newRois = new Roi[rm.getCount() - roiCountBefore];
        for (int i = 0; i < newRois.length; i++) {
            newRois[i] = allRoisInManager[roiCountBefore + i];
        }

        if (newRois.length == 0) {
            showNoCellsDetectedWarning(imageToSegment.getTitle());
        }

        if (segImg.getWindow() != null) segImg.close();

        return new Resultado(newRois, binaryImageToShow);
    }
    
    public void applyLabelingToImage(ImagePlus targetImage, RoiManager roiManager) {
        if (roiManager == null || roiManager.getCount() == 0 || targetImage == null) {
            return;
        }

        if (config.isShowDetailedProgress()) {
            IJ.log("  - Aplicando Overlay a '" + targetImage.getTitle() + "'");
        }

        Roi[] rois = roiManager.getRoisAsArray();
        
        Overlay overlay = new Overlay();
        for(Roi roi : rois) {
            overlay.add((Roi) roi.clone());
        }
        
        overlay.drawLabels(true);
        overlay.drawNames(true);
        overlay.drawBackgrounds(true);
        overlay.setLabelColor(Color.WHITE);
        overlay.setLabelFont(new Font("SansSerif", Font.PLAIN, 12));
        
        targetImage.setOverlay(overlay);
        
        // Intento de evitar "ghosting"
        roiManager.deselect();
        
        if (!roiManager.isVisible()) {
             roiManager.setVisible(true);
        }
        
        targetImage.updateAndDraw();
    }

    private void showNoCellsDetectedWarning(String frameTitle) {
        if (config.isShowDetailedProgress()) {
            IJ.log("     -> ¡Advertencia! No se detectaron células en " + frameTitle + ".");
        }
    }
}