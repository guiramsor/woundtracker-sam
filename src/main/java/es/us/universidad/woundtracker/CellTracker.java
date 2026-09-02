package es.us.universidad.woundtracker;

import ij.IJ;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;

import java.awt.geom.Point2D;
import java.util.HashMap;
import ij.process.ImageStatistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Seguimiento de células entre frames. Enlaza cada una con su vecino más
 * cercano del frame siguiente, por distancia entre centroides.
 */
public class CellTracker {

    /**
     * Re-etiqueta las ROIs del manager con un identificador estable en el tiempo.
     * Cada ROI tiene que traer ya su posición temporal.
     *
     * @return cuántas trayectorias únicas han salido.
     */
    public int trackCells(RoiManager roiManager, boolean showDetailedProgress) {
        if (roiManager == null || roiManager.getCount() == 0) {
            return 0;
        }
        
        if (showDetailedProgress) {
            IJ.log("  -> Agrupando " + roiManager.getCount() + " ROIs por frame...");
        }

        Map<Integer, List<Roi>> roisByFrame = groupRoisByFrame(roiManager);
        if (roisByFrame.size() <= 1) {
            if (showDetailedProgress) {
                IJ.log("  - Tracking no aplicable: solo se encontraron células en un único frame.");
            }
            return roiManager.getCount();
        }

        int maxFrame = Collections.max(roisByFrame.keySet());
        int firstFrame = Collections.min(roisByFrame.keySet());
        int nextCellId = 1;
        int totalLinks = 0;
        Map<Roi, String> finalLabels = new HashMap<>();

        if (showDetailedProgress) {
            IJ.log("  -> Iniciando enlace de trayectorias (tracks)...");
        }
        
        if (roisByFrame.containsKey(firstFrame)) {
            for (Roi roi : roisByFrame.get(firstFrame)) {
                finalLabels.put(roi, "Cell-" + nextCellId++);
            }
        }

        for (int t = firstFrame; t < maxFrame; t++) {
            if (!roisByFrame.containsKey(t) || !roisByFrame.containsKey(t + 1)) {
                continue;
            }

            List<Roi> currentFrameRois = roisByFrame.get(t);
            List<Roi> nextFrameRois = roisByFrame.get(t + 1);
            Set<Roi> unassignedNextFrameRois = new HashSet<>(nextFrameRois);

            Map<Roi, Point2D.Double> centroides = new HashMap<>();
            for (Roi roi : currentFrameRois) centroides.put(roi, getCentroid(roi));
            for (Roi roi : nextFrameRois) centroides.put(roi, getCentroid(roi));

            List<Link> potentialLinks = new ArrayList<>();
            for (Roi currentRoi : currentFrameRois) {
                if (!finalLabels.containsKey(currentRoi)) continue;
                for (Roi nextRoi : nextFrameRois) {
                    double d = centroides.get(currentRoi).distance(centroides.get(nextRoi));
                    potentialLinks.add(new Link(currentRoi, nextRoi, d));
                }
            }
            
            potentialLinks.sort(Comparator.comparingDouble(l -> l.distance));

            Set<Roi> assignedCurrentRois = new HashSet<>();
            int linksThisFrame = 0;
            for (Link link : potentialLinks) {
                if (!assignedCurrentRois.contains(link.source) && unassignedNextFrameRois.contains(link.target)) {
                    finalLabels.put(link.target, finalLabels.get(link.source));
                    assignedCurrentRois.add(link.source);
                    unassignedNextFrameRois.remove(link.target);
                    linksThisFrame++;
                }
            }

            if(showDetailedProgress) {
                String logMsg = String.format(Locale.US, "     - Frame %2d -> %2d: %d células enlazadas.", t, t + 1, linksThisFrame);
                if (!unassignedNextFrameRois.isEmpty()) {
                    logMsg += String.format(Locale.US, " (%d células nuevas detectadas)", unassignedNextFrameRois.size());
                }
                IJ.log(logMsg);
            }
            totalLinks += linksThisFrame;
            
            for (Roi newRoi : unassignedNextFrameRois) {
                finalLabels.put(newRoi, "Cell-" + nextCellId++);
            }
        }

        for (Map.Entry<Roi, String> entry : finalLabels.entrySet()) {
            entry.getKey().setName(entry.getValue());
        }

        List<Roi> allRois = Arrays.asList(roiManager.getRoisAsArray());
        roiManager.reset();
        for (Roi roi : allRois) {
            roiManager.addRoi(roi);
        }
        roiManager.runCommand("Sort");

        int totalCellsTracked = nextCellId - 1;

        if (showDetailedProgress) {
            IJ.log("  -> Resumen del Tracking:");
            IJ.log("     - Total de enlaces establecidos: " + totalLinks);
            IJ.log("     - Total de trayectorias (células) identificadas: " + totalCellsTracked);
        }

        return totalCellsTracked;
    }

    private Map<Integer, List<Roi>> groupRoisByFrame(RoiManager rm) {
        Map<Integer, List<Roi>> map = new HashMap<>();
        for (Roi roi : rm.getRoisAsArray()) {
            int frame = roi.getTPosition();
            if (frame > 0) {
                map.computeIfAbsent(frame, k -> new ArrayList<>()).add(roi);
            }
        }
        return map;
    }

    private Point2D.Double getCentroid(Roi roi) {
        ImageStatistics stats = roi.getStatistics();
        return new Point2D.Double(stats.xCentroid, stats.yCentroid);
    }

    private static class Link {
        final Roi source;
        final Roi target;
        final double distance;

        Link(Roi source, Roi target, double distance) {
            this.source = source;
            this.target = target;
            this.distance = distance;
        }
    }
}