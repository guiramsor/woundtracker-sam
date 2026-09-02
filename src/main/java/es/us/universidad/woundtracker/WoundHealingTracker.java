package es.us.universidad.woundtracker;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.gui.Roi;
import ij.plugin.ChannelSplitter;
import ij.plugin.frame.RoiManager;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.display.ImageDisplay;
import net.imagej.display.ImageDisplayService;
import net.imagej.legacy.LegacyService;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.module.MutableModuleItem;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

// Sin entrada de menú: la lanza WoundTrackerCommand tras preparar la serie.
@Plugin(type = Command.class, label = "Segmentacion celular (clasica)",
    priority = Double.MAX_VALUE)
public class WoundHealingTracker extends DynamicCommand {

    private static final String PRESET_PREFIX = "WoundTracker.preset.";
    private static final String PRESET_NAMES_KEY = PRESET_PREFIX + "names";
    private static final String LAST_USED_PRESET_KEY = PRESET_PREFIX + "lastSelected";
    private static final String DEFAULT_PRESET_NAME = "Estándar";
    private static final String CUSTOM_PRESET_NAME = "Personalizado";
    private static final String SAVE_ACTION_TEXT = "--- Guardar Preset Actual ---";
    private static final String DELETE_ACTION_TEXT = "--- Eliminar Preset Cargado ---";
    private static final String PRESET_NAME_SEPARATOR = ",";

    @Parameter private UIService uiService;
    @Parameter private ImageDisplayService imageDisplayService;
    @Parameter private LegacyService legacyService;

    @Parameter(
        label = "<html><b>CONFIGURACIÓN RÁPIDA</b></html>",
        callback = "onPresetChanged",
        description = "Selecciona, guarda o elimina una configuración predefinida."
    )
    private String selectedPreset;

    @Parameter(label = "<html><b>SEGMENTACIÓN CELULAR</b></html>", persist = false, required = false, visibility = ItemVisibility.MESSAGE)
    private String segmentationHeader;

    @Parameter(label = "Activar Segmentación", description = "Activa el pipeline de segmentación celular y tracking.")
    private boolean applyCellSegmentation;

    // Lo fija WoundTrackerCommand al lanzar esta via. CellSegmenter pasa la
    // imagen a 8 bits con el rango que define, y de ahi sale el umbral.
    @Parameter(visibility = ItemVisibility.INVISIBLE, persist = false, required = false)
    private double saturatedPercent = 0.35;

    @Parameter(
        label = "Filtro Gaussiano (Sigma)",
        min = "0", max = "10", stepSize = "0.5",
        description = "Aplica un filtro Gaussiano antes del threshold para suavizar. 0: sin filtro."
    )
    private double gaussianSigma;

    @Parameter(
        label = "Algoritmo Threshold",
        choices = {"Otsu", "Li", "Triangle", "Yen", "Huang", "Intermodes", "IsoData", "MaxEntropy", "Mean", "MinError"},
        description = "Algoritmo de binarización automática."
    )
    private String thresholdMethod;

    @Parameter(
        label = "Filtro Mediano (Radio)",
        min = "0", max = "20", stepSize = "1",
        description = "Aplica un filtro Mediano después del threshold para limpiar ruido. 0: sin filtro."
    )
    private int medianRadius;
    
    @Parameter(label = "Rellenar Huecos (Fill Holes)", description = "Rellena los agujeros dentro de las máscaras de las células.")
    private boolean applyFillHoles;

    @Parameter(label = "<html><b>CRITERIOS DE DETECCIÓN</b></html>", persist = false, required = false, visibility = ItemVisibility.MESSAGE)
    private String segDetectionHeader;

    @Parameter(label = "Tamaño Mín. (px)", min = "100", max = "10000", stepSize = "100")
    private double minCellSize;

    @Parameter(label = "Tamaño Máx. (px)", min = "10000", max = "200000", stepSize = "5000")
    private double maxCellSize;

    @Parameter(label = "Circularidad Mín.", min = "0.0", max = "1.0", stepSize = "0.05")
    private double minCircularity;

    @Parameter(label = "Circularidad Máx.", min = "0.0", max = "1.0", stepSize = "0.05")
    private double maxCircularity;

    @Parameter(label = "<html><b>RESULTADOS</b></html>", persist = false, required = false, visibility = ItemVisibility.MESSAGE)
    private String visualizationHeader;

    @Parameter(label = "Mostrar Stack Binario", description = "Genera un único stack con el resultado binario de la segmentación de todos los frames.")
    private boolean showProcessingImage;

    @Parameter(label = "<html><b>CONFIGURACIÓN</b></html>", persist = false, required = false, visibility = ItemVisibility.MESSAGE)
    private String infoHeader;

    @Parameter(label = "Progreso Detallado", description = "Muestra información detallada del progreso en el log de ImageJ.")
    private boolean showDetailedProgress;

    public WoundHealingTracker() {
        ensureDefaultPresetExists();
        loadSettings(Prefs.get(LAST_USED_PRESET_KEY, DEFAULT_PRESET_NAME));
    }

    @Override
    public void initialize() {
        @SuppressWarnings("unchecked")
        final MutableModuleItem<String> presetInput = (MutableModuleItem<String>) getInfo().getInput("selectedPreset");
        presetInput.setChoices(getPresetChoices());
    }

    @Override
    public void run() {
        if (showDetailedProgress) logInitialConfiguration();

        if(!isAction(this.selectedPreset)) {
            Prefs.set(LAST_USED_PRESET_KEY, this.selectedPreset);
            if (CUSTOM_PRESET_NAME.equals(this.selectedPreset)) {
                 saveSettings(CUSTOM_PRESET_NAME);
            }
        }

        ImagePlus inputImage = getActiveImagePlus();
        if (inputImage == null) {
            uiService.showDialog("No hay imagen activa. Por favor, abra una imagen.", "Error - WoundTracker");
            return;
        }

        final CellSegmenter cellSegmenter = new CellSegmenter(this);

        try {
            ImagePlus segmentationStack = prepararStack(inputImage.duplicate(), true);
            ImagePlus visualOutputStack = prepararStack(inputImage.duplicate(), false);
            visualOutputStack.setTitle(inputImage.getShortTitle() + " - Resultado del Tracking");

            if (!isApplyCellSegmentation()) {
                traza("-> La segmentación celular está desactivada. Mostrando solo el resultado visual.");
                visualOutputStack.show();
                logCompletion();
                return;
            }

            traza("\n// PASO 1: SEGMENTACIÓN FRAME A FRAME");
            
            RoiManager rm = RoiManager.getRoiManager();
            rm.reset();

            ImageStack binaryStack = null;
            if (isShowProcessingImage()) {
                binaryStack = new ImageStack(segmentationStack.getWidth(), segmentationStack.getHeight());
            }

            int nFrames = segmentationStack.getNFrames();
            if (nFrames == 0) nFrames = 1;

            for (int t = 1; t <= nFrames; t++) {
                traza(String.format("  -> Procesando Frame %d / %d...", t, nFrames));
                IJ.showStatus("Segmentando frame " + t + "/" + nFrames);
                
                segmentationStack.setT(t);
                ImagePlus currentFrameImage = new ImagePlus("Frame " + t, segmentationStack.getProcessor().duplicate());
                
                CellSegmenter.Resultado frameResult = cellSegmenter.segmentCells(currentFrameImage);

                if (frameResult != null && frameResult.hasDetections()) {
                    for (Roi roi : frameResult.rois) {
                        roi.setPosition(1, 1, t);
                    }
                }
                
                if (binaryStack != null && frameResult != null && frameResult.binaria != null) {
                    binaryStack.addSlice("Frame " + t, frameResult.binaria.getProcessor());
                }
            }
            
            if (rm.getCount() == 0) {
                traza("\n// FINALIZACIÓN: No se detectaron células que cumplan los criterios.");
                 showFinalNoCellsDialog();
                 visualOutputStack.show();
                 return;
            }
            
            traza("\n// PASO 2: TRACKING CELULAR");
            CellTracker tracker = new CellTracker();
            int totalCells = tracker.trackCells(rm, showDetailedProgress);
            
                if (isShowDetailedProgress()) {
                IJ.log("\n// PASO 3: VISUALIZACIÓN FINAL");
                IJ.log("  - Aplicando overlay con " + totalCells + " células únicas al resultado.");
            }
            cellSegmenter.applyLabelingToImage(visualOutputStack, rm);

                if (binaryStack != null && binaryStack.getSize() > 0) {
                ImagePlus finalBinaryImage = new ImagePlus("Stack Binario Resultado", binaryStack);
                finalBinaryImage.setDimensions(1, 1, binaryStack.getSize());
                finalBinaryImage.show();
            }
            
            visualOutputStack.show();
            
            logCompletion();

        } catch (Exception e) {
            IJ.handleException(e);
        } finally {
            IJ.showProgress(100);
            IJ.showStatus("WoundTracker: Listo.");
        }
    }
    
    /** Al Log solo si el usuario pidió progreso detallado. */
    private void traza(String mensaje) {
        if (showDetailedProgress) IJ.log(mensaje);
    }

    private void logInitialConfiguration() {
        String startTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
        IJ.log("\n//================================================================");
        IJ.log("//  WOUNDTRACKER - INICIO DEL ANÁLISIS");
        IJ.log("//================================================================");
        IJ.log("// Fecha y Hora: " + startTime);
        IJ.log(" ");
        IJ.log("-> Configuración Seleccionada: '" + getSelectedPreset() + "'");
        IJ.log("\n-> Parámetros de Análisis:");
        
        IJ.log(String.format(Locale.US, "  - Segmentación: Activada=%b", isApplyCellSegmentation()));
        if (isApplyCellSegmentation()) {
            IJ.log(String.format(Locale.US, "    - Algoritmo: Threshold=%s", getThresholdMethod()));
            IJ.log(String.format(Locale.US, "    - Filtros: Gauss (Sigma=%.1f), Mediano (Radio=%d), Rellenar Huecos=%b",
                getGaussianSigma(), getMedianRadius(), isApplyFillHoles()));
            IJ.log(String.format(Locale.US, "    - Criterios: Tamaño (px)=%.0f-%.0f, Circularidad=%.2f-%.2f",
                getMinCellSize(), getMaxCellSize(), getMinCircularity(), getMaxCircularity()));
        }
        
        IJ.log(String.format(Locale.US, "  - Resultados: Mostrar Stack Binario=%b", isShowProcessingImage()));
    }

    private ImagePlus prepararStack(ImagePlus inputImage, boolean contraste) {
        ImagePlus imageToProcess = inputImage;
        if (imageToProcess.getNChannels() > 1) {
            ImagePlus[] channels = ChannelSplitter.split(imageToProcess);
            imageToProcess = channels[0];
        }
        // Por si llega sin preparar; WoundTrackerCommand ya proyecta.
        if (imageToProcess.getNSlices() > 1) {
            imageToProcess = ImageProcessorService.performZProjection(imageToProcess, "Maximum Intensity");
        }
        if (contraste) ImageProcessorService.applyAutoContrast(imageToProcess, saturatedPercent);
        return imageToProcess;
    }

    
    private void logCompletion() {
        if (showDetailedProgress) {
            IJ.log("\n//================================================================");
            IJ.log("//  ANÁLISIS COMPLETADO");
            IJ.log("//================================================================");
        }
    }

    private void showFinalNoCellsDialog() {
        if (uiService != null) {
            String detailedMsg =
                "<html><body style='width: 400px;'>" +
                "<h3>Sin células detectadas</h3>" +
                "<p>No se encontraron células que cumplan los criterios en ninguno de los frames analizados.</p>" +
                "<p><b>Sugerencias para optimizar:</b></p>" +
                "<ul>" +
                "<li><b>Tamaño:</b> Ajuste el rango " + (int)getMinCellSize() + "-" + (int)getMaxCellSize() + " píxeles</li>" +
                "<li><b>Circularidad:</b> Relaje el rango " + String.format(Locale.US,"%.2f-%.2f", getMinCircularity(), getMaxCircularity()) + "</li>" +
                "<li><b>Threshold:</b> Pruebe un método diferente a '" + getThresholdMethod() + "'</li>" +
                "<li><b>Filtros:</b> Pruebe a incrementar el radio del Filtro Gaussiano o Mediano.</li>" +
                "</ul>" +
                "<p><i>Sugerencia: Active 'Mostrar Stack Binario' para depurar la segmentación.</i></p>" +
                "</body></html>";
            uiService.showDialog(detailedMsg, "WoundTracker - Optimización Requerida");
        }
    }

    @SuppressWarnings("unused")
    private void onPresetChanged() {
        String previousSelection = getPreviousSelection();
        if (!isAction(selectedPreset)) {
            Prefs.set(LAST_USED_PRESET_KEY, selectedPreset);
        }

        switch (selectedPreset) {
            case SAVE_ACTION_TEXT:
                saveNewPreset();
                break;
            case DELETE_ACTION_TEXT:
                deleteSelectedPreset(previousSelection);
                break;
            default:
                loadSettings(selectedPreset);
                break;
        }
    }

    private void saveNewPreset() {
        String newName = IJ.getString("Introduce el nombre del nuevo preset:", "Nuevo Preset");
        if (newName != null && !newName.trim().isEmpty() && !newName.equals("Nuevo Preset")) {
            newName = newName.trim();
            saveSettings(newName);

            List<String> presetNames = getPresetNameList();
            if (!presetNames.contains(newName)) {
                presetNames.add(newName);
            }
            savePresetNameList(presetNames);
            this.selectedPreset = newName;
            
            traza("Preset '" + newName + "' guardado.");
            IJ.showMessage("Preset Guardado", "El preset '" + newName + "' se ha guardado correctamente.\n" +
                                            "Estará disponible en la lista la próxima vez que ejecutes el plugin.");
        } else {
            this.selectedPreset = getPreviousSelection();
        }
    }

    private void deleteSelectedPreset(String presetToDelete) {
        if (isAction(presetToDelete) || presetToDelete.equals(CUSTOM_PRESET_NAME) || presetToDelete.equals(DEFAULT_PRESET_NAME)) {
            IJ.showMessage("Acción no permitida", "'" + presetToDelete + "' no se puede eliminar.");
            this.selectedPreset = presetToDelete; // Restaura la selección
            return;
        }
        boolean confirm = IJ.showMessageWithCancel("Confirmar Eliminación",
            "¿Estás seguro de que quieres eliminar el preset '" + presetToDelete + "'?");
        if (confirm) {
            List<String> presetNames = getPresetNameList();
            presetNames.remove(presetToDelete);
            savePresetNameList(presetNames);
            
            traza("Preset '" + presetToDelete + "' eliminado.");
            IJ.showMessage("Preset Eliminado", "El preset '" + presetToDelete + "' ha sido eliminado.\n" +
                                               "El cambio se reflejará la próxima vez que ejecutes el plugin.");

            loadSettings(DEFAULT_PRESET_NAME);
            this.selectedPreset = DEFAULT_PRESET_NAME;

        } else {
            this.selectedPreset = presetToDelete;
        }
    }

    private void loadSettings(String presetName) {
        if (CUSTOM_PRESET_NAME.equals(presetName)) {
            this.applyCellSegmentation = false;
            this.gaussianSigma = 0.0;
            this.thresholdMethod = "Otsu";
            this.medianRadius = 0;
            this.applyFillHoles = false;
            this.minCellSize = 100.0;
            this.maxCellSize = 10000.0;
            this.minCircularity = 0.0;
            this.maxCircularity = 1.0;
            this.showProcessingImage = false;
            this.showDetailedProgress = true;
            this.selectedPreset = presetName;
            return;
        }

        if (presetName == null || isAction(presetName)) {
            presetName = getPreviousSelection();
        }

        String p = PRESET_PREFIX + presetName + ".";
        
        this.applyCellSegmentation = Prefs.get(p + "applyCellSegmentation", "true").equals("true");
        this.thresholdMethod = Prefs.get(p + "thresholdMethod", "Otsu");
        this.medianRadius = (int) Prefs.get(p + "medianRadius", 8.0);
        this.minCellSize = Prefs.get(p + "minCellSize", 1000.0);
        this.maxCellSize = Prefs.get(p + "maxCellSize", 180000.0);
        this.minCircularity = Prefs.get(p + "minCircularity", 0.35);
        this.maxCircularity = Prefs.get(p + "maxCircularity", 1.0);
        this.showProcessingImage = Prefs.get(p + "showProcessingImage", "true").equals("true");
        this.showDetailedProgress = Prefs.get(p + "showDetailedProgress", "true").equals("true");
        this.gaussianSigma = Prefs.get(p + "gaussianSigma", 2.5);
        this.applyFillHoles = Prefs.get(p + "applyFillHoles", "true").equals("true");
        this.selectedPreset = presetName;
    }

    private void saveSettings(String presetName) {
        String p = PRESET_PREFIX + presetName + ".";
        Prefs.set(p + "applyCellSegmentation", applyCellSegmentation);
        Prefs.set(p + "thresholdMethod", thresholdMethod);
        Prefs.set(p + "medianRadius", medianRadius);
        Prefs.set(p + "minCellSize", minCellSize);
        Prefs.set(p + "maxCellSize", maxCellSize);
        Prefs.set(p + "minCircularity", minCircularity);
        Prefs.set(p + "maxCircularity", maxCircularity);
        Prefs.set(p + "showProcessingImage", showProcessingImage);
        Prefs.set(p + "showDetailedProgress", showDetailedProgress);
        Prefs.set(p + "gaussianSigma", gaussianSigma);
        Prefs.set(p + "applyFillHoles", applyFillHoles);
    }
    
    private List<String> getPresetChoices() {
        List<String> choices = new ArrayList<>();
        choices.add(CUSTOM_PRESET_NAME);
        choices.addAll(getPresetNameList());
        choices.add(SAVE_ACTION_TEXT);
        choices.add(DELETE_ACTION_TEXT);
        return choices;
    }

    private void ensureDefaultPresetExists() {
        String storedNames = Prefs.get(PRESET_NAMES_KEY, null);
        if (storedNames == null) {
            String p = PRESET_PREFIX + DEFAULT_PRESET_NAME + ".";
            
            Prefs.set(p + "applyCellSegmentation", true);
            Prefs.set(p + "gaussianSigma", 2.5);
            Prefs.set(p + "thresholdMethod", "Otsu");
            Prefs.set(p + "medianRadius", 8.0);
            Prefs.set(p + "applyFillHoles", true);
            Prefs.set(p + "minCellSize", 1000.0);
            Prefs.set(p + "maxCellSize", 180000.0);
            Prefs.set(p + "minCircularity", 0.35);
            Prefs.set(p + "maxCircularity", 1.0);
            Prefs.set(p + "showProcessingImage", true);
            Prefs.set(p + "showDetailedProgress", true);
            
            savePresetNameList(new ArrayList<>(Collections.singletonList(DEFAULT_PRESET_NAME)));
        }
    }

    private String getPreviousSelection() {
        return Prefs.get(LAST_USED_PRESET_KEY, CUSTOM_PRESET_NAME);
    }

    private List<String> getPresetNameList() {
        String storedNames = Prefs.get(PRESET_NAMES_KEY, DEFAULT_PRESET_NAME);
        if (storedNames == null || storedNames.isEmpty()) {
            return new ArrayList<>();
        }
        return new ArrayList<>(Arrays.asList(storedNames.split(PRESET_NAME_SEPARATOR)));
    }

    private void savePresetNameList(List<String> names) {
        List<String> distinctNames = names.stream().distinct().sorted().collect(Collectors.toList());
        String nameString = String.join(PRESET_NAME_SEPARATOR, distinctNames);
        Prefs.set(PRESET_NAMES_KEY, nameString);
    }
    
    private boolean isAction(String selection) {
        if (selection == null) return false;
        return selection.equals(SAVE_ACTION_TEXT) || selection.equals(DELETE_ACTION_TEXT);
    }

    private ImagePlus getActiveImagePlus() {
        ImageDisplay disp = imageDisplayService.getActiveImageDisplay();
        if (disp != null) {
            return legacyService.getImageMap().lookupImagePlus(disp);
        }
        return IJ.getImage();
    }

    
    public String getSelectedPreset() { return selectedPreset; }
    public double getSaturatedPercent() { return saturatedPercent; }
    public boolean isApplyCellSegmentation() { return applyCellSegmentation; }
    public String getThresholdMethod() { return thresholdMethod; }
    public int getMedianRadius() { return medianRadius; }
    public double getMinCellSize() { return minCellSize; }
    public double getMaxCellSize() { return maxCellSize; }
    public double getMinCircularity() { return minCircularity; }
    public double getMaxCircularity() { return maxCircularity; }
    public boolean isShowProcessingImage() { return showProcessingImage; }
    public boolean isShowDetailedProgress() { return showDetailedProgress; }
    public double getGaussianSigma() { return gaussianSigma; }
    public boolean isApplyFillHoles() { return applyFillHoles; }

    public static void main(String[] args) throws Exception {
        ImageJ ij = new ImageJ();
        ij.ui().showUI();
        File f = ij.ui().chooseFile(null, "open");
        if (f != null && f.exists()) {
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            ij.ui().show(d);
            ij.command().run(WoundTrackerCommand.class, true).get();
        } else {
            IJ.log("No se seleccionó ningún archivo o el archivo no existe.");
        }
    }
}