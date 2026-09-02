package es.us.universidad.woundtracker.sam;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import ij.process.ImageProcessor;

import org.apposed.appose.Environment;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.scijava.command.Command;
import org.scijava.plugin.Menu;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * P4: un frame real de la serie abierta y clics del usuario, contra el modelo.
 *
 * El usuario marca con la herramienta multipunto en dos pasadas (surco y luego
 * fondo), la imagen viaja por memoria compartida, vuelven las tres candidatas,
 * y {@link SelectorS1} elige en Java la que acaba siendo ROI.
 *
 * Lleva incorporada la comprobación de P3: Python calcula su propio S1 sobre
 * las mismas candidatas y se contrastan los dos índices.
 *
 * Los límites de normalización salen de la serie entera (P5), no de este
 * frame; se enseñan los dos en el Log para que la diferencia se vea.
 */
@Plugin(
    type = Command.class,
    menu = {
        @Menu(label = "Plugins", weight = 0),
        @Menu(label = "WoundTracker", weight = 1),
        @Menu(label = "Pruebas", weight = 3),
        @Menu(label = "P4 - Frame real y clics")
    }
)
public class SamFrameClicsTest implements Command {

    private static final String SCRIPT =
        "import os, sys\n" +
        "import appose, numpy as np\n" +
        "sys.path.insert(0, repo_src)\n" +
        "os.environ.setdefault('SURCO_PESOS', pesos)\n" +
        "from surco import adaptadores, modelos, selector\n" +
        "adaptadores.cargar_disponibles()\n" +
        "img = imagen.ndarray()\n" +
        // Los límites llegan de Java, calculados sobre la serie entera (P5).
        "rgb = modelos.preparar(img, (float(limites[0]), float(limites[1])))\n" +
        "m = modelos.crear(modelo, checkpoint)\n" +
        "m.cargar()\n" +
        "pts = [tuple(p) for p in puntos]\n" +
        "etqs = list(etiquetas)\n" +
        "c = m.segmentar(rgb, pts, etqs)\n" +
        "gris = rgb[:, :, 0]\n" +
        // Vuelven las k candidatas y el gris con el que se puntúan, para que el
        // S1 de Java trabaje sobre exactamente los mismos datos que el de Python.
        // uint8 y no bool: el lado Python de Appose deduce los bytes por elemento
        // de los dígitos del nombre del tipo, así que 'bool' no le vale.
        "salida = appose.NDArray('uint8', list(c.mascaras.shape))\n" +
        "salida.ndarray()[:] = c.mascaras.astype(np.uint8)\n" +
        "plano = appose.NDArray('uint8', list(gris.shape))\n" +
        "plano.ndarray()[:] = gris\n" +
        // El S1 de Python, para contrastarlo con el de Java.
        "indice, respaldo = selector.elegir(c, gris, pts, etqs)\n" +
        "rasgos = selector.rasgos_de(c, gris, pts, etqs)\n" +
        "task.outputs['candidatas'] = salida\n" +
        "task.outputs['gris'] = plano\n" +
        "task.outputs['s1_indice'] = int(indice)\n" +
        "task.outputs['s1_respaldo'] = bool(respaldo)\n" +
        "task.outputs['s1_z'] = [float(r['z']) for r in rasgos]\n" +
        "task.outputs['s1_pasa'] = [bool(r['pasa']) for r in rasgos]\n" +
        "task.outputs['s1_elong'] = [float(r['elongacion']) for r in rasgos]\n" +
        "task.outputs['n'] = len(c)\n" +
        "task.outputs['puntuaciones'] = [float(x) for x in c.puntuaciones.tolist()]\n" +
        "task.outputs['dispositivo'] = str(getattr(m, 'dispositivo', '?'))\n";

    @Parameter(label = "Modelo")
    private String modelo = "microsam_lm";

    @Parameter(label = "Checkpoint")
    private String checkpoint = "vit_b";

    @Override
    public void run() {
        ImagePlus imp = WindowManager.getCurrentImage();
        if (imp == null) {
            IJ.error("WoundTracker SAM", "No hay ninguna imagen abierta.");
            return;
        }
        if (imp.getBitDepth() != 16) {
            IJ.error("WoundTracker SAM",
                "Esperaba una imagen de 16 bits y esta es de " + imp.getBitDepth() + ".");
            return;
        }

        ImageProcessor ip = imp.getProcessor();   // el frame que se está viendo
        int ancho = ip.getWidth();
        int alto = ip.getHeight();

        List<List<Integer>> puntos = new ArrayList<>();
        List<Integer> etiquetas = new ArrayList<>();
        if (!SamComun.pedirClics(imp, "el SURCO", 1, puntos, etiquetas)) {
            IJ.log("-> Cancelado: hacen falta puntos sobre el surco.");
            return;
        }
        SamComun.pedirClics(imp, "el FONDO (opcional)", 0, puntos, etiquetas);

        IJ.log("\n//================================================================");
        IJ.log("// WOUNDTRACKER SAM - P4: FRAME REAL Y CLICS");
        IJ.log("//================================================================");
        IJ.log("-> Imagen: " + imp.getTitle() + " (" + ancho + "x" + alto
            + ", frame " + imp.getT() + "/" + imp.getNFrames() + ")");
        IJ.log("-> Clics: " + puntos.size() + " (etiquetas " + etiquetas + ")");

        double[] limites = ParametrosDeSerie.limitesDeLaSerie(imp);
        double[] soloEsteFrame = ParametrosDeSerie.limitesDeUnFrame(imp);
        IJ.log(String.format(Locale.US,
            "-> Límites de la serie (%d frames): negro=%.0f blanco=%.3f", imp.getNFrames(),
            limites[0], limites[1]));
        IJ.log(String.format(Locale.US,
            "   (solo este frame habrían sido: negro=%.0f blanco=%.3f)",
            soloEsteFrame[0], soloEsteFrame[1]));

        long t0 = System.currentTimeMillis();
        try (NDArray imagen = new NDArray(NDArray.DType.UINT16,
                new NDArray.Shape(NDArray.Shape.Order.C_ORDER, alto, ancho))) {

            SamComun.copiarPixeles(ip, imagen);

            Map<String, Object> inputs = new HashMap<>();
            inputs.put("repo_src", Recursos.codigoDelBanco().getAbsolutePath());
            inputs.put("pesos", Recursos.pesos().getAbsolutePath());
            inputs.put("modelo", modelo);
            inputs.put("checkpoint", checkpoint);
            inputs.put("imagen", imagen);
            inputs.put("puntos", puntos);
            inputs.put("etiquetas", etiquetas);
            inputs.put("limites", Arrays.asList(limites[0], limites[1]));

            Environment env = EntornoAppose.para(modelo);

            try (Service python = env.python()) {
                python.debug(line -> IJ.log("   [worker] " + line));
                python.init("import numpy");

                Task task = python.task(SCRIPT, inputs);
                task.waitFor();
                double segundos = (System.currentTimeMillis() - t0) / 1000.0;

                if (task.status != TaskStatus.COMPLETE) {
                    IJ.log("-> FALLO. Estado: " + task.status);
                    IJ.log("   " + task.error);
                    return;
                }

                int k = ((Number) task.outputs.get("n")).intValue();
                IJ.log("-> dispositivo  = " + task.outputs.get("dispositivo"));
                IJ.log("-> candidatas   = " + k
                    + ", puntuaciones " + task.outputs.get("puntuaciones"));

                try (NDArray candidatas = (NDArray) task.outputs.get("candidatas");
                     NDArray gris = (NDArray) task.outputs.get("gris")) {

                    byte[][] mascaras = SamComun.planos(candidatas, k, ancho * alto);
                    byte[] plano = new byte[ancho * alto];
                    gris.buffer().get(plano);

                    SelectorS1.Eleccion s1 = SelectorS1.elegir(
                        mascaras, plano, ancho, alto,
                        SamComun.aPuntos(puntos), SamComun.aEtiquetas(etiquetas),
                        ParametrosDeSerie.ladoVentana(imp));

                    compararConPython(s1, task, k);

                    Roi roi = SamComun.aRoi(mascaras[s1.indice], ancho, alto);
                    if (roi == null) {
                        IJ.log("-> La máscara elegida está vacía: no hay ROI que crear.");
                    } else {
                        publicar(imp, roi);
                        IJ.log("-> ROI de la candidata " + s1.indice + " añadida al ROI Manager.");
                    }
                }
                IJ.log(String.format(Locale.US, "-> P4 OK en %.1f s", segundos));
            }
        } catch (Exception e) {
            IJ.log("-> FALLO P4: " + e.getMessage());
            IJ.handleException(e);
        }
    }

    /**
     * P3: el S1 de Java tiene que elegir la misma candidata que el de Python
     * sobre las mismas máscaras y el mismo gris. Si divergen, la traducción
     * está mal y se ve aquí.
     */
    private void compararConPython(SelectorS1.Eleccion s1, Task task, int k) {
        IJ.log("-> S1 Java   = " + s1.indice
            + (s1.respaldo ? " (respaldo)" : "")
            + "  pasa=" + Arrays.toString(Arrays.copyOf(s1.pasa, k))
            + " z=" + Arrays.toString(Arrays.copyOf(s1.z, k))
            + " elong=" + Arrays.toString(Arrays.copyOf(s1.elongacion, k)));
        IJ.log("-> S1 Python = " + task.outputs.get("s1_indice")
            + (Boolean.TRUE.equals(task.outputs.get("s1_respaldo")) ? " (respaldo)" : "")
            + "  pasa=" + task.outputs.get("s1_pasa")
            + " z=" + task.outputs.get("s1_z")
            + " elong=" + task.outputs.get("s1_elong"));

        int enPython = ((Number) task.outputs.get("s1_indice")).intValue();
        IJ.log(s1.indice == enPython
            ? "-> S1 COINCIDE (Java y Python eligen la " + s1.indice + ")."
            : "-> S1 DIVERGE: Java " + s1.indice + " frente a Python " + enPython + ".");
    }

    /** La ROI al ROI Manager y encima de la imagen, para verla al momento. */
    private void publicar(ImagePlus imp, Roi roi) {
        roi.setPosition(imp.getC(), imp.getZ(), imp.getT());
        roi.setName("surco_t" + imp.getT());

        RoiManager rm = RoiManager.getInstance();
        if (rm == null) rm = new RoiManager();
        rm.addRoi(roi);

        Overlay ov = new Overlay(roi);
        imp.setOverlay(ov);
    }
}
