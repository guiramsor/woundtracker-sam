package es.us.universidad.woundtracker.sam;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apposed.appose.Environment;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.module.MutableModuleItem;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import ij.process.ImageProcessor;

/**
 * Segmenta el surco en la serie entera, desde el frame de los clics hasta el
 * final. Tres de los modos temporales del banco:
 *
 * - M1, re-promptear cada frame con los mismos clics. Vale para cualquier modelo.
 * - M4, lo mismo más los logits de la candidata que eligió S1 en el frame
 *   anterior. Pide un modelo que acepte prompt de máscara.
 * - M3, propagación con banco de memoria de vídeo, solo en la familia de SAM 2.
 *   Da una candidata por frame, así que ahí S1 no interviene.
 *
 * La cuarta entrada, "Solo el frame marcado", no es un modo del banco: segmenta
 * el instante de los clics y para. No confundir con el M0 del banco, que copia
 * esa máscara al resto de la ventana (correr_m0 en scripts/profundidad.py).
 *
 * El modelo se carga una vez y los frames comparten el predictor, que Appose
 * deja exportar de una tarea a las siguientes.
 */

// Sin entrada de menú: se llega desde WoundTrackerCommand.

@Plugin(type = Command.class, label = "Segmentacion del surco (clics)")
public class SamSerieCommand extends DynamicCommand {

    /** Sin letra a propósito: no es un modo del banco. Ver el javadoc. */
    static final String MODO_UNICO = "Solo el frame marcado";

    static final String MODO_M1 = "M1 - re-promptear cada frame";
    static final String MODO_M3 = "M3 - propagación de vídeo (solo SAM 2)";
    static final String MODO_M4 = "M4 - encadenar la máscara anterior";

    /** Con lo que empiezan las ROIs de esta vía, para reconocer las suyas. */
    private static final String PREFIJO_ROI = "surco_t";


    /** Se ejecuta una vez: deja el modelo cargado en el worker. */
    private static final String SCRIPT_CARGA =
        "import os, sys\n" +
        "import appose, numpy as np\n" +
        "sys.path.insert(0, repo_src)\n" +
        "os.environ.setdefault('SURCO_PESOS', pesos)\n" +
        "os.environ.setdefault('SURCO_REPOS', repos)\n" +
        "from surco import config, modelos\n" +
        // Ya escalada: el S1 de Java trabaja con esta misma ventana.
        "config.LADO_REGION_CLIC_PX = int(lado_ventana)\n" +
        // Los sigmas se escalan aquí: config.py los deriva de su rango y su
        // paso, así que Java manda el factor y no una copia de la tupla.
        "_banco = list(config.SIGMAS_SATO)\n" +
        "_escalados = [s * factor for s in _banco]\n" +
        "config.SIGMAS_SATO = tuple(max(suelo_sigma, s) for s in _escalados)\n" +
        "task.outputs['sigmas_banco'] = _banco\n" +
        "task.outputs['sigmas'] = list(config.SIGMAS_SATO)\n" +
        "task.outputs['sigmas_al_suelo'] = sum(1 for s in _escalados if s < suelo_sigma)\n" +
        "if modelo == 'sato':\n" +
        // Envoltorio para darle a candidatas_de_cresta la interfaz de un modelo.
        "    from surco import clasicos\n" +
        "    class _Sato:\n" +
        "        identificador = 'sato'\n" +
        "        dispositivo = 'cpu'\n" +
        "        def segmentar(self, rgb, puntos, etiquetas):\n" +
        "            ms, ps = clasicos.candidatas_de_cresta(rgb, puntos, etiquetas)\n" +
        "            return modelos.Candidatas(ms.astype(bool), ps)\n" +
        "    m = _Sato()\n" +
        "else:\n" +
        "    from surco import adaptadores\n" +
        // 'fallidos' dice qué adaptador no importó y por qué; sin eso el error
        // saldría como un 'modelo desconocido' a secas.
        "    _, fallidos = adaptadores.cargar_disponibles()\n" +
        "    if modelo not in modelos.registrados():\n" +
        "        raise RuntimeError('el adaptador de %s no cargó. Fallidos: %s'\n" +
        "                           % (modelo, fallidos))\n" +
        "    m = modelos.crear(modelo, checkpoint)\n" +
        "    m.cargar()\n" +
        // Lo exportado queda disponible en las tareas siguientes del mismo worker.
        "task.export(m=m, modelos=modelos, appose=appose, np=np)\n" +
        "task.outputs['dispositivo'] = str(getattr(m, 'dispositivo', '?'))\n" +
        "task.outputs['id'] = m.identificador\n" +
        "task.outputs['propaga'] = isinstance(m, modelos.ModeloConMemoria)\n" +
        "task.outputs['encadena'] = isinstance(m, modelos.ModeloConMascara)\n";

    // M1 y M4 solo cambian en la llamada al modelo: cabecera y cola son comunes.
    private static final String CABECERA_FRAME =
        "rgb = modelos.preparar(imagen.ndarray(), (float(limites[0]), float(limites[1])))\n" +
        "gris = rgb[:, :, 0]\n";

    private static final String COLA_CANDIDATAS =
        "salida = appose.NDArray('uint8', list(c.mascaras.shape))\n" +
        "salida.ndarray()[:] = c.mascaras\n" +
        "plano = appose.NDArray('uint8', list(gris.shape))\n" +
        "plano.ndarray()[:] = gris\n" +
        "task.outputs['candidatas'] = salida\n" +
        "task.outputs['gris'] = plano\n" +
        "task.outputs['n'] = len(c)\n";

    /** M1: una llamada por frame, reutilizando el modelo ya cargado. */
    private static final String SCRIPT_M1 = CABECERA_FRAME +
        "c = m.segmentar(rgb, [tuple(p) for p in puntos], list(etiquetas))\n" +
        COLA_CANDIDATAS;

    /**
     * M4: como M1, más la máscara del frame anterior. Se encadenan los logits de
     * baja resolución de la candidata que eligió S1, que es lo que espera el
     * mask_input de SAM. Se quedan en el worker; Java solo manda el índice.
     */
    private static final String SCRIPT_M4 = CABECERA_FRAME +
        // indice_previo < 0 en el frame semilla: allí todavía no hay máscara.
        "previa = logits_previos[indice_previo] if indice_previo >= 0 else None\n" +
        "c, logits = m.segmentar_con_mascara(\n" +
        "    rgb, [tuple(p) for p in puntos], list(etiquetas), previa)\n" +
        "task.export(logits_previos=logits)\n" +
        COLA_CANDIDATAS;

    /**
     * M3: una sola llamada con la secuencia entera. La semilla es el frame 0 de
     * lo que se manda, que es el de impacto, así que propaga solo hacia delante.
     */
    private static final String SCRIPT_M3 =
        "serie = imagenes.ndarray()\n" +
        "lim = (float(limites[0]), float(limites[1]))\n" +
        "preparadas = [modelos.preparar(serie[i], lim) for i in range(serie.shape[0])]\n" +
        "cs = m.propagar(preparadas, [tuple(p) for p in puntos], list(etiquetas), 0)\n" +
        "alto, ancho = preparadas[0].shape[:2]\n" +
        "salida = appose.NDArray('uint8', [len(cs), alto, ancho])\n" +
        "vista = salida.ndarray()\n" +
        "for i, c in enumerate(cs):\n" +
        "    vista[i] = c.mascaras[0].astype(np.uint8)\n" +
        "task.outputs['mascaras'] = salida\n" +
        "task.outputs['t'] = len(cs)\n";

    // Sin choices en la anotacion: la lista sale del catalogo en initialize().
    // persist = false porque SciJava restaura lo guardado DESPUES de
    // initialize(), y dejaria los checkpoints hablando del modelo anterior.
    @Parameter(label = "Modelo", callback = "alCambiarModelo", persist = false,
        description = "Quién segmenta el surco. Cada modelo vive en su propio entorno de "
            + "Python, que se construye solo la primera vez que se usa. \n* 'sato' no es un "
            + "modelo: es el filtro de cresta clásico + filtro.")
    private String modelo;

    @Parameter(label = "Checkpoint", persist = false,
        description = "Los pesos del modelo. Los grandes afinan más y tardan más; se "
            + "descargan solos la primera vez.")
    private String checkpoint;

    // Los modos salen del catálogo: solo se ofrece lo que el modelo admite.
    @Parameter(label = "Modo temporal", persist = false,
        description = "Hasta dónde llega. \n* 'Solo el frame marcado' se queda en el "
            + "instante de los clics. \n* M1 recorre la serie repitiendo esos mismos clics, "
            + "\n* M4 le añade además la máscara del frame anterior y \n* M3 usa la memoria de "
            + "vídeo de SAM 2.")
    private String modo;

    @Override
    public void initialize() {
        List<String> catalogo = EntornoAppose.modelos();
        fijarOpciones("modelo", catalogo);
        if (!catalogo.contains(modelo)) modelo = catalogo.get(0);
        alCambiarModelo();
    }

    /** Checkpoints y modos dependen del modelo: se repueblan al cambiarlo. */
    private void alCambiarModelo() {
        List<String> checkpoints = EntornoAppose.checkpointsDe(modelo);
        fijarOpciones("checkpoint", checkpoints);
        checkpoint = checkpoints.isEmpty() ? null : checkpoints.get(0);

        // El recomendado va primero en la lista.
        String recomendado = EntornoAppose.propaga(modelo) ? MODO_M3 :
            EntornoAppose.encadena(modelo) ? MODO_M4 : MODO_M1;
        
        List<String> modos = new ArrayList<>();
        modos.add(MODO_M1);
        modos.add(MODO_UNICO);
        if (EntornoAppose.encadena(modelo)) modos.add(MODO_M4);
        if (EntornoAppose.propaga(modelo)) modos.add(MODO_M3);
        modos.remove(recomendado);
        modos.add(0, recomendado);
        fijarOpciones("modo", modos);
        modo = recomendado;
    }

    @SuppressWarnings("unchecked")
    private void fijarOpciones(String parametro, List<String> opciones) {
        ((MutableModuleItem<String>) getInfo().getInput(parametro)).setChoices(opciones);
    }

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

        List<List<Integer>> puntos = new ArrayList<>();
        List<Integer> etiquetas = new ArrayList<>();
        if (!SamComun.pedirClics(imp, "el SURCO", 1, puntos, etiquetas)) {
            IJ.log("-> Cancelado: hacen falta puntos sobre el surco.");
            return;
        }
        // Donde se ha marcado el surco, que puede no ser donde estaba la serie
        // al lanzar el comando.
        int impacto = imp.getT();
        SamComun.pedirClics(imp, "el FONDO (opcional)", 0, puntos, etiquetas);
        // Los clics son coordenadas, no frames: si se navegó, el fondo se marcó
        // mirando otra imagen.
        if (imp.getT() != impacto) {
            IJ.log("-> AVISO: el fondo se marcó en el frame " + imp.getT()
                + " pero se aplica al " + impacto + ", que es donde está el surco.");
            imp.setT(impacto);
        }

        int ancho = imp.getWidth();
        int alto = imp.getHeight();
        int nFrames = Math.max(imp.getNFrames(), 1);
        int ultimo = MODO_UNICO.equals(modo) ? impacto : nFrames;
        double[] limites = ParametrosDeSerie.limitesDeLaSerie(imp);

        IJ.log("\n//================================================================");
        IJ.log("// WOUNDTRACKER SAM - SERIE COMPLETA");
        IJ.log("//================================================================");
        IJ.log("-> Imagen: " + imp.getTitle() + " (" + ancho + "x" + alto + ")");
        IJ.log("-> Modelo: " + modelo + " / " + checkpoint + "   Modo: " + modo);
        IJ.log(ultimo > impacto
            ? "-> Frames " + impacto + " a " + ultimo
            : "-> Solo el frame " + impacto);
        IJ.log("-> Clics: " + puntos.size() + " (etiquetas " + etiquetas + ")");
        IJ.log(String.format(Locale.US,
            "-> Límites de la serie: negro=%.0f blanco=%.3f", limites[0], limites[1]));
        if (!EntornoAppose.yaInstalado(modelo)) {
            IJ.log("-> AVISO: el entorno de " + modelo + " todavía no existe.");
            IJ.log("   La primera ejecución descarga varios GB y puede tardar bastante.");
        }

        List<String> validos = EntornoAppose.checkpointsDe(modelo);
        if (!validos.isEmpty() && !validos.contains(checkpoint)) {
            IJ.log("-> " + checkpoint + " no es un checkpoint de " + modelo
                + "; se usa " + validos.get(0) + ".");
            checkpoint = validos.get(0);
        }

        // Los sigmas no salen aquí: los escala Python y se registran al cargar.
        int lado = ParametrosDeSerie.ladoVentana(imp);
        double factor = ParametrosDeSerie.factorDeEscala(imp);
        double micras = ParametrosDeSerie.micrasPorPixel(imp);
        IJ.log(micras > 0
            ? String.format(Locale.US,
                "-> Calibración %.4f µm/px (factor %.2fx): ventana del clic %d px"
                    + "  (el banco, a %.3f µm/px: %d px)",
                micras, factor, lado,
                ParametrosDeSerie.MICRAS_POR_PIXEL_BANCO,
                ParametrosDeSerie.LADO_VENTANA_BANCO_PX)
            : "-> Sin calibración utilizable: se usan los valores del banco sin escalar"
                + " (ventana " + lado + " px).");
        for (String aviso : ParametrosDeSerie.avisosDeEscala(imp)) IJ.log("   " + aviso);

        long t0 = System.currentTimeMillis();
        try {
            Environment env = EntornoAppose.para(modelo);
            try (Service python = env.python()) {
                python.debug(linea -> IJ.log("   [worker] " + linea));
                python.init("import numpy");

                if (!preparar(python)) return;

                Task carga = python.task(SCRIPT_CARGA, entradasDeCarga(lado, factor));
                carga.waitFor();
                if (carga.status != TaskStatus.COMPLETE) {
                    IJ.log("-> FALLO al cargar el modelo: " + carga.error);
                    return;
                }
                IJ.log("-> " + carga.outputs.get("id")
                    + " cargado en " + carga.outputs.get("dispositivo")
                    + String.format(Locale.US, " (%.1f s)",
                        (System.currentTimeMillis() - t0) / 1000.0));
                registrarSigmas(carga);

                boolean quiereM3 = MODO_M3.equals(modo);
                boolean quiereM4 = MODO_M4.equals(modo);
                if (quiereM3 && !Boolean.TRUE.equals(carga.outputs.get("propaga"))) {
                    IJ.log("-> " + modelo + " no tiene banco de memoria: M3 no está disponible.");
                    IJ.log("   Lo tienen: " + modelosQuePropagan() + ". Si no, usa M1.");
                    return;
                }
                if (quiereM4 && !Boolean.TRUE.equals(carga.outputs.get("encadena"))) {
                    IJ.log("-> " + modelo + " no acepta prompt de máscara: M4 no está disponible.");
                    return;
                }

                Salida salida = quiereM3
                    ? propagar(python, imp, impacto, nFrames, limites, puntos, etiquetas)
                    : porFrames(python, imp, impacto, ultimo, limites, puntos, etiquetas,
                                quiereM4, lado);

                publicar(imp, salida.rois);
                double segundos = (System.currentTimeMillis() - t0) / 1000.0;
                IJ.log(String.format(Locale.US,
                    "-> %d ROIs de %d frames en %.1f s (%.2f s por frame)",
                    salida.rois.size(), salida.frames, segundos,
                    salida.frames == 0 ? 0.0 : segundos / salida.frames));
            }
        } catch (Exception e) {
            IJ.log("-> FALLO: " + e.getMessage());
            IJ.handleException(e);
        } finally {
            IJ.showProgress(1.0);
            IJ.showStatus("WoundTracker SAM: listo.");
        }
    }

    /** Las ROIs y los frames que costaron: uno puede salir con la máscara vacía. */
    private static final class Salida {
        final List<Roi> rois;
        final int frames;

        Salida(List<Roi> rois, int frames) {
            this.rois = rois;
            this.frames = frames;
        }
    }

    /**
     * Un frame cada vez, con S1 eligiendo entre las candidatas.
     *
     * @param encadenado false para M1, que manda los mismos clics y nada más;
     *                   true para M4, que añade la máscara del frame anterior.
     */
    private Salida porFrames(Service python, ImagePlus imp, int impacto, int ultimo,
        double[] limites, List<List<Integer>> puntos, List<Integer> etiquetas,
        boolean encadenado, int ladoVentana) throws Exception
    {
        int ancho = imp.getWidth();
        int alto = imp.getHeight();
        int[][] puntosS1 = SamComun.aPuntos(puntos);
        int[] etiquetasS1 = SamComun.aEtiquetas(etiquetas);
        ImageStack pila = imp.getStack();
        List<Roi> rois = new ArrayList<>();
        int conRespaldo = 0;
        int procesados = 0;
        int indicePrevio = -1;   // en el frame semilla todavía no hay máscara

        for (int t = impacto; t <= ultimo; t++) {
            if (IJ.escapePressed()) {
                IJ.log("-> Interrumpido en el frame " + t + ".");
                break;
            }
            procesados++;
            IJ.showStatus("WoundTracker SAM: frame " + t + "/" + ultimo);
            IJ.showProgress(t - impacto, ultimo - impacto + 1);

            ImageProcessor ip = pila.getProcessor(SamComun.indiceDe(imp, t));

            try (NDArray imagen = new NDArray(NDArray.DType.UINT16,
                    new NDArray.Shape(NDArray.Shape.Order.C_ORDER, alto, ancho))) {
                SamComun.copiarPixeles(ip, imagen);

                Map<String, Object> entradas = entradasDeFrame(puntos, etiquetas, limites);
                entradas.put("imagen", imagen);
                if (encadenado) entradas.put("indice_previo", indicePrevio);

                Task tarea = python.task(encadenado ? SCRIPT_M4 : SCRIPT_M1, entradas);
                tarea.waitFor();
                if (tarea.status != TaskStatus.COMPLETE) {
                    IJ.log("   frame " + t + ": FALLO - " + tarea.error);
                    continue;
                }

                int k = ((Number) tarea.outputs.get("n")).intValue();
                try (NDArray candidatas = (NDArray) tarea.outputs.get("candidatas");
                     NDArray gris = (NDArray) tarea.outputs.get("gris")) {
                    byte[][] mascaras = SamComun.planos(candidatas, k, ancho * alto);
                    byte[] plano = new byte[ancho * alto];
                    gris.buffer().get(plano);

                    SelectorS1.Eleccion s1 = SelectorS1.elegir(
                        mascaras, plano, ancho, alto, puntosS1, etiquetasS1, ladoVentana);
                    if (s1.respaldo) conRespaldo++;
                    // Se encadena lo que elige el selector: M4 con S0 y con S1
                    // son dos cadenas distintas.
                    indicePrevio = s1.indice;
                    anadir(rois, SamComun.aRoi(mascaras[s1.indice], ancho, alto), imp, t);
                }
            }
        }
        if (conRespaldo > 0) {
            IJ.log("-> S1 tiró del respaldo en " + conRespaldo + " de " + procesados
                + " frames (ninguna candidata pasó la puerta).");
        }
        return new Salida(rois, procesados);
    }

    /** M3: la secuencia entera de una vez, con la semilla en el frame de impacto. */
    private Salida propagar(Service python, ImagePlus imp, int impacto, int nFrames,
        double[] limites, List<List<Integer>> puntos, List<Integer> etiquetas) throws Exception
    {
        int ancho = imp.getWidth();
        int alto = imp.getHeight();
        int total = nFrames - impacto + 1;
        ImageStack pila = imp.getStack();
        List<Roi> rois = new ArrayList<>();

        IJ.showStatus("WoundTracker SAM: propagando " + total + " frames...");
        try (NDArray serie = new NDArray(NDArray.DType.UINT16,
                new NDArray.Shape(NDArray.Shape.Order.C_ORDER, total, alto, ancho))) {

            // Los frames van seguidos en el mismo bloque, en orden temporal.
            java.nio.ShortBuffer destino = SamComun.vistaShort(serie);
            for (int t = impacto; t <= nFrames; t++) {
                destino.put((short[]) pila.getPixels(SamComun.indiceDe(imp, t)));
            }

            Map<String, Object> entradas = entradasDeFrame(puntos, etiquetas, limites);
            entradas.put("imagenes", serie);

            Task tarea = python.task(SCRIPT_M3, entradas);
            tarea.waitFor();
            if (tarea.status != TaskStatus.COMPLETE) {
                IJ.log("-> FALLO en la propagación: " + tarea.error);
                return new Salida(rois, 0);
            }

            int devueltos = ((Number) tarea.outputs.get("t")).intValue();
            IJ.log("-> Propagados " + devueltos + " frames (S1 no interviene en M3:"
                + " el predictor de vídeo da una sola candidata por frame).");

            try (NDArray mascaras = (NDArray) tarea.outputs.get("mascaras")) {
                byte[][] planos = SamComun.planos(mascaras, devueltos, ancho * alto);
                for (int i = 0; i < devueltos; i++) {
                    anadir(rois, SamComun.aRoi(planos[i], ancho, alto), imp, impacto + i);
                }
            }
            return new Salida(rois, devueltos);
        }
    }

    private void anadir(List<Roi> rois, Roi roi, ImagePlus imp, int t) {
        if (roi == null) {
            IJ.log("   frame " + t + ": la máscara está vacía.");
            return;
        }
        roi.setPosition(imp.getC(), imp.getZ(), t);
        roi.setName(String.format(PREFIJO_ROI + "%02d", t));
        rois.add(roi);
    }

    /**
     * Al ROI Manager y como overlay. Antes retira las de una ejecución anterior,
     * que si no se apilan con el mismo nombre. Solo las suyas, para no borrar lo
     * que haya puesto el usuario.
     */
    private void publicar(ImagePlus imp, List<Roi> rois) {
        RoiManager rm = RoiManager.getRoiManager();
        retirarAnteriores(rm);
        Overlay overlay = new Overlay();
        for (Roi roi : rois) {
            rm.addRoi(roi);
            overlay.add(roi);
        }
        imp.setOverlay(overlay);
    }

    /**
     * Corre el fragmento de preparación de la receta antes de cargar el modelo.
     *
     * @return false si algo falló y no tiene sentido seguir.
     */
    private boolean preparar(Service python) throws Exception {
        String script = EntornoAppose.scriptPreparacion(modelo);
        if (script == null) return true;   // se lo trae él solo

        IJ.showStatus("WoundTracker SAM: comprobando lo que necesita " + modelo + "...");
        Map<String, Object> entradas = new HashMap<>();
        entradas.put("checkpoint", checkpoint);
        entradas.put("dir_pesos", Recursos.pesos().getAbsolutePath());
        entradas.put("dir_repos", Recursos.repos().getAbsolutePath());

        Task tarea = python.task(script, entradas);
        tarea.listen(evento -> {
            if (evento.message != null) IJ.log("   " + evento.message);
        });
        tarea.waitFor();

        if (tarea.status != TaskStatus.COMPLETE) {
            IJ.log("-> FALLO preparando " + modelo + ": " + tarea.error);
            return false;
        }
        List<?> hecho = (List<?>) tarea.outputs.get("hecho");
        if (hecho != null && !hecho.isEmpty()) {
            IJ.log("-> Traído (solo la primera vez): " + hecho);
        }
        return true;
    }

    private static void retirarAnteriores(RoiManager rm) {
        List<Integer> viejas = new ArrayList<>();
        for (int i = 0; i < rm.getCount(); i++) {
            String nombre = rm.getName(i);
            if (nombre != null && nombre.startsWith(PREFIJO_ROI)) viejas.add(i);
        }
        // Con la selección vacía, el Delete del ROI Manager abre un diálogo
        // preguntando si borrar la lista entera.
        if (viejas.isEmpty()) return;

        int[] indices = new int[viejas.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = viejas.get(i);
        rm.setSelectedIndexes(indices);
        rm.runCommand("Delete");
        rm.deselect();
    }

    /** Los del catálogo que pueden hacer M3, para no nombrarlos a mano. */
    private static List<String> modelosQuePropagan() {
        List<String> conMemoria = new ArrayList<>();
        for (String m : EntornoAppose.modelos()) {
            if (EntornoAppose.propaga(m)) conMemoria.add(m);
        }
        return conMemoria;
    }

    /** Lo que necesitan por igual los tres modos. */
    private Map<String, Object> entradasDeFrame(List<List<Integer>> puntos,
        List<Integer> etiquetas, double[] limites)
    {
        Map<String, Object> entradas = new HashMap<>();
        entradas.put("puntos", puntos);
        entradas.put("etiquetas", etiquetas);
        entradas.put("limites", Arrays.asList(limites[0], limites[1]));
        return entradas;
    }

    private Map<String, Object> entradasDeCarga(int ladoVentana, double factor)
        throws IOException
    {
        Map<String, Object> entradas = new HashMap<>();
        entradas.put("repo_src", Recursos.codigoDelBanco().getAbsolutePath());
        entradas.put("pesos", Recursos.pesos().getAbsolutePath());
        entradas.put("repos", Recursos.repos().getAbsolutePath());
        entradas.put("modelo", modelo);
        entradas.put("checkpoint", checkpoint);
        entradas.put("lado_ventana", ladoVentana);
        entradas.put("factor", factor);
        entradas.put("suelo_sigma", ParametrosDeSerie.SIGMA_MINIMO);
        return entradas;
    }

    /** Las escalas del filtro de cresta como han quedado; las deriva Python. */
    private static void registrarSigmas(Task carga) {
        IJ.log("-> Sigmas de cresta: " + carga.outputs.get("sigmas")
            + "  (el banco: " + carga.outputs.get("sigmas_banco") + ")");
        int alSuelo = ((Number) carga.outputs.get("sigmas_al_suelo")).intValue();
        if (alSuelo > 0) {
            IJ.log(String.format(Locale.US,
                "   AVISO: %d escalas caían por debajo de %.1f y se han subido a ese suelo,"
                + " donde el filtro ya responde al ruido. Solo afecta a sato.",
                alSuelo, ParametrosDeSerie.SIGMA_MINIMO));
        }
    }
}
