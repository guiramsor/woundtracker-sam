package es.us.universidad.woundtracker.sam;

import ij.IJ;

import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.scijava.command.Command;
import org.scijava.plugin.Menu;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * P2: el puente llamando a surco.modelos con un frame sintético y un clic.
 * Reutiliza los adaptadores del banco vía cargar_disponibles().
 */
@Plugin(
    type = Command.class,
    menu = {
        @Menu(label = "Plugins", weight = 0),
        @Menu(label = "WoundTracker", weight = 1),
        @Menu(label = "Pruebas", weight = 3),
        @Menu(label = "P2 - Test surco.modelos")
    }
)
public class ApposeModelosTest implements Command {

    private static final String SCRIPT =
        "import os, sys\n" +
        "sys.path.insert(0, repo_src)\n" +
        // config.py lee SURCO_PESOS al importarse: tiene que estar puesta antes.
        "os.environ.setdefault('SURCO_PESOS', pesos)\n" +
        "import torch\n" +
        "task.outputs['torch'] = torch.__version__\n" +
        "task.outputs['cuda'] = bool(torch.cuda.is_available())\n" +
        "try:\n" +
        "    task.outputs['arquitecturas'] = list(torch.cuda.get_arch_list())\n" +
        "except Exception as e:\n" +
        "    task.outputs['arquitecturas'] = ['sin CUDA: %s' % e]\n" +
        "from surco import adaptadores, modelos\n" +
        "importados, fallidos = adaptadores.cargar_disponibles()\n" +
        "task.outputs['adaptadores'] = importados\n" +
        "task.outputs['fallidos'] = sorted(fallidos)\n" +
        "m = modelos.crear(modelo, checkpoint)\n" +
        "m.cargar()\n" +
        "imagen = modelos.imagen_de_prueba_rgb()\n" +
        // Negativo = usar el punto de prueba (el centro de la barra sintética).
        "punto = (int(clic_fila), int(clic_col)) if clic_fila >= 0 and clic_col >= 0 \\\n" +
        "        else modelos.punto_de_prueba()\n" +
        "c = m.segmentar(imagen, [punto], [1])\n" +
        "task.outputs['id'] = m.identificador\n" +
        "task.outputs['dispositivo'] = str(getattr(m, 'dispositivo', '?'))\n" +
        "task.outputs['punto'] = [int(punto[0]), int(punto[1])]\n" +
        "task.outputs['n'] = len(c)\n" +
        "task.outputs['forma'] = list(c.mascaras.shape)\n" +
        "task.outputs['puntuaciones'] = [float(x) for x in c.puntuaciones.tolist()]\n" +
        "task.outputs['pixels'] = [int(mask.sum()) for mask in c.mascaras]\n";

    @Parameter(label = "Modelo")
    private String modelo = "microsam_lm";

    @Parameter(label = "Checkpoint")
    private String checkpoint = "vit_b";

    // SciJava no deja vacío un Integer opcional: lo rellena con 0, que es una
    // esquina válida. El "no hay clic" se marca con negativo, no con nulo.
    @Parameter(label = "Clic fila (negativo = punto de prueba)")
    private int clicFila = -1;

    @Parameter(label = "Clic columna (negativo = punto de prueba)")
    private int clicCol = -1;

    @Override
    public void run() {
        IJ.log("\n//================================================================");
        IJ.log("// WOUNDTRACKER SAM - P2: surco.modelos vía Appose");
        IJ.log("//================================================================");
        IJ.log("-> Modelo:     " + modelo + " / " + checkpoint);


        long t0 = System.currentTimeMillis();
        try {
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("repo_src", Recursos.codigoDelBanco().getAbsolutePath());
            inputs.put("pesos", Recursos.pesos().getAbsolutePath());
            inputs.put("modelo", modelo);
            inputs.put("checkpoint", checkpoint);
            inputs.put("clic_fila", clicFila);
            inputs.put("clic_col", clicCol);

            IJ.log("-> Preparando el entorno de micro-sam (la primera vez tarda)...");
            Environment env = EntornoAppose.para(modelo);
            IJ.log("-> Entorno: " + env.base());

            try (Service python = env.python()) {
                python.debug(line -> IJ.log("   [worker] " + line));
                // En Windows, importar numpy dentro de una tarea cuelga el worker
                // (numpy#24290). Appose lo evita importándolo antes del bucle de E/S.
                python.init("import numpy");

                Task task = python.task(SCRIPT, inputs);
                task.waitFor();
                double segundos = (System.currentTimeMillis() - t0) / 1000.0;

                if (task.status != TaskStatus.COMPLETE) {
                    IJ.log("-> FALLO. Estado: " + task.status);
                    IJ.log("   " + task.error);
                    return;
                }

                IJ.log("-> torch " + task.outputs.get("torch")
                    + " | CUDA disponible: " + task.outputs.get("cuda")
                    + " | arquitecturas: " + task.outputs.get("arquitecturas"));
                IJ.log("-> adaptadores cargados: "
                    + ((java.util.List<?>) task.outputs.get("adaptadores")).size());
                if (!task.outputs.get("fallidos").toString().equals("[]")) {
                    IJ.log("-> fallidos: " + task.outputs.get("fallidos"));
                }
                IJ.log("-> id           = " + task.outputs.get("id"));
                IJ.log("-> dispositivo  = " + task.outputs.get("dispositivo"));
                IJ.log("-> punto        = " + task.outputs.get("punto"));
                IJ.log("-> n            = " + task.outputs.get("n"));
                IJ.log("-> forma        = " + task.outputs.get("forma"));
                IJ.log("-> puntuaciones = " + task.outputs.get("puntuaciones"));
                IJ.log("-> pixels       = " + task.outputs.get("pixels"));
                IJ.log(String.format(Locale.US, "-> P2 OK en %.1f s", segundos));
            }
        } catch (Exception e) {
            IJ.log("-> FALLO P2: " + e.getMessage());
            IJ.handleException(e);
        }
    }
}
