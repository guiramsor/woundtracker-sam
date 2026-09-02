package es.us.universidad.woundtracker.sam;

import ij.IJ;

import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.scijava.command.Command;
import org.scijava.plugin.Menu;
import org.scijava.plugin.Plugin;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * P1: prueba mínima del puente Java-Python con Appose, sin ningún modelo.
 *
 * Manda una lista de números al worker de Python y la recibe doblada. No tiene
 * parámetros: el entorno es mínimo y fijo. Está para separar un fallo del
 * puente de uno del modelo cuando P2 no funcione.
 */
@Plugin(
    type = Command.class,
    menu = {
        @Menu(label = "Plugins", weight = 0),
        @Menu(label = "WoundTracker", weight = 1),
        @Menu(label = "Pruebas", weight = 3),
        @Menu(label = "P1 - Test puente Appose")
    }
)
public class ApposeBridgeTest implements Command {

    private static final String ENTORNO = "woundtracker-p1";

    /** Las entradas de la tarea llegan como variables del script. */
    private static final String SCRIPT =
        "import sys\n" +
        "task.outputs['doblados'] = [2 * n for n in numeros]\n" +
        "task.outputs['python'] = sys.version.split()[0]\n" +
        "task.outputs['ejecutable'] = sys.executable\n";

    @Override
    public void run() {
        IJ.log("\n//================================================================");
        IJ.log("// WOUNDTRACKER SAM - P1: TEST DEL PUENTE APPOSE");
        IJ.log("//================================================================");

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("numeros", Arrays.asList(3, 1, 4, 1, 5));
        IJ.log("-> Enviando a Python: " + inputs.get("numeros"));

        long t0 = System.currentTimeMillis();
        try {
            Environment env = EntornoAppose.minimo(ENTORNO);
            IJ.log("-> Entorno: " + env.base());

            try (Service python = env.python()) {
                python.debug(line -> IJ.log("   [worker] " + line));

                Task task = python.task(SCRIPT, inputs);
                task.waitFor();
                double segundos = (System.currentTimeMillis() - t0) / 1000.0;

                if (task.status != TaskStatus.COMPLETE) {
                    IJ.log("-> FALLO. Estado: " + task.status);
                    IJ.log("   " + task.error);
                    return;
                }
                IJ.log("-> Recibido: " + task.outputs.get("doblados"));
                IJ.log("-> Python " + task.outputs.get("python")
                    + " en " + task.outputs.get("ejecutable"));
                IJ.log(String.format(Locale.US, "-> PUENTE OK en %.1f s.", segundos));
            }
        } catch (Exception e) {
            IJ.log("-> FALLO del puente: " + e.getMessage());
            IJ.handleException(e);
        }
    }
}
