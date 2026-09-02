package es.us.universidad.woundtracker;

import es.us.universidad.woundtracker.sam.SamSerieCommand;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.plugin.ChannelSplitter;

import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.plugin.Menu;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.NumberWidget;

/**
 * Punto de entrada del plugin. Prepara la serie y lanza una de las dos vías.
 *
 * La saturación se elige aquí y se le pasa a la vía celular: su conversión a 8
 * bits usa el rango de visualización, así que acaba fijando el umbral.
 */
@Plugin(
    type = Command.class,
    menu = {
        @Menu(label = "Plugins", weight = 0),
        @Menu(label = "WoundTracker", weight = 1)
    },
    priority = Double.MAX_VALUE
)
public class WoundTrackerCommand implements Command {

    static final String VIA_CLASICA = "Segmentacion celular (clasica, por umbral)";
    static final String VIA_SURCO = "Segmentacion del surco (por clics)";

    @Parameter private CommandService commandService;

    @Parameter(label = "<html><b>PREPARACIÓN DE LA SERIE</b></html>",
        visibility = ItemVisibility.MESSAGE, persist = false, required = false)
    private String cabeceraPreparacion;

    @Parameter(label = "Proyección Z",
        description = "Colapsa el Z-stack.")
    private boolean aplicarProyeccion = true;

    @Parameter(label = "Método de proyección",
        choices = { "Maximum Intensity", "Average Intensity", "Sum Slices",
                    "Minimum Intensity", "Standard Deviation", "Median" })
    private String metodoProyeccion = "Maximum Intensity";

    @Parameter(label = "Auto contraste",
        description = "Ajusta el rango de visualización para poder ver el surco.")
    private boolean aplicarContraste = true;

    @Parameter(label = "Saturación (%)", min = "0", max = "5", stepSize = "0.05",
        style = NumberWidget.SLIDER_STYLE,
        description = "Qué porcentaje de píxeles se deja quemar al estirar el contraste. "
            + "Más saturación, más contraste y menos detalle en los extremos. Solo afecta "
            + "a lo que se ve.")
    private double saturacion = 0.35;

    @Parameter(label = "<html><b>VÍA DE SEGMENTACIÓN</b></html>",
        visibility = ItemVisibility.MESSAGE, persist = false, required = false)
    private String cabeceraVia;

    @Parameter(label = "Vía", choices = { VIA_CLASICA, VIA_SURCO },
        description = "Qué se va a segmentar. \n* La clásica busca células por umbral y "
            + "morfología.\n* La del surco busca la marca del láser a partir de tus clics.")
    private String via = VIA_SURCO;

    @Override
    public void run() {
        ImagePlus original = WindowManager.getCurrentImage();
        if (original == null) {
            IJ.error("WoundTracker", "No hay ninguna imagen abierta.");
            return;
        }

        ImagePlus preparada = preparar(original);
        if (preparada != original) {
            preparada.show();
        }

        // ----------------------
        
        if (VIA_CLASICA.equals(via)) {
            commandService.run(WoundHealingTracker.class, true, "saturatedPercent", saturacion);
        } else {
            commandService.run(SamSerieCommand.class, true);
        }
        
        // ----------------------
    }

    /** Duplica solo si hay Z que proyectar; si no, trabaja sobre la original. */
    private ImagePlus preparar(ImagePlus original) {
        ImagePlus imagen = original;

        if (aplicarProyeccion && original.getNSlices() > 1) {
            imagen = original.duplicate();
            if (imagen.getNChannels() > 1) {
                imagen = ChannelSplitter.split(imagen)[0];
            }
            imagen = ImageProcessorService.performZProjection(imagen, metodoProyeccion);
            imagen.setTitle(original.getShortTitle() + " - preparada");
        }

        if (aplicarContraste) {
            ImageProcessorService.applyAutoContrast(imagen, saturacion);
        }
        return imagen;
    }
}
