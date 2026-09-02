package es.us.universidad.woundtracker.sam;

import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Lo que se saca de la serie antes de llamar al modelo: los límites de
 * normalización y la ventana del clic.
 *
 * Las escalas del filtro de cresta se ajustan en Python, donde config.py las
 * deriva de su rango y su paso. De aquí solo sale el factor.
 */
final class ParametrosDeSerie {

    /** config.PERCENTIL_BLANCO_MODELOS del banco. */
    private static final double PERCENTIL_BLANCO = 99.9;

    /** µm por píxel del banco: 100x OIL TIRF con binning 2x2. */
    static final double MICRAS_POR_PIXEL_BANCO = 0.129;

    /** config.LADO_REGION_CLIC_PX, a la escala de arriba. */
    static final int LADO_VENTANA_BANCO_PX = 128;

    /** Por debajo de 1 el filtro de cresta ya no responde. */
    static final double SIGMA_MINIMO = 1.0;

    /** Cuánto puede alejarse la calibración de la del banco antes de avisar. */
    private static final double TOLERANCIA = 0.01;

    private ParametrosDeSerie() { }

    // -----------------------------------------------------------------------
    // Normalización: de uint16 a las tres bandas uint8 que esperan los modelos
    // -----------------------------------------------------------------------

    /**
     * Punto negro y punto blanco de los N frames. Por serie y no por frame:
     * frame a frame se borra la diferencia entre los débiles y los brillantes.
     */
    static double[] limitesDeLaSerie(ImagePlus imp) {
        int n = Math.max(imp.getNFrames(), 1);
        int[] indices = new int[n];
        for (int t = 1; t <= n; t++) indices[t - 1] = SamComun.indiceDe(imp, t);
        return limites(imp, indices);
    }

    /** Por histograma de 65536 bins, para no ordenar millones de píxeles. */
    private static double[] limites(ImagePlus imp, int[] indices) {
        ImageStack pila = imp.getStack();
        long[] histograma = new long[65536];
        for (int indice : indices) {
            for (short pixel : (short[]) pila.getPixels(indice)) {
                histograma[pixel & 0xFFFF]++;
            }
        }
        long total = (long) indices.length * imp.getWidth() * imp.getHeight();
        // El mínimo es el primer bin no vacío.
        return new double[] { valorEnRango(histograma, 0), percentil(histograma, total) };
    }

    /**
     * El percentil como lo calcula np.percentile.
     */
    private static double percentil(long[] histograma, long total) {
        double indiceVirtual = (PERCENTIL_BLANCO / 100.0) * (total - 1);
        long inferior = (long) Math.floor(indiceVirtual);
        double fraccion = indiceVirtual - inferior;
        int valorInferior = valorEnRango(histograma, inferior);
        if (fraccion == 0.0) return valorInferior;
        return valorInferior + fraccion * (valorEnRango(histograma, inferior + 1) - valorInferior);
    }

    /** El valor que ocuparía esa posición si estuvieran todos ordenados. */
    private static int valorEnRango(long[] histograma, long rango) {
        long acumulado = 0;
        for (int valor = 0; valor < histograma.length; valor++) {
            acumulado += histograma[valor];
            if (acumulado > rango) return valor;
        }
        return histograma.length - 1;
    }

    // -----------------------------------------------------------------------
    // Los parámetros en píxeles del banco, en la rejilla de esta imagen
    // -----------------------------------------------------------------------

    /**
     * Cuántos píxeles de esta imagen ocupa lo que en el banco ocupaba uno. Sin
     * calibración utilizable devuelve 1 y se dejan los valores del banco.
     */
    static double factorDeEscala(ImagePlus imp) {
        double micras = micrasPorPixel(imp);
        return micras == 0 ? 1 : MICRAS_POR_PIXEL_BANCO / micras;
    }

    static int ladoVentana(ImagePlus imp) {
        int lado = (int) Math.round(LADO_VENTANA_BANCO_PX * factorDeEscala(imp));
        return acotar(lado, imp);
    }

    /**
     * Avisos sobre el escalado, o nada si la imagen viene a la calibración del
     * banco. Escalar es una aproximación: config.py pide rehacer estos
     * parámetros mirando la geometría de los datos nuevos.
     */
    static List<String> avisosDeEscala(ImagePlus imp) {
        List<String> avisos = new ArrayList<>();
        double micras = micrasPorPixel(imp);
        // Sin calibrar se dan por buenos los valores del banco a ciegas.
        if (micras == 0) {
            avisos.add(String.format(Locale.US,
                "AVISO: la imagen no trae calibración utilizable, así que la ventana del clic"
                + " y los sigmas se quedan en los valores del banco. Si el píxel no mide"
                + " %.3f µm, calíbrala en Image > Properties.", MICRAS_POR_PIXEL_BANCO));
            return avisos;
        }

        double factor = factorDeEscala(imp);
        if (Math.abs(micras - MICRAS_POR_PIXEL_BANCO) / MICRAS_POR_PIXEL_BANCO > TOLERANCIA) {
            avisos.add(String.format(Locale.US,
                "AVISO: la ventana del clic y los sigmas de cresta se fijaron a %.3f µm/px."
                + " Aquí se escalan por %.2fx como aproximación; el banco pide rehacerlos"
                + " mirando la geometría de los datos nuevos, no multiplicando.",
                MICRAS_POR_PIXEL_BANCO, factor));
        }

        return avisos;
    }

    private static int acotar(int lado, ImagePlus imp) {
        return Math.max(16, Math.min(lado, Math.min(imp.getWidth(), imp.getHeight())));
    }

    /** 0 si no está calibrada o si la unidad no se reconoce. */
    static double micrasPorPixel(ImagePlus imp) {
        Calibration cal = imp.getCalibration();
        if (cal == null || !cal.scaled()) return 0;
        double factor = aMicras(cal.getUnit());
        return factor == 0 ? 0 : cal.pixelWidth * factor;
    }

    private static double aMicras(String unidad) {
        if (unidad == null) return 0;
        switch (unidad.trim().toLowerCase()) {
            case "micron": case "microns": case "um": case "µm": return 1;
            case "nm": case "nanometer": case "nanometre": return 0.001;
            case "mm": case "millimeter": case "millimetre": return 1000;
            case "cm": case "centimeter": case "centimetre": return 10000;
            case "m": case "meter": case "metre": return 1000000;
            // "pixel", "inch" y las demás: mejor no escalar que escalar mal.
            default: return 0;
        }
    }
}
