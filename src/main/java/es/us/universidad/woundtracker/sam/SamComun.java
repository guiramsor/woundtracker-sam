package es.us.universidad.woundtracker.sam;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.gui.WaitForUserDialog;
import ij.plugin.filter.ThresholdToSelection;
import ij.process.ByteProcessor;
import ij.process.FloatPolygon;
import ij.process.ImageProcessor;

import org.apposed.appose.NDArray;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

final class SamComun {

    private SamComun() { }

    /**
     * Pide una tanda de clics con la multipunto y los acumula con su etiqueta.
     * Se llama una vez por etiqueta.
     *
     * @return false si canceló o no marcó nada.
     */
    static boolean pedirClics(ImagePlus imp, String que, int etiqueta,
        List<List<Integer>> puntos, List<Integer> etiquetas)
    {
        imp.deleteRoi();
        IJ.setTool("multipoint");
        WaitForUserDialog dialogo = new WaitForUserDialog("WoundTracker SAM",
            "Marca los puntos sobre " + que + " y pulsa OK.");
        dialogo.show();

        // Se limpia siempre: por ESC el PointRoi se quedaba puesto en la imagen.
        Roi roi = imp.getRoi();
        FloatPolygon fp = (roi instanceof PointRoi) ? roi.getFloatPolygon() : null;
        imp.deleteRoi();
        if (dialogo.escPressed() || fp == null) return false;

        // Viene en (x, y); los adaptadores esperan (fila, columna).
        for (int i = 0; i < fp.npoints; i++) {
            puntos.add(Arrays.asList(
                dentro(Math.round(fp.ypoints[i]), imp.getHeight()),
                dentro(Math.round(fp.xpoints[i]), imp.getWidth())));
            etiquetas.add(etiqueta);
        }
        return fp.npoints > 0;
    }

    /** Un clic en el borde redondea fuera de la imagen, y S1 indexa directo. */
    private static int dentro(int coordenada, int tamano) {
        return Math.max(0, Math.min(coordenada, tamano - 1));
    }

    /**
     * Vista de 16 bits sobre la memoria compartida, con el orden de bytes
     * forzado al nativo: buffer() da big-endian y numpy lee el de la máquina.
     */
    static ShortBuffer vistaShort(NDArray destino) {
        return destino.buffer().order(ByteOrder.nativeOrder()).asShortBuffer();
    }

    static void copiarPixeles(ImageProcessor ip, NDArray destino) {
        vistaShort(destino).put((short[]) ip.getPixels());
    }

    /** El índice de pila del frame t, en el canal y la rodaja que se ven ahora. */
    static int indiceDe(ImagePlus imp, int t) {
        return imp.getStackIndex(imp.getC(), imp.getZ(), t);
    }

    /** Parte el bloque (k, alto, ancho) que vuelve de Python en k máscaras planas. */
    static byte[][] planos(NDArray candidatas, int k, int pixeles) {
        ByteBuffer buf = candidatas.buffer();
        byte[][] mascaras = new byte[k][pixeles];
        for (int i = 0; i < k; i++) buf.get(mascaras[i]);
        return mascaras;
    }

    static int[][] aPuntos(List<List<Integer>> puntos) {
        int[][] salida = new int[puntos.size()][2];
        for (int i = 0; i < puntos.size(); i++) {
            salida[i][0] = puntos.get(i).get(0);
            salida[i][1] = puntos.get(i).get(1);
        }
        return salida;
    }

    static int[] aEtiquetas(List<Integer> etiquetas) {
        int[] salida = new int[etiquetas.size()];
        for (int i = 0; i < etiquetas.size(); i++) salida[i] = etiquetas.get(i);
        return salida;
    }

    /** Máscara a ROI por el trazador de contornos de ImageJ. Null si está vacía. */
    static Roi aRoi(byte[] mascara, int ancho, int alto) {
        ByteProcessor bp = new ByteProcessor(ancho, alto, mascara, null);
        bp.setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE);
        return new ThresholdToSelection().convert(bp);
    }
}
