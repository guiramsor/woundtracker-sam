package es.us.universidad.woundtracker.sam;

import java.util.Arrays;

/**
 * El selector S1, traducción de surco/selector.py. S0 es quedarse con la
 * candidata de mayor puntuación del modelo; S1 es el criterio propio:
 *
 * 1. Puerta: sobrevive la que contiene el clic 0 y ningún clic negativo.
 * 2. Ranking: z robusto descendente, y a igualdad, elongación descendente.
 * 3. Respaldo: si no sobrevive ninguna, la de mayor z, con bandera.
 */
final class SelectorS1 {

    private static final double Z_SIN_MUESTRA = Double.NEGATIVE_INFINITY;

    static final class Eleccion {
        final int indice;
        final boolean respaldo;
        final boolean[] pasa;
        final double[] z;
        final double[] elongacion;

        Eleccion(int indice, boolean respaldo, boolean[] pasa, double[] z, double[] elongacion) {
            this.indice = indice;
            this.respaldo = respaldo;
            this.pasa = pasa;
            this.z = z;
            this.elongacion = elongacion;
        }
    }

    private SelectorS1() { }

    /**
     * @param mascaras  una por candidata, aplanadas fila a fila, 0 o distinto de 0.
     * @param gris      el plano de grises ya preparado (uint8), aplanado igual.
     * @param puntos    clics en (fila, columna); el 0 es el que manda en la puerta.
     * @param etiquetas 1 positivo, 0 fondo.
     * @param ladoVentana lado de la ventana del clic, ya escalado a esta imagen.
     */
    static Eleccion elegir(byte[][] mascaras, byte[] gris, int ancho, int alto,
        int[][] puntos, int[] etiquetas, int ladoVentana)
    {
        int k = mascaras.length;
        boolean[] pasa = new boolean[k];
        double[] z = new double[k];
        double[] elongacion = new double[k];

        for (int i = 0; i < k; i++) {
            pasa[i] = pasaLaPuerta(mascaras[i], ancho, puntos, etiquetas);
            z[i] = zRobusto(gris, mascaras[i], ancho, alto, puntos[0], ladoVentana);
            elongacion[i] = elongacion(mascaras[i], ancho, alto);
        }

        boolean hayAlguna = false;
        for (boolean p : pasa) hayAlguna |= p;
        boolean respaldo = !hayAlguna;

        // max() de Python se queda con el primero de los máximos; aquí sale
        // igual sustituyendo solo con desigualdad estricta.
        int mejor = -1;
        for (int i = 0; i < k; i++) {
            if (!respaldo && !pasa[i]) continue;
            if (mejor < 0
                || z[i] > z[mejor]
                || (z[i] == z[mejor] && elongacion[i] > elongacion[mejor]))
            {
                mejor = i;
            }
        }
        return new Eleccion(mejor, respaldo, pasa, z, elongacion);
    }

    private static boolean pasaLaPuerta(byte[] mascara, int ancho,
        int[][] puntos, int[] etiquetas)
    {
        if (!contiene(mascara, ancho, puntos[0])) return false;
        for (int i = 0; i < puntos.length; i++) {
            if (etiquetas[i] == 0 && contiene(mascara, ancho, puntos[i])) return false;
        }
        return true;
    }

    private static boolean contiene(byte[] mascara, int ancho, int[] punto) {
        return mascara[punto[0] * ancho + punto[1]] != 0;
    }

    /**
     * Cuánto se separa la candidata del fondo local, en MAD. El fondo local son
     * los píxeles de la ventana del clic que la candidata deja fuera. Con la
     * ventana plana el MAD es cero y cualquier diferencia sale infinita.
     */
    private static double zRobusto(byte[] gris, byte[] mascara, int ancho, int alto,
        int[] punto, int lado)
    {
        int fila0 = recorte(punto[0], alto, lado);
        int col0 = recorte(punto[1], ancho, lado);

        double[] dentro = new double[lado * lado];
        double[] fuera = new double[lado * lado];
        int nDentro = 0;
        int nFuera = 0;
        for (int f = 0; f < lado; f++) {
            int fila = (fila0 + f) * ancho + col0;
            for (int c = 0; c < lado; c++) {
                double valor = gris[fila + c] & 0xFF;
                if (mascara[fila + c] != 0) dentro[nDentro++] = valor;
                else fuera[nFuera++] = valor;
            }
        }
        // Hace falta muestra de las dos: ni toda dentro ni toda fuera.
        if (nDentro == 0 || nFuera == 0) return Z_SIN_MUESTRA;

        double medianaFuera = mediana(fuera, nFuera);
        double diferencia = mediana(dentro, nDentro) - medianaFuera;
        double dispersion = mad(fuera, nFuera, medianaFuera);
        if (dispersion == 0.0) {
            return diferencia == 0.0 ? 0.0 : Math.signum(diferencia) * Double.POSITIVE_INFINITY;
        }
        return diferencia / dispersion;
    }

    /** La esquina de la ventana, como el np.clip de clasicos.recortar. */
    private static int recorte(int coordenada, int tamano, int lado) {
        return Math.min(Math.max(coordenada - lado / 2, 0), tamano - lado);
    }

    /**
     * Eje mayor entre eje menor de la elipse equivalente. skimage saca los ejes
     * como 4*sqrt(autovalor); como solo importa la razón, basta la raíz.
     */
    private static double elongacion(byte[] mascara, int ancho, int alto) {
        long n = 0;
        double sumaF = 0;
        double sumaC = 0;
        for (int f = 0; f < alto; f++) {
            for (int c = 0; c < ancho; c++) {
                if (mascara[f * ancho + c] != 0) {
                    n++;
                    sumaF += f;
                    sumaC += c;
                }
            }
        }
        if (n == 0) return 0.0;   // una candidata vacía no tiene forma

        double mediaF = sumaF / n;
        double mediaC = sumaC / n;
        double vFF = 0;
        double vCC = 0;
        double vFC = 0;
        for (int f = 0; f < alto; f++) {
            for (int c = 0; c < ancho; c++) {
                if (mascara[f * ancho + c] != 0) {
                    double df = f - mediaF;
                    double dc = c - mediaC;
                    vFF += df * df;
                    vCC += dc * dc;
                    vFC += df * dc;
                }
            }
        }
        vFF /= n;
        vCC /= n;
        vFC /= n;

        double media = (vFF + vCC) / 2;
        double radio = Math.sqrt(Math.pow((vFF - vCC) / 2, 2) + vFC * vFC);
        double mayor = media + radio;
        double menor = media - radio;
        if (menor <= 0) return Double.NaN;   // eje_menor 0: el nan de Python
        return Math.sqrt(mayor / menor);
    }

    /** Como np.median: con un número par de elementos promedia los dos centrales. */
    private static double mediana(double[] valores, int n) {
        double[] copia = Arrays.copyOf(valores, n);
        Arrays.sort(copia);
        int mitad = n / 2;
        return (n % 2 == 1) ? copia[mitad] : (copia[mitad - 1] + copia[mitad]) / 2.0;
    }

    /** Desviación absoluta mediana, sin el 1.4826. Python lo omite igual. */
    private static double mad(double[] valores, int n, double centro) {
        double[] desvios = new double[n];
        for (int i = 0; i < n; i++) desvios[i] = Math.abs(valores[i] - centro);
        return mediana(desvios, n);
    }
}
