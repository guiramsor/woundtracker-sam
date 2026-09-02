package es.us.universidad.woundtracker.sam;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.JarURLConnection;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Enumeration;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * Dónde guarda el plugin sus cosas: pesos, clones y el código del banco.
 *
 * Todo cuelga de ~/.local/share/woundtracker. WOUNDTRACKER_HOME lo mueve, que
 * los pesos ocupan varios GB.
 */
final class Recursos {

    /** El paquete Python que viaja empotrado en el jar. */
    private static final String PAQUETE = "surco";

    private Recursos() { }

    static File base() {
        String propia = System.getenv("WOUNDTRACKER_HOME");
        if (propia != null && !propia.isEmpty()) return new File(propia);
        return new File(System.getProperty("user.home"), ".local/share/woundtracker");
    }

    /** Los checkpoints, que se descargan solos. */
    static File pesos() {
        return creada("pesos");
    }

    /** Lo que hay que clonar por no ser instalable. De momento, TinySAM. */
    static File repos() {
        return creada("repos");
    }

    /** micro-sam se baja sus pesos solo, así que la carpeta tiene que existir ya. */
    private static File creada(String nombre) {
        File carpeta = new File(base(), nombre);
        carpeta.mkdirs();
        return carpeta;
    }

    /**
     * La carpeta que va a sys.path para poder importar surco. Se reextrae en
     * cada arranque para no dejar viva la copia de una versión anterior.
     * SURCO_SRC apunta al árbol del banco y se salta la copia del jar.
     */
    static File codigoDelBanco() throws IOException {
        String vivo = System.getenv("SURCO_SRC");
        if (vivo != null && !vivo.isEmpty()) return new File(vivo);

        // Un fichero y no la carpeta: no todo jar trae entradas de directorio.
        URL marca = Recursos.class.getResource("/" + PAQUETE + "/__init__.py");
        if (marca == null) {
            throw new IOException("este jar no trae el paquete " + PAQUETE
                + ": se compiló sin src/main/resources");
        }

        // Se borra después de saber que hay con qué reemplazarlo: sobrescribir
        // dejaría vivo un módulo que ya no viene, pero borrar y luego fallar
        // dejaría al usuario sin la copia que le funcionaba.
        File destino = new File(base(), "py");
        borrar(new File(destino, PAQUETE));

        if ("jar".equals(marca.getProtocol())) desdeJar(marca, destino);
        else desdeCarpeta(carpetaDe(marca).getParentFile(), new File(destino, PAQUETE));
        return destino;
    }

    /** Instalado en Fiji: los .py son entradas del jar. */
    private static void desdeJar(URL marca, File destino) throws IOException {
        JarURLConnection conexion = (JarURLConnection) marca.openConnection();
        // Sin esto cerraríamos el JarFile que la JVM cachea para el propio plugin.
        conexion.setUseCaches(false);
        try (JarFile jar = conexion.getJarFile()) {
            Enumeration<JarEntry> entradas = jar.entries();
            while (entradas.hasMoreElements()) {
                JarEntry entrada = entradas.nextElement();
                if (entrada.isDirectory()) continue;
                if (!entrada.getName().startsWith(PAQUETE + "/")) continue;
                try (InputStream flujo = jar.getInputStream(entrada)) {
                    copiar(flujo, new File(destino, entrada.getName()));
                }
            }
        }
    }

    /** Ejecutando desde Eclipse: están sueltos en target/classes. */
    private static void desdeCarpeta(File origen, File destino) throws IOException {
        File[] hijos = origen.listFiles();
        if (hijos == null) return;
        for (File hijo : hijos) {
            File salida = new File(destino, hijo.getName());
            if (hijo.isDirectory()) {
                desdeCarpeta(hijo, salida);
            } else try (InputStream flujo = new FileInputStream(hijo)) {
                copiar(flujo, salida);
            }
        }
    }

    /** Por toURI() y no por getPath(): las rutas con espacios vienen escapadas. */
    private static File carpetaDe(URL url) throws IOException {
        try {
            return new File(url.toURI());
        } catch (URISyntaxException e) {
            throw new IOException("ruta ilegible: " + url, e);
        }
    }

    private static void borrar(File fichero) {
        File[] hijos = fichero.listFiles();
        if (hijos != null) for (File hijo : hijos) borrar(hijo);
        fichero.delete();
    }

    private static void copiar(InputStream origen, File destino) throws IOException {
        destino.getParentFile().mkdirs();
        Files.copy(origen, destino.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }
}
