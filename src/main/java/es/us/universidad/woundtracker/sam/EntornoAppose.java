package es.us.universidad.woundtracker.sam;

import ij.IJ;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.util.Environments;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Catálogo de la vía SAM: qué entorno, qué checkpoints y qué le falta a cada
 * modelo.
 *
 * Los entornos se construyen aquí en vez de reutilizar los del banco, uno por
 * modelo, porque sus versiones de torch y sus repositorios no conviven. Appose
 * los cachea por nombre, así que solo se instalan la primera vez.
 *
 * El que necesita pesos o clones lleva un fragmento de Python que se los trae.
 */
final class EntornoAppose {

    /** Índice de ruedas de torch compiladas para Blackwell y anteriores. */
    private static final String INDICE_CU128 = "https://download.pytorch.org/whl/cu128";

    private EntornoAppose() { }

    private static final class Receta {
        final String nombre;
        final String conda;
        final String condaGpu;      // dependencias conda solo de la variante CUDA
        final String pypi;          // se sustituye por el índice de torch
        final String preparacion;   // Python que consigue pesos, clones o paquetes
        // Copia de lo que el adaptador declara por herencia en Python: el
        // dialogo lo necesita antes de que el entorno exista.
        final boolean encadena;     // M4: acepta prompt de mascara
        final boolean propaga;      // M3: tiene banco de memoria de video
        final List<String> checkpoints;

        Receta(String nombre, String conda, String condaGpu, String pypi,
            String preparacion, boolean encadena, boolean propaga,
            String... checkpoints)
        {
            this.nombre = nombre;
            this.conda = conda;
            this.condaGpu = condaGpu;
            this.pypi = pypi;
            this.preparacion = preparacion;
            this.encadena = encadena;
            this.propaga = propaga;
            this.checkpoints = Arrays.asList(checkpoints);
        }
    }

    /** Los modelos que el plugin sabe montar, en el orden del desplegable. */
    private static final Map<String, Receta> CATALOGO = new LinkedHashMap<>();

    /**
     * Por conda-forge, que es donde están sus dependencias (elf, z5py). Su torch
     * sale de conda, con 'pytorch-gpu' para pedir CUDA.
     *
     * numpy y tifffile van declarados aunque micro_sam los arrastre: son lo que
     * importa surco.modelos pasando por datos.py.
     */
    private static final Receta MICROSAM = new Receta(
        "woundtracker-microsam",
        "micro_sam = \"1.8.7.*\"\nnumpy = \"*\"\ntifffile = \"*\"\n",
        "pytorch-gpu = \"*\"\n",
        "appose = \"==0.12.0\"\n",
        "",
        true, false,
        "vit_b", "vit_t", "vit_l");

    /**
     * torch llega por pip desde el índice de CUDA, así que no hace falta
     * [system-requirements]: pip no consulta __cuda.
     *
     * sam2 no está en la receta porque solo se publica como fuente y su
     * pyproject pide torch para construir, que durante la resolución todavía no
     * existe. Se instala en la preparación. Los pesos se piden a Hugging Face
     * con los identificadores que declara el propio paquete.
     */
    private static final Receta SAM21 = new Receta(
        "woundtracker-sam21",
        "numpy = \"*\"\nscikit-image = \"*\"\nscipy = \"*\"\ntifffile = \"*\"\n"
            + "setuptools = \"*\"\nwheel = \"*\"\n",
        "",
        "torch = { version = \"*\", index = \"%1$s\" }\n" +
        "torchvision = { version = \"*\", index = \"%1$s\" }\n" +
        "huggingface_hub = \"*\"\n" +
        "appose = \"==0.12.0\"\n",
        "if importlib.util.find_spec('sam2') is None:\n" +
        "    instalar_pip('sam2==1.1.0', '--no-build-isolation')\n" +
        "nombre = 'sam2.1_hiera_' + checkpoint + '.pt'\n" +
        "destino = os.path.join(dir_pesos, nombre)\n" +
        "if not os.path.exists(destino):\n" +
        "    from huggingface_hub import hf_hub_download\n" +
        "    task.update(message='descargando ' + nombre + ' de Hugging Face...')\n" +
        "    cache = hf_hub_download(\n" +
        "        repo_id='facebook/sam2.1-hiera-' + checkpoint.replace('_', '-'),\n" +
        "        filename=nombre)\n" +
        "    shutil.copy(cache, destino)\n" +
        "    hecho.append(nombre)\n",
        true, true,
        "large", "tiny", "small", "base_plus");

    /**
     * No es instalable: su repositorio no trae setup.py ni pyproject.toml, así
     * que la preparación lo clona y su adaptador mete la ruta en sys.path.
     */
    private static final Receta TINYSAM = new Receta(
        "woundtracker-tinysam",
        "numpy = \"*\"\nscikit-image = \"*\"\nscipy = \"*\"\ntifffile = \"*\"\n",
        "",
        "torch = { version = \"*\", index = \"%1$s\" }\n" +
        "torchvision = { version = \"*\", index = \"%1$s\" }\n" +
        "timm = \"*\"\n" +
        "appose = \"==0.12.0\"\n",
        "repo = os.path.join(dir_repos, 'TinySAM')\n" +
        "if not os.path.isdir(repo):\n" +
        "    task.update(message='clonando TinySAM...')\n" +
        "    os.makedirs(dir_repos, exist_ok=True)\n" +
        "    r = subprocess.run(\n" +
        "        ['git', 'clone', '--depth', '1',\n" +
        "         'https://github.com/xinghaochen/TinySAM.git', repo],\n" +
        "        stdin=subprocess.DEVNULL, capture_output=True, text=True)\n" +
        "    if r.returncode != 0:\n" +
        "        raise RuntimeError('no se pudo clonar TinySAM; hace falta git en el "
            + "PATH.\\n' + r.stderr[-800:])\n" +
        "    hecho.append('clon de TinySAM')\n" +
        "nombre = {'vit_t': 'tinysam_42.3.pth'}[checkpoint]\n" +
        "destino = os.path.join(dir_pesos, nombre)\n" +
        "if not os.path.exists(destino):\n" +
        "    descargar('https://github.com/xinghaochen/TinySAM/releases/download/3.0/'\n" +
        "              + nombre, destino, nombre)\n",
        true, false,
        "vit_t");

    /**
     * El finalista de radiología. Como TinySAM, no es instalable: su repositorio
     * no trae setup.py ni pyproject.toml, así que la preparación lo clona y su
     * adaptador mete esa ruta en sys.path.
     *
     * Sus pesos son el único caso del catálogo que no se descarga solo. Están en
     * Google Drive, detrás de un aviso de antivirus que hay que sortear a mano,
     * y el fichero de sam_med2d ocupa 2,4 GB. La preparación comprueba que esté
     * y dice dónde ponerlo si falta, en vez de fingir que puede traerlo.
     *
     * albumentations y opencv-python son dependencias suyas que el repositorio
     * no declara. pillow va con ellas por conda y no por pip porque instalarlas
     * en pip por separado deja una versión que el banco tuvo que reinstalar;
     * resolviéndolas todas juntas ese conflicto no llega a darse.
     *
     * Solo se ofrece 'sam_med2d', que es el que lleva encoder_adapter activado y
     * lo que distingue a SAM-Med2D del vit_b de SAM 1 sobre el que se construye.
     * El otro, 'ft_sam', es un SAM afinado solo en el decodificador.
     */
    private static final Receta SAMMED2D = new Receta(
        "woundtracker-sammed2d",
        "numpy = \"*\"\nscikit-image = \"*\"\nscipy = \"*\"\ntifffile = \"*\"\n"
            + "pillow = \"*\"\n",
        "",
        "torch = { version = \"*\", index = \"%1$s\" }\n" +
        "torchvision = { version = \"*\", index = \"%1$s\" }\n" +
        "albumentations = \"*\"\n" +
        "opencv-python = \"*\"\n" +
        "appose = \"==0.12.0\"\n",
        "repo = os.path.join(dir_repos, 'SAM-Med2D')\n" +
        "if not os.path.isdir(repo):\n" +
        "    task.update(message='clonando SAM-Med2D...')\n" +
        "    os.makedirs(dir_repos, exist_ok=True)\n" +
        "    r = subprocess.run(\n" +
        "        ['git', 'clone', '--depth', '1',\n" +
        "         'https://github.com/OpenGVLab/SAM-Med2D.git', repo],\n" +
        "        stdin=subprocess.DEVNULL, capture_output=True, text=True)\n" +
        "    if r.returncode != 0:\n" +
        "        raise RuntimeError('no se pudo clonar SAM-Med2D; hace falta git en el "
            + "PATH.\\n' + r.stderr[-800:])\n" +
        "    hecho.append('clon de SAM-Med2D')\n" +
        "nombre = {'sam_med2d': 'sam-med2d_b.pth'}[checkpoint]\n" +
        "destino = os.path.join(dir_pesos, nombre)\n" +
        "if not os.path.exists(destino):\n" +
        "    raise RuntimeError(\n" +
        "        'falta ' + nombre + ' en ' + dir_pesos + '. Es el unico peso del "
            + "catalogo'\n" +
        "        ' que no se descarga solo: esta en Google Drive, detras de un aviso "
            + "de'\n" +
        "        ' antivirus. Hay que bajarlo a mano del repositorio oficial de "
            + "SAM-Med2D'\n" +
        "        ' y dejarlo en esa carpeta.')\n",
        true, false,
        "sam_med2d");

    /**
     * El filtro de cresta. No es un modelo, pero candidatas_de_cresta devuelve
     * candidatas con puntuación, así que el mismo S1 vale para los dos lados.
     * Su entorno no lleva torch. Lo configuran sus escalas, que salen de
     * config.SIGMAS_SATO escaladas por la calibración de la serie.
     */
    private static final Receta SATO = new Receta(
        "woundtracker-sato",
        "numpy = \"*\"\nscikit-image = \"*\"\nscipy = \"*\"\ntifffile = \"*\"\n",
        "",
        "appose = \"==0.12.0\"\n",
        "",
        false, false,
        "(no usa checkpoint: se configura con sus escalas)");

    /** Sin modelo: solo el intérprete y el puente. */
    private static final Receta MINIMO = new Receta(
        "woundtracker-minimo", "", "", "appose = \"==0.12.0\"\n", "", false, false);

    static {
        CATALOGO.put("microsam_lm", MICROSAM);
        CATALOGO.put("sam21", SAM21);
        CATALOGO.put("tinysam", TINYSAM);
        CATALOGO.put("sammed2d", SAMMED2D);
        CATALOGO.put("sato", SATO);
    }

    /**
     * Las dos herramientas que usan las recetas, y la lista de lo traído.
     *
     * stdin al vacío en todo lo que lance procesos: heredarlo sería heredar el
     * canal de Appose, y cualquier lectura ahí cuelga en silencio.
     */
    private static final String CABECERA_PREPARACION =
        "import importlib.util, os, shutil, subprocess, sys, urllib.request\n" +
        "os.makedirs(dir_pesos, exist_ok=True)\n" +
        "hecho = []\n" +
        "def instalar_pip(paquete, *extra):\n" +
        "    orden = [sys.executable, '-m', 'pip', 'install', '--no-input',\n" +
        "             '--disable-pip-version-check'] + list(extra) + [paquete]\n" +
        "    entorno = dict(os.environ, SAM2_BUILD_CUDA='0', SAM2_BUILD_ALLOW_ERRORS='1')\n" +
        "    p = subprocess.Popen(orden, stdin=subprocess.DEVNULL,\n" +
        "                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n" +
        "                         text=True, env=entorno)\n" +
        "    ultimas = []\n" +
        "    for linea in p.stdout:\n" +
        "        linea = linea.rstrip()\n" +
        "        ultimas = (ultimas + [linea])[-40:]\n" +
        "        if linea.startswith(('Collecting', 'Downloading', 'Building',\n" +
        "                             'Installing', 'Successfully', 'ERROR')):\n" +
        "            task.update(message='pip: ' + linea)\n" +
        "    if p.wait() != 0:\n" +
        "        raise RuntimeError('no se pudo instalar ' + paquete + ':\\n'\n" +
        "                           + '\\n'.join(ultimas))\n" +
        "    hecho.append(paquete)\n" +
        "def descargar(url, destino, etiqueta):\n" +
        // A un fichero parcial primero: una descarga cortada no puede dejar un
        // peso a medias que la próxima vez parezca bueno.
        "    parcial = destino + '.parcial'\n" +
        "    task.update(message='descargando ' + etiqueta + '...')\n" +
        "    with urllib.request.urlopen(url) as r, open(parcial, 'wb') as f:\n" +
        "        total = int(r.headers.get('Content-Length') or 0)\n" +
        "        leido = 0\n" +
        "        ultimo = -1\n" +
        "        while True:\n" +
        "            trozo = r.read(1 << 20)\n" +
        "            if not trozo:\n" +
        "                break\n" +
        "            f.write(trozo)\n" +
        "            leido += len(trozo)\n" +
        "            if total:\n" +
        "                pct = 100 * leido // total\n" +
        "                if pct >= ultimo + 5:\n" +
        "                    ultimo = pct\n" +
        "                    task.update(message='%s %d%%' % (etiqueta, pct))\n" +
        "    os.replace(parcial, destino)\n" +
        "    hecho.append(etiqueta)\n";

    private static final String CIERRE_PREPARACION =
        "task.outputs['hecho'] = hecho\n";

    static List<String> modelos() {
        return new ArrayList<>(CATALOGO.keySet());
    }

    /** Lanza si no está en el catálogo; los llamantes sacan el nombre de modelos(). */
    private static Receta receta(String modelo) {
        Receta receta = CATALOGO.get(modelo);
        if (receta == null) throw new IllegalArgumentException("modelo desconocido: " + modelo);
        return receta;
    }

    /** Los checkpoints de ese modelo, el de por defecto primero. */
    static List<String> checkpointsDe(String modelo) {
        return receta(modelo).checkpoints;
    }

    /** El script que le consigue lo que le falte. Null si no necesita nada. */
    static String scriptPreparacion(String modelo) {
        Receta receta = receta(modelo);
        if (receta.preparacion.isEmpty()) return null;
        return CABECERA_PREPARACION + receta.preparacion + CIERRE_PREPARACION;
    }

    /** Si acepta prompt de máscara. Sin eso no hay M4. */
    static boolean encadena(String modelo) {
        return receta(modelo).encadena;
    }

    /** Si tiene banco de memoria de vídeo, condición para M3. */
    static boolean propaga(String modelo) {
        return receta(modelo).propaga;
    }

    /**
     * Si su entorno ya está en la caché de Appose, para avisar antes de lanzar
     * la instalación de varios GB de la primera vez.
     */
    static boolean yaInstalado(String modelo) {
        File raiz = new File(Environments.apposeEnvsDir());
        String nombre = receta(modelo).nombre;
        return new File(raiz, nombre + "-gpu").isDirectory()
            || new File(raiz, nombre + "-cpu").isDirectory();
    }

    /** El entorno que le toca a ese modelo, con CUDA si la máquina puede. */
    static Environment para(String modelo) throws IOException, BuildException {
        Receta receta = receta(modelo);

        if (hayGpuNvidia()) {
            String nombre = receta.nombre + "-gpu";
            try {
                IJ.log("-> GPU NVIDIA detectada: entorno '" + nombre + "'.");
                return construir(nombre, manifiesto(nombre, receta, true));
            } catch (BuildException e) {
                IJ.log("-> El entorno con CUDA no se pudo resolver, se usará CPU.");
                IJ.log("   Motivo: " + e.getMessage());
            }
        } else {
            IJ.log("-> Sin GPU NVIDIA visible: entorno de CPU.");
        }
        String nombre = receta.nombre + "-cpu";
        return construir(nombre, manifiesto(nombre, receta, false));
    }

    /**
     * Solo Python y appose, para separar un fallo del puente de uno del modelo.
     * Lo usa la prueba P1 de PRUEBAS/, no el plugin.
     */
    static Environment minimo(String nombre) throws IOException, BuildException {
        return construir(nombre, manifiesto(nombre, MINIMO, false));
    }

    /**
     * Un pixi.toml a mano y no un environment.yml: el yml no sabe expresar
     * [system-requirements] ni un índice de pip por paquete.
     */
    private static String manifiesto(String nombre, Receta receta, boolean cuda) {
        boolean pideCuda = cuda && !receta.condaGpu.isEmpty();
        return "[workspace]\n" +
            "channels = [\"conda-forge\"]\n" +
            "name = \"" + nombre + "\"\n" +
            "platforms = [\"" + plataformaPixi() + "\"]\n" +
            "version = \"0.1.0\"\n" +
            "\n" +
            (pideCuda ? "[system-requirements]\ncuda = \"12\"\n\n" : "") +
            "[dependencies]\n" +
            "python = \"3.12.*\"\n" +
            "pip = \"*\"\n" +
            receta.conda +
            (cuda ? receta.condaGpu : "") +
            "\n" +
            "[pypi-dependencies]\n" +
            String.format(receta.pypi, cuda ? INDICE_CU128 : "https://pypi.org/simple");
    }

    private static String plataformaPixi() {
        String so = System.getProperty("os.name", "").toLowerCase();
        String arq = System.getProperty("os.arch", "").toLowerCase();
        boolean brazo = arq.contains("aarch64") || arq.contains("arm");
        if (so.contains("win")) return "win-64";
        if (so.contains("mac") || so.contains("darwin")) return brazo ? "osx-arm64" : "osx-64";
        return brazo ? "linux-aarch64" : "linux-64";
    }

    /**
     * Si nvidia-smi responde, hay tarjeta y driver. No dice nada de si el torch
     * que se instale sabrá usarla: eso lo dice torch.cuda.get_arch_list().
     */
    private static Boolean gpu;

    private static boolean hayGpuNvidia() {
        if (gpu != null) return gpu;
        gpu = preguntarPorGpu();
        return gpu;
    }

    private static boolean preguntarPorGpu() {
        try {
            Process p = new ProcessBuilder("nvidia-smi", "-L")
                .redirectErrorStream(true).start();
            if (!p.waitFor(10, TimeUnit.SECONDS)) {
                p.destroy();
                return false;
            }
            return p.exitValue() == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

    /**
     * @param nombre     identifica el entorno en la caché de Appose; cambiarlo
     *                   fuerza una instalación nueva.
     * @param manifiesto contenido completo de un pixi.toml.
     */
    private static Environment construir(String nombre, String manifiesto)
        throws IOException, BuildException
    {
        File toml = File.createTempFile(nombre, ".toml");
        toml.deleteOnExit();
        try (PrintWriter w = new PrintWriter(toml, "UTF-8")) {
            w.print(manifiesto);
        }
        return Appose.file(toml).name(nombre).logDebug().build();
    }
}
