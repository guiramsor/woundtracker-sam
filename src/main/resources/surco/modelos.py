"""Interfaz comun de los modelos.

Un modelo recibe una imagen y unos clics y devuelve **candidatas**, no una
mascara. Esto no es un detalle: el factor selector del apartado 5.5 compara S0,
quedarse con la de mayor puntuacion del modelo, contra S1, el criterio propio.
Si la interfaz devolviera ya elegida una sola mascara, S0 quedaria cableado y el
factor no existiria. Reordenar candidatas ya calculadas es aritmetica y no
consume GPU, asi que se calculan una vez y se reordenan despues.

Los puntos van en coordenadas `(fila, columna)`, como todo lo demas en este
trabajo y como los guarda el fichero de clics. Cada adaptador se encarga de
darle la vuelta si su modelo espera `(x, y)`.

Aqui no hay ningun modelo real: solo el contrato y el registro. Los adaptadores
llegan de uno en uno, y cada uno declara lo que sabe hacer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from surco import config, datos


@dataclass(frozen=True)
class Candidatas:
    """Las mascaras que propone un modelo para un prompt, con su puntuacion.

    `mascaras` tiene forma (k, alto, ancho) y `puntuaciones` forma (k,). Un
    modelo que solo proponga una candidata usa k = 1, y entonces S0 y S1 dan lo
    mismo para el, que es informacion y no un problema.
    """

    mascaras: np.ndarray
    puntuaciones: np.ndarray

    def __post_init__(self) -> None:
        if self.mascaras.ndim != 3:
            raise ValueError(
                f"las mascaras tienen que ser (k, alto, ancho), llego {self.mascaras.shape}"
            )
        if self.mascaras.dtype != np.bool_:
            raise ValueError(f"las mascaras tienen que ser booleanas, son {self.mascaras.dtype}")
        if self.puntuaciones.shape != (len(self.mascaras),):
            raise ValueError(
                f"hacen falta {len(self.mascaras)} puntuaciones, "
                f"llegaron {self.puntuaciones.shape}"
            )

    def __len__(self) -> int:
        return len(self.mascaras)

    def mejor(self) -> np.ndarray:
        """La de mayor puntuacion: el selector S0, el comportamiento por defecto."""
        return self.mascaras[int(np.argmax(self.puntuaciones))]


class Modelo(ABC):
    """Contrato que cumple cada adaptador.

    `acepta_negativos` no excluye a nadie: un modelo que no los admita compite
    en los protocolos que no los usan, y se declara (apartado 5.2).

    Un adaptador cubre **todos los checkpoints** de su modelo, no uno. Entran
    todos los tamanos publicados, y la agregacion por familia se queda con un
    representante por modelo (apartado 6.3), asi que la clase tiene que poder
    instanciarse en cualquiera de ellos.
    """

    nombre: str
    familia: str
    acepta_negativos: bool
    checkpoints: tuple[str, ...]
    checkpoint_por_defecto: str
    # Lado al que el modelo reescala la entrada, leido de su codigo al
    # instalarlo. No es documentacion: con un surco de unos 17 px de ancho sobre
    # 1024, a 512 quedan 8 y a 256 quedan 4, asi que es la variable con la que
    # hay que contrastar el orden de la tabla antes de atribuirlo al dominio.
    # Puede depender del checkpoint, y entonces se declara como propiedad.
    lado_entrada: int
    # El entorno conda donde este modelo esta instalado. Con uno por modelo
    # (apartado 11), un adaptador solo se puede ejecutar en el suyo. Se declara
    # aqui y no en un mapa aparte para que anadir un adaptador no obligue a
    # acordarse de tocar dos sitios.
    entorno: str

    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint or self.checkpoint_por_defecto
        if self.checkpoint not in self.checkpoints:
            raise ValueError(
                f"{self.nombre} no tiene el checkpoint {self.checkpoint!r}; "
                f"los que hay son {self.checkpoints}"
            )

    @property
    def identificador(self) -> str:
        """Como se llama esta fila en las tablas: modelo y checkpoint."""
        return f"{self.nombre}:{self.checkpoint}"

    @property
    def es_por_defecto(self) -> bool:
        """Si es el checkpoint que se mide en la primera pasada del cribado."""
        return self.checkpoint == self.checkpoint_por_defecto

    @abstractmethod
    def cargar(self) -> None:
        """Trae los pesos a memoria. Separado del constructor a proposito.

        Instanciar tiene que ser barato para poder listar modelos sin pagar la
        carga; el coste se mide en F7 y no se quiere mezclado con esto.
        """

    @abstractmethod
    def segmentar(
        self,
        imagen_rgb: np.ndarray,
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
    ) -> Candidatas:
        """Segmenta con esos clics.

        `imagen_rgb` llega **ya preparada** por `preparar_frame`, en uint8 y con
        tres bandas iguales: el adaptador no la toca. `puntos` va en
        `(fila, columna)` y `etiquetas` es 1 para positivo y 0 para fondo.
        """

    def hay_que_codificar(self, imagen_rgb: np.ndarray) -> bool:
        """Si esta imagen no es la ultima que se codifico, hay que codificarla.

        Los modelos con `set_image` calculan el embedding una vez y lo pueden
        reutilizar para prompts sucesivos sobre el mismo frame. Sin esta cache,
        `segmentar` recodificaria en cada llamada y las dos columnas de coste
        del apartado 6.6 medirian lo mismo en todos: la separacion dejaria de
        separar, y la tabla diria que un prompt adicional es caro en todo el
        conjunto cuando solo lo es en los modelos que no tienen `set_image`.

        Primero se compara por identidad, que es el caso normal cuando la
        rejilla pasa el mismo array a los cuatro protocolos, y despues por
        contenido, que cuesta menos de un milisegundo frente a decenas o
        cientos de codificar. Nunca al reves: reutilizar de mas seria devolver
        una mascara de otra imagen.
        """
        anterior = getattr(self, "_ultima_imagen", None)
        if anterior is imagen_rgb:
            return False
        if (
            anterior is not None
            and anterior.shape == imagen_rgb.shape
            and np.array_equal(anterior, imagen_rgb)
        ):
            return False
        self._ultima_imagen = imagen_rgb
        return True

    def olvidar_imagen(self) -> None:
        """Invalida la cache. Se llama al cargar, por si se recarga el modelo."""
        self._ultima_imagen = None

    @property
    def soporta_memoria(self) -> bool:
        """Si este modelo puede hacer M3, propagacion con banco de memoria."""
        return isinstance(self, ModeloConMemoria)

    @property
    def soporta_mascara(self) -> bool:
        """Si este modelo puede hacer M4, encadenado de mascara."""
        return isinstance(self, ModeloConMascara)

    def modos_admitidos(self) -> tuple[str, ...]:
        """Los modos en los que este modelo puede competir.

        M0 y M1 son obligatorios para todos. M3 solo existe para quien tenga
        banco de memoria y M4 para quien acepte prompt de mascara, que es lo que
        dice el apartado 5.1: esas comparaciones viven dentro de los modelos que
        las soportan y no cruzan la tabla.
        """
        capacidad = {"M3": self.soporta_memoria, "M4": self.soporta_mascara}
        return tuple(modo for modo in config.MODOS if capacidad.get(modo, True))

    def protocolos_admitidos(self) -> tuple[str, ...]:
        """Los protocolos en los que este modelo puede competir.

        Sale de los propios protocolos y de que indices son negativos, no de una
        lista escrita a mano: si mañana cambiara la composicion de P1 a P4, esto
        seguiria diciendo la verdad.
        """
        if self.acepta_negativos:
            return tuple(config.PROTOCOLOS)
        negativos = set(config.CLICS_NEGATIVOS)
        return tuple(
            nombre
            for nombre, indices in config.PROTOCOLOS.items()
            if not negativos & set(indices)
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.identificador} ({self.familia})>"


class ModeloConMemoria(Modelo):
    """Modelos con banco de memoria: los unicos que pueden hacer M3.

    La propagacion es una **segunda operacion**, no un parametro mas de
    `segmentar`. Son cosas distintas: `segmentar` resuelve un frame suelto a
    partir de unos clics, y `propagar` recorre una secuencia arrastrando
    memoria. Meterlas en la misma firma obligaria a los once modelos sin
    memoria a cargar con parametros que ignorarian.

    Que un modelo pueda hacer M3 se sabe por herencia y no por un atributo, asi
    que no se puede declarar mal: o implementa `propagar` o no esta en el nivel.
    """

    @abstractmethod
    def propagar(
        self,
        imagenes_rgb: list[np.ndarray],
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
        indice_semilla: int,
    ) -> list[Candidatas]:
        """Propaga por la secuencia a partir de los clics puestos en la semilla.

        `imagenes_rgb` va en orden temporal y ya preparada. `indice_semilla` es
        la posicion dentro de esa lista del frame donde se ponen los clics, que
        es el `start` de la unidad.

        Devuelve una entrada por frame de entrada, en el mismo orden. La del
        propio frame semilla tambien sale, porque el modelo la produce; es el
        apartado 6.2 quien la deja fuera de la evaluacion, y esa decision es de
        quien llama, no del adaptador.
        """


class ModeloConMascara(Modelo):
    """Modelos que aceptan prompt de mascara: los que pueden hacer M4.

    Tercera operacion del contrato, por el mismo motivo que `propagar` fue la
    segunda: `segmentar` resuelve un frame suelto y no tiene por que cargar con
    un parametro que la mayoria de modelos ignoraria.

    **Lo que se encadena son los logits de baja resolucion, no la mascara
    binaria.** El `mask_input` de la familia de SAM espera exactamente lo que
    devuelve su propio `predict` como tercer valor, y darle en su lugar una
    mascara booleana reescalada obligaria a inventar una escala para convertir
    booleano en logit. Por eso esta operacion devuelve las dos cosas: las
    candidatas, para elegir y medir, y sus logits, para volver a entrar.
    """

    @abstractmethod
    def segmentar_con_mascara(
        self,
        imagen_rgb: np.ndarray,
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
        mascara_previa: np.ndarray | None = None,
    ) -> tuple[Candidatas, np.ndarray]:
        """Segmenta con los clics y, si la hay, con la mascara del frame anterior.

        `mascara_previa` son los logits de **una** candidata, la que eligio el
        selector activo en el frame anterior, con la forma que devuelve este
        mismo metodo. Con `None` el resultado es el de `segmentar`, que es como
        M4 resuelve su frame semilla: alli no hay mascara previa todavia.

        Devuelve las candidatas y los logits de todas ellas, en el mismo orden.
        """


def encadenar_por_predict(
    modelo: Modelo,
    predictor,
    imagen_rgb: np.ndarray,
    puntos: list[tuple[int, int]],
    etiquetas: list[int],
    mascara_previa: np.ndarray | None,
    multimask: bool = True,
) -> tuple[Candidatas, np.ndarray]:
    """M4 para los predictores de la familia de SAM, que comparten `predict`.

    Vive aqui y no repetido en cada adaptador porque los cuatro modelos de la
    rejilla llaman al mismo metodo con la misma firma; lo unico que cambia entre
    ellos es si declaran `multimask_output` y cuantas candidatas devuelven.

    **No toca `segmentar`.** Es una llamada aparte con su propio camino, para
    que anadir M4 no pueda mover ni un decimal de lo que ya se midio en M0, M1
    y M3.
    """
    if predictor is None:
        raise RuntimeError("hay que llamar a cargar() antes de segmentar")
    if modelo.hay_que_codificar(imagen_rgb):
        predictor.set_image(imagen_rgb)
    # Aqui todo va en (fila, columna); SAM espera (x, y) = (columna, fila).
    coordenadas = np.array([[col, fila] for fila, col in puntos], dtype=float)
    extra = {"multimask_output": True} if multimask else {}
    if mascara_previa is not None:
        extra["mask_input"] = np.asarray(mascara_previa)[None]
    mascaras, puntuaciones, logits = predictor.predict(
        point_coords=coordenadas,
        point_labels=np.array(etiquetas, dtype=int),
        **extra,
    )
    mascaras = np.asarray(mascaras)
    if mascaras.ndim == 2:  # una sola candidata, sin eje de mascara
        mascaras = mascaras[None]
    candidatas = Candidatas(
        mascaras.astype(bool), np.atleast_1d(np.asarray(puntuaciones, dtype=float))
    )
    return candidatas, np.asarray(logits, dtype=np.float32)


# Calidad del JPEG intermedio de M3 en la familia de SAM 2. Su predictor de
# video solo admite carpeta de JPEG o MP4; 100 y sin submuestreo de croma es lo
# mas cerca del array que permite el formato, y la perdida que queda esta medida
# en `scripts/unicos/medir_perdida_jpeg.py`.
CALIDAD_JPEG = 100


def propagar_por_jpeg(
    video,
    imagenes_rgb: list[np.ndarray],
    puntos: list[tuple[int, int]],
    etiquetas: list[int],
    indice_semilla: int,
) -> list[Candidatas]:
    """Propagacion M3 para los modelos de la familia de SAM 2.

    Vive aqui y no en cada adaptador porque SAM 2.1 y MedSAM2 son el mismo
    modelo y su fork: comparten predictor de video, y duplicar esto seria
    invitar a que los dos caminos se separen sin que nadie se entere.

    Devuelve una entrada por frame de entrada. Hacia delante desde la semilla y,
    si la semilla no es el primer frame, tambien hacia atras.

    Una sola candidata por frame: el predictor de video no expone puntuacion por
    candidata, asi que en M3 el factor selector es degenerado y S0 y S1
    coinciden.
    """
    import tempfile
    from pathlib import Path

    from PIL import Image

    def recorrer(estado, reverse: bool) -> dict[int, np.ndarray]:
        salidas = {}
        for indice, _, logits in video.propagate_in_video(
            estado, start_frame_idx=indice_semilla, reverse=reverse
        ):
            salidas[int(indice)] = (logits[0] > 0).squeeze(0).cpu().numpy()
        return salidas

    with tempfile.TemporaryDirectory(prefix="surco_m3_") as temporal:
        carpeta = Path(temporal)
        for indice, imagen in enumerate(imagenes_rgb):
            Image.fromarray(imagen).save(
                carpeta / f"{indice:05d}.jpg", quality=CALIDAD_JPEG, subsampling=0
            )
        estado = video.init_state(video_path=str(carpeta))
        video.add_new_points_or_box(
            inference_state=estado,
            frame_idx=indice_semilla,
            obj_id=1,
            points=np.array([[col, fila] for fila, col in puntos], dtype=np.float32),
            labels=np.array(etiquetas, dtype=np.int32),
        )
        mascaras = recorrer(estado, reverse=False)
        if indice_semilla > 0:
            mascaras.update(recorrer(estado, reverse=True))

    alto, ancho = imagenes_rgb[0].shape[:2]
    vacia = np.zeros((alto, ancho), dtype=bool)
    return [
        Candidatas(mascaras.get(indice, vacia)[None].astype(bool), np.array([1.0]))
        for indice in range(len(imagenes_rgb))
    ]


_REGISTRO: dict[str, type[Modelo]] = {}


def registrar(clase: type[Modelo]) -> type[Modelo]:
    """Decorador que apunta un adaptador en el registro."""
    if clase.familia not in config.FAMILIAS:
        raise ValueError(f"familia desconocida: {clase.familia!r}")
    if clase.nombre in _REGISTRO:
        raise ValueError(f"ya hay un modelo llamado {clase.nombre!r}")
    if clase.checkpoint_por_defecto not in clase.checkpoints:
        raise ValueError(
            f"{clase.nombre}: el checkpoint por defecto "
            f"{clase.checkpoint_por_defecto!r} no esta entre los suyos"
        )
    if not getattr(clase, "entorno", ""):
        raise ValueError(f"{clase.nombre}: falta declarar su entorno")
    _REGISTRO[clase.nombre] = clase
    return clase


def registrados() -> dict[str, type[Modelo]]:
    return dict(_REGISTRO)


def crear(nombre: str, checkpoint: str | None = None) -> Modelo:
    if nombre not in _REGISTRO:
        raise KeyError(f"no hay ningun adaptador para {nombre!r}")
    return _REGISTRO[nombre](checkpoint)


def filas(
    solo_por_defecto: bool = False, entorno: str | None = None
) -> list[tuple[str, str]]:
    """Los pares (modelo, checkpoint) que hay que medir, en orden.

    Con `solo_por_defecto` salen los de la primera pasada del cribado: el
    apartado 5.2 manda medir el checkpoint por defecto de todos los modelos
    antes de pasar al resto de tamanos, para tener el ranking por familia
    completo cuanto antes.

    Con `entorno` salen solo los modelos instalados ahi. Sin ese filtro, un
    informe de disponibilidad recoge tambien modelos que en ese entorno no
    existen, y como la tabla final es la union de los informes, esas filas
    ensucian con fallos que no significan nada. El caso claro es TinySAM, que no
    se instala sino que entra por `sys.path` y por tanto es importable desde
    todos los entornos.
    """
    pares = []
    for nombre, clase in sorted(_REGISTRO.items()):
        if entorno is not None and clase.entorno != entorno:
            continue
        if solo_por_defecto:
            pares.append((nombre, clase.checkpoint_por_defecto))
        else:
            pares.extend((nombre, punto) for punto in clase.checkpoints)
    return pares


@lru_cache(maxsize=None)
def limites_de_serie(serie: int) -> tuple[float, float]:
    """Punto negro y blanco comunes a los 21 frames de una serie.

    **Por serie, no por imagen.** Normalizar cada frame por su cuenta borraria
    la diferencia de dificultad entre los frames debiles y los brillantes, que
    es justo lo que el modelo tiene que enfrentar, y en un frame con poca senal
    estiraria el ruido de fondo hasta el rango entero. Es ademas lo que hara el
    plugin: el usuario abre una secuencia y se procesa entera.

    Y tampoco global, porque mezclaria condiciones de adquisicion distintas: la
    serie 7 esta bastante mas brillante que las demas.
    """
    apilados = np.stack(
        [datos.cargar_frame(serie, frame) for frame in range(1, config.N_FRAMES + 1)]
    )
    return (
        float(apilados.min()),
        float(np.percentile(apilados, config.PERCENTIL_BLANCO_MODELOS)),
    )


def preparar(imagen: np.ndarray, limites: tuple[float, float]) -> np.ndarray:
    """Pasa un frame uint16 a las tres bandas uint8 que esperan los modelos.

    La preparacion vive aqui y no en cada adaptador, y por eso `segmentar`
    recibe la imagen **ya preparada**: asi todos los modelos ven exactamente lo
    mismo por construccion y no por convenio. Si cada uno preparara la suya, la
    comparativa dejaria de comparar modelos y pasaria a comparar preprocesados.
    """
    negro, blanco = limites
    escalada = np.clip(
        (imagen.astype(np.float32) - negro) / max(blanco - negro, 1.0), 0, 1
    )
    ocho = (escalada * 255).round().astype(np.uint8)
    return np.repeat(ocho[:, :, None], 3, axis=2)


def preparar_frame(serie: int, frame: int) -> np.ndarray:
    """La entrada exacta que recibe cualquier modelo para ese par."""
    return preparar(datos.cargar_frame(serie, frame), limites_de_serie(serie))


def imagen_de_prueba(lado: int | None = None) -> np.ndarray:
    """Imagen sintetica con una barra clara, para comprobar que un modelo carga.

    No sirve para medir nada: solo para ver que el modelo se instancia, traga un
    punto y devuelve una mascara de la forma correcta.
    """
    lado = lado or config.LADO_PX
    imagen = np.full((lado, lado), 400, dtype=np.uint16)
    alto, largo = max(1, lado // 100), max(2, lado // 8)
    fila, columna = lado // 2, lado // 2
    imagen[fila : fila + alto, columna - largo : columna + largo] = 3000
    return imagen


def imagen_de_prueba_rgb(lado: int | None = None) -> np.ndarray:
    """La imagen de prueba ya preparada, tal como la recibiria un modelo."""
    imagen = imagen_de_prueba(lado)
    blanco = float(np.percentile(imagen, config.PERCENTIL_BLANCO_MODELOS))
    return preparar(imagen, (float(imagen.min()), blanco))


def secuencia_de_prueba(n: int = 3, lado: int | None = None) -> list[np.ndarray]:
    """Una secuencia corta y sintetica, para comprobar la propagacion de M3."""
    return [imagen_de_prueba_rgb(lado) for _ in range(n)]


def punto_de_prueba(lado: int | None = None) -> tuple[int, int]:
    """El centro de la barra de `imagen_de_prueba`."""
    lado = lado or config.LADO_PX
    return lado // 2, lado // 2
